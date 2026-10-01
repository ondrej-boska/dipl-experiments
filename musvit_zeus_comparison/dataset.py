"""
Unified Dataset Loader for single-staff OMR.
Supports loading raw images for Run 1 (Zeus) and pre-extracted features for Run 2 (MuSViT).
Handles token sequence parsing from LMX or MusicXML, vocabulary construction, and batch collation.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ==============================================================================
# Simple Tokenizer & MusicXML Token Extractor
# ==============================================================================

class TokenVocabulary:
    """Manages string token to integer ID mapping."""
    def __init__(self, special_tokens: list[str] | None = None):
        self.special_tokens = special_tokens or ["<bos>", "<eos>", "<pad>", "<unk>"]
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

        for tok in self.special_tokens:
            self.add_token(tok)

    @property
    def bos_idx(self) -> int:
        return self.token2id[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self.token2id[self.eos_token]

    @property
    def pad_idx(self) -> int:
        return self.token2id[self.pad_token]

    @property
    def unk_idx(self) -> int:
        return self.token2id[self.unk_token]

    def add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            return idx
        return self.token2id[token]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token2id.get(t, self.unk_idx) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id2token.get(i, self.unk_token) for i in ids]

    def __len__(self) -> int:
        return len(self.token2id)

    def save(self, filepath: str | Path):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "token2id": self.token2id,
                "special_tokens": self.special_tokens
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> "TokenVocabulary":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(special_tokens=data.get("special_tokens"))
        vocab.token2id = data["token2id"]
        vocab.id2token = {int(v): k for k, v in vocab.token2id.items()}
        return vocab


def extract_tokens_from_musicxml(musicxml_path: str | Path) -> list[str]:
    """
    Extracts high-level sequential musical tokens from a MusicXML file
    when raw LMX files are not pre-generated.
    Covers clefs, keys, time signatures, barlines, note pitches, and durations.
    """
    try:
        tree = ET.parse(musicxml_path)
        root = tree.getroot()
    except Exception:
        return []

    tokens = []
    # Search measures
    for measure in root.iter("measure"):
        tokens.append("measure")
        # Clef
        for clef in measure.iter("clef"):
            sign = clef.findtext("sign", "G")
            line = clef.findtext("line", "2")
            tokens.append(f"clef:{sign}{line}")
        # Key
        for key in measure.iter("key"):
            fifths = key.findtext("fifths", "0")
            tokens.append(f"key:fifths:{fifths}")
        # Time
        for time in measure.iter("time"):
            beats = time.findtext("beats", "4")
            beat_type = time.findtext("beat-type", "4")
            tokens.append(f"time:{beats}/{beat_type}")
        # Notes & Rests
        for note in measure.iter("note"):
            if note.find("rest") is not None:
                dur_type = note.findtext("type", "quarter")
                tokens.append(f"rest:{dur_type}")
            elif note.find("pitch") is not None:
                step = note.findtext("pitch/step", "C")
                octave = note.findtext("pitch/octave", "4")
                alter = note.findtext("pitch/alter", "0")
                dur_type = note.findtext("type", "quarter")
                alter_str = "#" if alter == "1" else ("b" if alter == "-1" else "")
                tokens.append(f"note:{step}{alter_str}{octave}:{dur_type}")

    return tokens


# ==============================================================================
# Single-Staff OMR Dataset
# ==============================================================================

class StaveOMRDataset(Dataset):
    """
    Dataset that provides both:
    - Raw image (for Run 1: ZeusEncoder)
    - Pre-extracted MuSViT features (for Run 2: MusvitEncoder)
    Along with target token sequences.
    """
    def __init__(
        self,
        samples: list[dict],
        vocab: TokenVocabulary,
        mode: str = "zeus",  # 'zeus' or 'musvit'
        image_height: int = 96,  # Zeus single-staff default height
        max_image_width: int = 1536,
    ):
        self.samples = samples
        self.vocab = vocab
        self.mode = mode.lower()
        self.image_height = image_height
        self.max_image_width = max_image_width

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, img_path: Path) -> torch.Tensor:
        """Loads and normalizes image for Zeus (Run 1). Height is fixed, width preserves aspect ratio."""
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback blank image
            img = np.ones((self.image_height, 256), dtype=np.uint8) * 255

        h, w = img.shape
        new_w = max(1, int(round(w * self.image_height / h)))
        if self.max_image_width is not None:
            new_w = min(new_w, self.max_image_width)

        resized = cv2.resize(img, (new_w, self.image_height), interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1] float tensor of shape (1, H, W)
        tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
        return tensor

    def _load_tokens(self, sample: dict) -> list[int]:
        # 1. Check for .lmx file
        if "lmx_path" in sample and Path(sample["lmx_path"]).exists():
            lmx_text = Path(sample["lmx_path"]).read_text(encoding="utf-8").strip()
            raw_tokens = lmx_text.split()
        # 2. Check for .musicxml file
        elif "musicxml_path" in sample and Path(sample["musicxml_path"]).exists():
            raw_tokens = extract_tokens_from_musicxml(sample["musicxml_path"])
        else:
            raw_tokens = ["measure", "rest:quarter"]

        return self.vocab.encode(raw_tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        tokens = self._load_tokens(sample)
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        if self.mode == "musvit":
            feat_path = Path(sample.get("feature_path", ""))
            if feat_path.exists():
                feat = torch.load(feat_path, map_location="cpu", weights_only=True)
                # Convert to float32 for training stability
                return feat.float(), token_tensor
            else:
                # If feature is missing, return a dummy placeholder tensor
                return torch.zeros((64, 768), dtype=torch.float32), token_tensor

        else:
            # Mode 'zeus': load normalized 2D image
            img_path = Path(sample["image_path"])
            img_tensor = self._load_image(img_path)
            return img_tensor, token_tensor


# ==============================================================================
# Collate Function
# ==============================================================================

class StaveCollate:
    """Collates variable-width images or feature sequences and variable-length token targets."""
    def __init__(self, pad_idx: int = 2, bos_idx: int = 0, eos_idx: int = 1):
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, targets = zip(*batch)

        # Pad inputs (either 3D images [1, H, W] or 2D feature sequences [T, D])
        is_image = (inputs[0].dim() == 3)

        if is_image:
            max_w = max(x.shape[2] for x in inputs)
            padded_inputs = []
            for x in inputs:
                pad_w = max_w - x.shape[2]
                # Pad right with 1.0 (white background)
                if pad_w > 0:
                    x = torch.nn.functional.pad(x, (0, pad_w, 0, 0), value=1.0)
                padded_inputs.append(x)
            batch_inputs = torch.stack(padded_inputs, dim=0)  # (B, 1, H, max_W)

        else:
            # Feature sequences: (T, 768)
            max_t = max(x.shape[0] for x in inputs)
            padded_inputs = []
            for x in inputs:
                pad_t = max_t - x.shape[0]
                if pad_t > 0:
                    x = torch.nn.functional.pad(x, (0, 0, 0, pad_t), value=0.0)
                padded_inputs.append(x)
            batch_inputs = torch.stack(padded_inputs, dim=0)  # (B, max_T, 768)

        # Pad target sequences
        # Teacher forcing inputs: [BOS, t_1, t_2, ..., t_L]
        # Target labels for loss: [t_1, t_2, ..., t_L, EOS]
        max_len = max(len(t) for t in targets)
        batch_input_seqs = []
        batch_target_labels = []

        for t in targets:
            inp = torch.cat([torch.tensor([self.bos_idx]), t])
            lbl = torch.cat([t, torch.tensor([self.eos_idx])])

            pad_len = (max_len + 1) - len(inp)
            if pad_len > 0:
                inp = torch.nn.functional.pad(inp, (0, pad_len), value=self.pad_idx)
                lbl = torch.nn.functional.pad(lbl, (0, pad_len), value=self.pad_idx)

            batch_input_seqs.append(inp)
            batch_target_labels.append(lbl)

        batch_input_seqs = torch.stack(batch_input_seqs, dim=0)
        batch_target_labels = torch.stack(batch_target_labels, dim=0)

        return batch_inputs, batch_input_seqs, batch_target_labels
