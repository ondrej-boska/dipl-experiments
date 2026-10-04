"""
Unified Dataset Loader for single-staff OMR.

Follows the Zeus repository workflow:
- Ingests datasets from Zeus pickled slices (ZeusDatasetSample).
- Run 1 (Zeus Baseline): decodes raw image bytes in-memory with aspect-ratio preserving scaling.
- Run 2 (MuSViT + Zeus): loads pre-extracted feature embeddings from feature_cache.
- Ground truth: tokenized directly from sample.lmx strings.
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from musvit_zeus_comparison.zeus.data.zeus_dataset import ZeusDataset, ZeusDatasetSample
except ImportError:
    from zeus.data.zeus_dataset import ZeusDataset, ZeusDatasetSample
except ImportError:
    # Standalone dataclass fallback if zeus package is accessed without editable install
    @dataclass
    class ZeusDatasetSample:  # type: ignore
        sample_name: str
        image: bytes
        lmx: str

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if name == "ZeusDatasetSample":
                return ZeusDatasetSample
            return super().find_class(module, name)

    class ZeusDataset:  # type: ignore
        def __init__(self, name: str, samples: list[ZeusDatasetSample]):
            self.name = name
            self.samples = samples

        @staticmethod
        def load_from_pickle_file(pickle_path: Path) -> ZeusDataset:
            with open(str(pickle_path), "rb") as f:
                samples = _Unpickler(f).load()
            return ZeusDataset(name=pickle_path.as_posix(), samples=samples)

        @staticmethod
        def combine_multiple(datasets: list[ZeusDataset]) -> ZeusDataset:
            combined = [s for d in datasets for s in d.samples]
            name = "+".join(d.name for d in datasets)
            return ZeusDataset(name=name, samples=combined)


# ==============================================================================
# Token Vocabulary
# ==============================================================================

class TokenVocabulary:
    """Manages string token to integer ID mapping, mirroring Zeus TokenMap."""
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
    def load(cls, filepath: str | Path) -> TokenVocabulary:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(special_tokens=data.get("special_tokens"))
        vocab.token2id = data["token2id"]
        vocab.id2token = {int(v): k for k, v in vocab.token2id.items()}
        return vocab

    @classmethod
    def build_from_samples(cls, samples: list[ZeusDatasetSample]) -> TokenVocabulary:
        """Constructs vocabulary from all LMX tokens present in training samples."""
        vocab = cls()
        for sample in samples:
            for token in sample.lmx.split():
                vocab.add_token(token)
        return vocab


# ==============================================================================
# Helper to Load and Combine Pickled Slices
# ==============================================================================

def load_zeus_pickles(pickle_paths: list[str | Path]) -> list[ZeusDatasetSample]:
    """Loads one or multiple Zeus dataset pickle slices and returns combined samples."""
    datasets = []
    for p in pickle_paths:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"Zeus dataset pickle not found at: '{path}'")
        ds = ZeusDataset.load_from_pickle_file(path)
        print(f"Loaded {len(ds.samples):,} samples from '{path}'")
        datasets.append(ds)

    if not datasets:
        return []

    combined = ZeusDataset.combine_multiple(datasets)
    return combined.samples


# ==============================================================================
# Single-Staff OMR Dataset
# ==============================================================================

class StaveOMRDataset(Dataset):
    """
    Dataset backed directly by in-memory ZeusDatasetSample objects.
    - Run 1 ('zeus'): decodes sample.image bytes into normalized grayscale tensor (1, H, W).
    - Run 2 ('musvit'): loads pre-extracted feature embedding from feature_cache_dir.
    - Targets: tokenized from sample.lmx string.
    """
    def __init__(
        self,
        samples: list[ZeusDatasetSample],
        vocab: TokenVocabulary,
        mode: str = "zeus",  # 'zeus' or 'musvit'
        feature_cache_dir: str | Path | None = None,
        image_height: int = 96,  # Zeus single-staff standard height
        max_image_width: int = 1536,
    ):
        self.samples = samples
        self.vocab = vocab
        self.mode = mode.lower()
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.image_height = image_height
        self.max_image_width = max_image_width

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_bytes: bytes) -> torch.Tensor:
        """Loads and normalizes image for Zeus (Run 1). Height is fixed, width preserves aspect ratio."""
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.ones((self.image_height, 256), dtype=np.uint8) * 255

        h, w = img.shape
        new_w = max(1, int(round(w * self.image_height / h)))
        if self.max_image_width is not None:
            new_w = min(new_w, self.max_image_width)

        resized = cv2.resize(img, (new_w, self.image_height), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
        return tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        tokens = sample.lmx.split()
        token_tensor = torch.tensor(self.vocab.encode(tokens), dtype=torch.long)

        if self.mode == "musvit":
            if self.feature_cache_dir is None:
                raise ValueError("feature_cache_dir must be specified for mode='musvit'.")

            # Canonical slug: e.g. 'samples_chopin_op17_Staves_0'
            sample_slug = sample.sample_name.replace("/", "_").replace("\\", "_")
            feat_path = self.feature_cache_dir / f"{sample_slug}_vertical_mean_float16.pt"
            if not feat_path.is_file() and sample_slug.startswith("samples_"):
                # Fallback without 'samples_' prefix
                alt_path = self.feature_cache_dir / f"{sample_slug[8:]}_vertical_mean_float16.pt"
                if alt_path.is_file():
                    feat_path = alt_path

            if feat_path.is_file():
                feat = torch.load(feat_path, map_location="cpu", weights_only=True)
                return feat.float(), token_tensor
            else:
                raise FileNotFoundError(
                    f"Pre-extracted MuSViT feature file not found for sample '{sample.sample_name}'. "
                    f"Looked at: '{feat_path}'. Please run extract_features.py before training MuSViT."
                )

        else:
            # Mode 'zeus': decode image bytes
            img_tensor = self._load_image(sample.image)
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
                if pad_w > 0:
                    x = torch.nn.functional.pad(x, (0, pad_w, 0, 0), value=1.0)
                padded_inputs.append(x)
            batch_inputs = torch.stack(padded_inputs, dim=0)

        else:
            # Feature sequences: (T, 768)
            max_t = max(x.shape[0] for x in inputs)
            padded_inputs = []
            for x in inputs:
                pad_t = max_t - x.shape[0]
                if pad_t > 0:
                    x = torch.nn.functional.pad(x, (0, 0, 0, pad_t), value=0.0)
                padded_inputs.append(x)
            batch_inputs = torch.stack(padded_inputs, dim=0)

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
