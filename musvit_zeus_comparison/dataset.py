"""
Unified Dataset Loader for single-staff OMR.

Follows the Zeus repository workflow:
- Datasets are subfolders of a common directory (e.g. datasets/omniomr, datasets/dolores),
  each with its own predefined splits pickled by Zeus (samples.{train,dev,test}.pickle).
- Run 1 (Zeus Baseline): decodes raw image bytes in-memory with aspect-ratio preserving scaling.
- Run 2 (MuSViT + Zeus): loads pre-extracted feature embeddings from feature_cache.
- Ground truth: tokenized directly from sample.lmx strings.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

try:
    from musvit_zeus_comparison.zeus.data.zeus_dataset import ZeusDataset, ZeusDatasetSample
except ImportError:
    from zeus.data.zeus_dataset import ZeusDataset, ZeusDatasetSample


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

    def to_dict(self) -> dict:
        return {"token2id": self.token2id, "special_tokens": self.special_tokens}

    @classmethod
    def from_dict(cls, data: dict) -> TokenVocabulary:
        # Older checkpoints stored the bare token2id mapping
        if "token2id" not in data:
            data = {"token2id": data}
        vocab = cls(special_tokens=data.get("special_tokens"))
        vocab.token2id = dict(data["token2id"])
        vocab.id2token = {int(v): k for k, v in vocab.token2id.items()}
        return vocab

    def save(self, filepath: str | Path):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> TokenVocabulary:
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def build_from_samples(cls, samples: list[ZeusDatasetSample]) -> TokenVocabulary:
        """Constructs vocabulary from all LMX tokens present in training samples."""
        vocab = cls()
        for sample in samples:
            for token in sample.lmx.split():
                vocab.add_token(token)
        return vocab


# ==============================================================================
# Dataset Selection and Loading of Pickled Splits
# ==============================================================================

SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "dev": ("dev", "val", "validation"),
    "test": ("test",),
}

SplitSamples = dict[str, list[ZeusDatasetSample]]
"""Samples of one split, keyed by the name of the dataset they come from."""


def discover_datasets(dataset_dir: str | Path) -> list[str]:
    """Names of all subfolders of `dataset_dir` that contain a pickled Zeus split."""
    base = Path(dataset_dir)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and any(p.glob("samples.*.pickle")))


def dataset_name(pickle_path: Path) -> str:
    """A dataset is named after the folder holding its pickles, e.g. 'omniomr'."""
    return pickle_path.parent.name


def resolve_dataset_folder(spec: str | Path, dataset_dir: str | Path) -> Path:
    """Resolves a dataset name (subfolder of `dataset_dir`) or a path to a dataset folder."""
    folder = Path(dataset_dir) / spec
    if folder.is_dir():
        return folder
    if Path(spec).is_dir():
        return Path(spec)
    available = ", ".join(discover_datasets(dataset_dir)) or "none"
    raise FileNotFoundError(
        f"Dataset '{spec}' not found. Datasets available in '{dataset_dir}': {available}"
    )


def resolve_split_pickle(spec: str | Path, dataset_dir: str | Path, split: str) -> Path:
    """
    Resolves a dataset specification to the pickle of its predefined `split`.
    `spec` is a dataset name, a path to a dataset folder, or a path to a .pickle file (used as-is).
    """
    if Path(spec).suffix == ".pickle":
        if not Path(spec).is_file():
            raise FileNotFoundError(f"Zeus dataset pickle not found at: '{spec}'")
        return Path(spec)

    folder = resolve_dataset_folder(spec, dataset_dir)
    for alias in SPLIT_ALIASES[split]:
        candidate = folder / f"samples.{alias}.pickle"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Dataset '{spec}' has no pickled '{split}' split in '{folder}'. "
        f"Run: python -m musvit_zeus_comparison.zeus pickle {folder}/samples.{split}.txt"
    )


def load_split(specs: list[str], dataset_dir: str | Path, split: str, keep_images: bool = True) -> SplitSamples:
    """
    Loads the predefined `split` of each selected dataset.
    Without `keep_images`, the image bytes are dropped after loading, to save memory when only
    pre-extracted features are used.
    """
    loaded: SplitSamples = {}
    for spec in specs:
        pickle_path = resolve_split_pickle(spec, dataset_dir, split)
        name = dataset_name(pickle_path)
        if name in loaded:
            raise ValueError(f"Dataset '{name}' is selected more than once for the '{split}' split.")
        samples = ZeusDataset.load_from_pickle_file(pickle_path).samples
        if not keep_images:
            samples = [ZeusDatasetSample(sample_name=s.sample_name, image=b"", lmx=s.lmx) for s in samples]
        loaded[name] = samples
        print(f"Loaded {len(samples):,} {split} samples of '{name}' from '{pickle_path}'")
    return loaded


def feature_variant(stave_height: int, precision: str) -> str:
    """Names a feature extraction setting, so that differently extracted features never mix."""
    return f"stave{stave_height}_{precision}"


def feature_cache_path(cache_dir: str | Path, dataset: str, sample_name: str, variant: str) -> Path:
    """
    Location of one sample's pre-extracted MuSViT features, shared by extraction and training.
    Namespaced by dataset, since sample names are only unique within a dataset.
    """
    return Path(cache_dir) / dataset / variant / f"{sample_name}.pt"


def missing_feature_files(samples: SplitSamples, cache_dir: str | Path, variant: str) -> list[Path]:
    """Feature files of the given samples that have not been extracted yet."""
    paths = (
        feature_cache_path(cache_dir, name, sample.sample_name, variant)
        for name, dataset_samples in samples.items()
        for sample in dataset_samples
    )
    return [path for path in paths if not path.is_file()]


# ==============================================================================
# Single-Staff OMR Dataset
# ==============================================================================

class StaveOMRDataset(Dataset):
    """
    Dataset backed directly by in-memory ZeusDatasetSample objects.
    - Run 1 ('zeus'): decodes sample.image bytes into normalized grayscale tensor (1, H, W).
    - Run 2 ('musvit'): loads the pre-extracted (rows, columns, dim) MuSViT patch grid from feature_cache_dir
      and arranges it as a sequence:
        - 'columns': one timestep per patch column, its rows concatenated -> (columns, rows * dim)
        - 'raster': row-major patch sequence, as in the MuSViT documentation -> (rows * columns, dim)
    - Targets: tokenized from sample.lmx string.
    """
    def __init__(
        self,
        samples: SplitSamples,
        vocab: TokenVocabulary,
        mode: str = "zeus",  # 'zeus' or 'musvit'
        feature_cache_dir: str | Path | None = None,
        feature_variant: str = "stave64_float16",
        feature_layout: str = "columns",  # 'columns' or 'raster'
        preload_features: bool = False,
        image_height: int = 96,  # Zeus single-staff standard height
        max_image_width: int = 1536,
    ):
        self.items = [(name, sample) for name, dataset_samples in samples.items() for sample in dataset_samples]
        self.vocab = vocab
        self.mode = mode.lower()
        self.feature_cache_dir = feature_cache_dir
        self.feature_variant = feature_variant
        self.feature_layout = feature_layout
        self.image_height = image_height
        self.max_image_width = max_image_width

        if self.mode == "musvit" and feature_cache_dir is None:
            raise ValueError("feature_cache_dir must be specified for mode='musvit'.")
        if feature_layout not in ("columns", "raster"):
            raise ValueError(f"Unknown feature_layout '{feature_layout}'. Must be 'columns' or 'raster'.")

        # Reading every feature file once up front, instead of in every epoch, spares slow (e.g. network) filesystems
        self.features: list[torch.Tensor] | None = None
        if self.mode == "musvit" and preload_features and len(self) > 0:
            self.features = [self._load_grid(i) for i in tqdm(range(len(self)), desc="Preloading features")]

    def __len__(self) -> int:
        return len(self.items)

    def gold_lmx(self) -> list[str]:
        """Gold LMX strings in dataset order."""
        return [sample.lmx for _, sample in self.items]

    def dataset_names(self) -> list[str]:
        """Name of the source dataset of each sample, in dataset order."""
        return [name for name, _ in self.items]

    def feature_path(self, idx: int) -> Path:
        name, sample = self.items[idx]
        return feature_cache_path(self.feature_cache_dir, name, sample.sample_name, self.feature_variant)

    def target_lengths(self) -> list[int]:
        """Number of LMX tokens of each sample, in dataset order."""
        return [len(sample.lmx.split()) for _, sample in self.items]

    def _load_grid(self, idx: int) -> torch.Tensor:
        """The (rows, columns, dim) MuSViT patch grid of a sample, as stored (FP16 or FP32)."""
        if self.features is not None:
            return self.features[idx]
        feat_path = self.feature_path(idx)
        if not feat_path.is_file():
            raise FileNotFoundError(
                f"Pre-extracted MuSViT feature file not found for sample '{self.items[idx][1].sample_name}'. "
                f"Looked at: '{feat_path}'. Please run extract_features.py before training MuSViT."
            )
        grid = torch.load(feat_path, map_location="cpu", weights_only=True)
        if grid.dim() != 3:
            raise ValueError(f"Expected a (rows, columns, dim) patch grid in '{feat_path}', got shape {tuple(grid.shape)}.")
        return grid

    def _load_image(self, image_bytes: bytes, sample_name: str) -> torch.Tensor:
        """Loads and normalizes image for Zeus (Run 1). Height is fixed, width preserves aspect ratio."""
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not decode the image of sample '{sample_name}'.")

        h, w = img.shape
        new_w = max(1, int(round(w * self.image_height / h)))
        if self.max_image_width is not None:
            new_w = min(new_w, self.max_image_width)

        resized = cv2.resize(img, (new_w, self.image_height), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0
        return tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, sample = self.items[idx]
        token_tensor = torch.tensor(self.vocab.encode(sample.lmx.split()), dtype=torch.long)

        if self.mode == "musvit":
            grid = self._load_grid(idx).float()
            rows, columns, dim = grid.shape
            if self.feature_layout == "columns":
                # Like the Zeus encoder flattening H x C per column, keeps the vertical (pitch) position
                feat = grid.permute(1, 0, 2).reshape(columns, rows * dim)
            else:
                feat = grid.reshape(rows * columns, dim)
            return feat, token_tensor

        return self._load_image(sample.image, sample.sample_name), token_tensor


# ==============================================================================
# Collate Function
# ==============================================================================

class StaveCollate:
    """Collates variable-width images or feature sequences and variable-length token targets."""
    def __init__(self, pad_idx: int = 2, bos_idx: int = 0, eos_idx: int = 1):
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __call__(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            inputs: (B, 1, H, W_max) images padded with white, or (B, T_max, D) features padded with zeros
            input_lengths: (B,) unpadded widths (images) or sequence lengths (features)
            input_seqs: (B, L+1) teacher forcing inputs [BOS, t_1, ..., t_L]
            target_labels: (B, L+1) labels for loss [t_1, ..., t_L, EOS]
        """
        inputs, targets = zip(*batch, strict=True)

        # Images are (1, H, W) and padded along W; feature sequences are (T, D) and padded along T
        is_image = (inputs[0].dim() == 3)
        length_dim = 2 if is_image else 0
        pad_value = 1.0 if is_image else 0.0

        input_lengths = torch.tensor([x.shape[length_dim] for x in inputs], dtype=torch.long)
        max_len_in = int(input_lengths.max())
        padded_inputs = []
        for x in inputs:
            pad = max_len_in - x.shape[length_dim]
            if pad > 0:
                padding = (0, pad, 0, 0) if is_image else (0, 0, 0, pad)
                x = torch.nn.functional.pad(x, padding, value=pad_value)
            padded_inputs.append(x)
        batch_inputs = torch.stack(padded_inputs, dim=0)

        # Pad target sequences
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

        return batch_inputs, input_lengths, batch_input_seqs, batch_target_labels


# ==============================================================================
# Length-Bucketed Batching
# ==============================================================================

class LengthBucketBatchSampler(Sampler[list[int]]):
    """
    Batches samples of similar transcription length. The decoder runs for the longest transcription
    of a batch, so mixing short and long ones wastes most of its steps on padding.

    - shuffle=True: samples are shuffled, sorted by length within pools of `pool_batches` batches,
      cut into batches, and the batches are shuffled. A pool of 1 batch means plain random batches.
    - shuffle=False: batches follow the length order, deterministically (for evaluation).
    """
    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool, pool_batches: int = 50):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool_batches = max(1, pool_batches)

    def _batches(self, order: list[int]) -> list[list[int]]:
        return [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]

    def __iter__(self):
        if not self.shuffle:
            yield from self._batches(sorted(range(len(self.lengths)), key=self.lengths.__getitem__))
            return

        # Seeded from the global RNG like torch's RandomSampler, so seeding and resuming stay reproducible
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        permutation = torch.randperm(len(self.lengths), generator=generator).tolist()
        pool_size = self.batch_size * self.pool_batches
        batches = []
        for start in range(0, len(permutation), pool_size):
            pool = sorted(permutation[start:start + pool_size], key=self.lengths.__getitem__)
            batches.extend(self._batches(pool))
        for i in torch.randperm(len(batches), generator=generator).tolist():
            yield batches[i]

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)
