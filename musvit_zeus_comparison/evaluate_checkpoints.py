"""Evaluate one or more saved checkpoints on one converted Zeus dataset.

Examples:
    python -m musvit_zeus_comparison.evaluate_checkpoints \
        --dataset datasets/dolores_small \
        --checkpoint old=/models/zeus_old.pt \
        --checkpoint new=/models/zeus_new.pt \
        --model-type old=zeus --model-type new=zeus \
        --device cuda

The dataset must contain samples.test.pickle (or a pickle may be passed
directly). Checkpoints are specified as LABEL=PATH so that results from
different runs can be compared without overwriting one another.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from musvit_zeus_comparison.dataset import (
        StaveCollate,
        StaveOMRDataset,
        TokenVocabulary,
        load_split,
        feature_variant,
    )
    from musvit_zeus_comparison.models import CombinedOMRModel
    from musvit_zeus_comparison.train import evaluate_ser, make_loader
except ImportError:
    from dataset import StaveCollate, StaveOMRDataset, TokenVocabulary, load_split, feature_variant
    from models import CombinedOMRModel
    from train import evaluate_ser, make_loader


def parse_named(value: str, option: str) -> tuple[str, str]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f"{option} must have the form LABEL=VALUE, got '{value}'")
    return label, path


def checkpoint_model_dimensions(checkpoint: dict, model_type: str) -> tuple[int, int]:
    """Infer dimensions that are encoded in every checkpoint's tensor shapes."""
    state = checkpoint["model"]
    dim = int(state["decoder.embedding.weight"].shape[1])
    if model_type == "musvit":
        musvit_dim = int(state["encoder.proj.0.weight"].shape[1])
    else:
        musvit_dim = 768
    return dim, musvit_dim


def checkpoint_path(value: str) -> tuple[str, Path]:
    label, path = parse_named(value, "--checkpoint")
    return label, Path(path)


def model_type_path(value: str) -> tuple[str, str]:
    label, model_type = parse_named(value, "--model-type")
    model_type = model_type.lower()
    if model_type not in {"zeus", "musvit"}:
        raise argparse.ArgumentTypeError("--model-type must be zeus or musvit")
    return label, model_type


def test_pickle(dataset: Path) -> Path:
    if dataset.suffix == ".pickle":
        return dataset
    for name in ("test", "validation", "val"):
        candidate = dataset / f"samples.{name}.pickle"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No test pickle found in '{dataset}'. Expected samples.test.pickle."
    )


def load_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: '{path}'")
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate_one(
    label: str,
    checkpoint_path_value: Path,
    model_type: str,
    samples,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    checkpoint = load_checkpoint(checkpoint_path_value)
    if "model" not in checkpoint or "vocab" not in checkpoint:
        raise ValueError(
            f"'{checkpoint_path_value}' is not a checkpoint produced by this trainer "
            "(missing 'model' or 'vocab')."
        )

    vocab = TokenVocabulary.from_dict(checkpoint["vocab"])
    dim, musvit_dim = checkpoint_model_dimensions(checkpoint, model_type)
    dataset = StaveOMRDataset(
        samples,
        vocab=vocab,
        mode=model_type,
        feature_cache_dir=args.feature_cache_dir,
        feature_variant=feature_variant(args.feature_stave_height, args.feature_precision),
        feature_layout=args.feature_layout,
        preload_features=args.preload_features,
        image_height=args.image_height,
        max_image_width=args.max_image_width,
    )
    model = CombinedOMRModel(
        encoder_type=model_type,
        vocab_size=len(vocab),
        dim=dim,
        timestep_width=args.timestep_width,
        input_height=args.image_height,
        musvit_dim=musvit_dim,
        bos_idx=vocab.bos_idx,
        eos_idx=vocab.eos_idx,
        pad_idx=vocab.pad_idx,
        max_length=args.max_gen_length,
        dropout=args.dropout,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    loader = make_loader(dataset, args, shuffle=False)
    ser = evaluate_ser(model, loader, vocab, device)
    return {
        "label": label,
        "checkpoint": str(checkpoint_path_value),
        "model_type": model_type,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "samples": len(dataset),
        "ser": round(ser, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Converted Zeus dataset folder or test pickle.")
    parser.add_argument("--checkpoint", required=True, action="append", type=checkpoint_path)
    parser.add_argument("--model-type", action="append", default=[], type=model_type_path,
                        help="LABEL=zeus or LABEL=musvit; defaults to zeus for every checkpoint.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--feature-cache-dir", type=Path, default="feature_cache")
    parser.add_argument("--feature-stave-height", type=int, default=64)
    parser.add_argument("--feature-precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--feature-layout", choices=["columns", "raster"], default="columns")
    parser.add_argument("--preload-features", action="store_true")
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--max-image-width", type=int, default=1536)
    parser.add_argument("--timestep-width", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-gen-length", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--length-bucket-batches", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_pickle = test_pickle(args.dataset)
    samples = load_split([str(dataset_pickle)], dataset_pickle.parent, "test", keep_images=True)
    model_types = dict(args.model_type)
    results = []

    for label, path in args.checkpoint:
        model_type = model_types.get(label, "zeus")
        print(f"\nEvaluating {label} ({model_type}) from {path}")
        try:
            result = evaluate_one(label, path, model_type, samples, args, device)
        except (RuntimeError, KeyError, ValueError, FileNotFoundError) as error:
            raise RuntimeError(
                f"Could not evaluate checkpoint '{label}'. This usually means the checkpoint "
                f"was created with incompatible model code or settings: {error}"
            ) from error
        results.append(result)
        print(f"{label}: SER={result['ser']:.4f}% on {result['samples']:,} samples")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
