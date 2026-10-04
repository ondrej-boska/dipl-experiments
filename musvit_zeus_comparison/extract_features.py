"""
Feature Pre-Extraction Script for MuSViT.

Extracts and compresses visual features from single stave images using pre-trained MuSViT.
Follows the Zeus repository workflow:
- Reads samples directly from Zeus pickled datasets (samples.train.pickle, samples.dev.pickle, samples.test.pickle).
- Applies vertical pooling and FP16 half-precision:
  - Vertical Mean Pooling (FP16): ~98 KB per stave (under 1 GB for 10,000 staves) -> 128x smaller!
  - Vertical 4-Slice Pooling (FP16): ~393 KB per stave (~3.9 GB for 10,000 staves) -> 32x smaller!
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

try:
    from musvit_zeus_comparison.dataset import ZeusDatasetSample, load_zeus_pickles
except ImportError:
    from dataset import ZeusDatasetSample, load_zeus_pickles

try:
    from transformers import AutoModel, ViTModel
except ImportError:
    AutoModel = None
    ViTModel = None


def get_transform(image_size: int = 1024) -> T.Compose:
    return T.Compose([
        T.Resize([image_size, image_size]),
        T.ToTensor(),
    ])


def load_musvit(model_name: str = "PRAIG/musvit", device: str = "cuda", token: str | None = None):
    if ViTModel is None and AutoModel is None:
        raise ImportError("Please install transformers: pip install transformers")

    print(f"Loading pre-trained MuSViT model '{model_name}' on {device}...")
    hf_token = token or os.environ.get("HF_TOKEN")
    try:
        try:
            model = ViTModel.from_pretrained(model_name, token=hf_token)
        except Exception:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    except Exception as e:
        raise RuntimeError(f"Failed to load '{model_name}'. Note: HF token may be needed if gated. Error: {e}")

    model.to(device)
    model.eval()
    return model


def pool_embeddings(patch_embeddings: torch.Tensor, mode: str = "vertical_mean") -> torch.Tensor:
    """
    Downsamples the 2D patch grid (64x64) into a compact sequence representation.

    Args:
        patch_embeddings: (1, 4096, 768)
        mode:
          - 'vertical_mean': Averages over height -> (64, 768). Size: 98 KB in FP16.
          - 'vertical_4slice': Adaptive pool into 4 vertical slices -> (256, 768). Size: 393 KB in FP16.
          - 'full': Keeps all 4096 patches -> (4096, 768). Size: 6.3 MB in FP16.
    """
    grid_dim = int(patch_embeddings.shape[1] ** 0.5)  # 64
    dim = patch_embeddings.shape[2]  # 768

    grid = patch_embeddings.view(1, grid_dim, grid_dim, dim).permute(0, 3, 1, 2)

    if mode == "vertical_mean":
        pooled = grid.mean(dim=2).squeeze(0).permute(1, 0)
        return pooled
    elif mode == "vertical_4slice":
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (4, 64))
        pooled = pooled.squeeze(0).permute(1, 2, 0).reshape(4 * 64, dim)
        return pooled
    elif mode == "full":
        return patch_embeddings.squeeze(0)
    else:
        raise ValueError(f"Unknown pooling mode '{mode}'. Choose 'vertical_mean', 'vertical_4slice', or 'full'.")


def resolve_pickle_files(input_paths: list[str | Path]) -> list[Path]:
    """Finds all Zeus .pickle files from list of files or directories."""
    pickle_files = []
    for p in input_paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".pickle":
            pickle_files.append(path)
        elif path.is_dir():
            found = sorted(list(path.glob("*.pickle")) + list(path.glob("*/*.pickle")))
            pickle_files.extend(found)
        else:
            cand = path.with_suffix(".pickle")
            if cand.is_file():
                pickle_files.append(cand)
    return sorted(list(set(pickle_files)))


def extract_and_cache(
    input_paths: list[str | Path],
    output_dir: str | Path,
    model_name: str = "PRAIG/musvit",
    pool_mode: str = "vertical_mean",
    precision: str = "float16",  # 'float16' or 'float32'
    device: str | None = None,
    token: str | None = None,
    skip_existing: bool = True,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pickle_files = resolve_pickle_files(input_paths)
    if not pickle_files:
        raise FileNotFoundError(f"No Zeus dataset .pickle files found in: {input_paths}")

    print(f"Found {len(pickle_files)} Zeus dataset pickle file(s):")
    for pf in pickle_files:
        print(f"  - {pf}")

    # Load samples directly from pickles
    samples = load_zeus_pickles(pickle_files)
    # Deduplicate by sample_name
    unique_samples = {}
    for s in samples:
        if s.sample_name not in unique_samples:
            unique_samples[s.sample_name] = s
    sample_list = list(unique_samples.values())
    print(f"Total unique stave samples to extract: {len(sample_list):,}")

    model = load_musvit(model_name=model_name, device=device, token=token)
    transform = get_transform(image_size=1024)
    dtype = torch.float16 if precision == "float16" else torch.float32

    skipped_count = 0
    saved_count = 0

    print(f"Extracting features with pooling='{pool_mode}', precision='{precision}'...")
    for sample in tqdm(sample_list, desc="Extracting features"):
        sample_slug = sample.sample_name.replace("/", "_").replace("\\", "_")
        feat_filename = f"{sample_slug}_{pool_mode}_{precision}.pt"
        feat_path = output_path / feat_filename

        if skip_existing and feat_path.is_file():
            skipped_count += 1
            continue

        try:
            image = Image.open(io.BytesIO(sample.image)).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                last_hidden_state = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                patch_embeddings = last_hidden_state[:, 1:, :]  # drop [CLS] -> (1, 4096, 768)
                pooled = pool_embeddings(patch_embeddings, mode=pool_mode).to(dtype=dtype, device="cpu")

            torch.save(pooled, feat_path)
            saved_count += 1

        except Exception as e:
            print(f"\n[Warning] Failed on sample '{sample.sample_name}': {e}")

    total_size_mb = sum(p.stat().st_size for p in output_path.glob("*.pt")) / (1024 * 1024)
    print("\nExtraction Complete!")
    print(f"  - Saved new features: {saved_count:,}")
    print(f"  - Skipped (already cached): {skipped_count:,}")
    print(f"  - Total cache size: {total_size_mb:.2f} MB")
    print(f"  - Output directory: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-extract and compress MuSViT features from Zeus dataset pickles.")
    parser.add_argument(
        "--datasets",
        "--dataset-dir",
        "--input",
        dest="inputs",
        type=str,
        nargs="+",
        default=["datasets"],
        help="Path(s) to dataset pickle files (e.g. datasets/omniomr/samples.train.pickle) or dataset directory.",
    )
    parser.add_argument("--output-dir", type=str, default="feature_cache", help="Path to save pre-extracted features.")
    parser.add_argument("--model-name", type=str, default="PRAIG/musvit", help="HuggingFace model ID.")
    parser.add_argument(
        "--pool-mode",
        type=str,
        default="vertical_mean",
        choices=["vertical_mean", "vertical_4slice", "full"],
        help="Feature pooling strategy (vertical_mean = ~98KB/sample, vertical_4slice = ~393KB/sample)."
    )
    parser.add_argument("--precision", type=str, default="float16", choices=["float16", "float32"], help="Tensor precision.")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu'). Auto-detected if not given.")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token if required.")
    parser.add_argument("--no-skip", action="store_true", help="Force re-extraction of existing cached files.")

    args = parser.parse_args()
    extract_and_cache(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pool_mode=args.pool_mode,
        precision=args.precision,
        device=args.device,
        token=args.token,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
