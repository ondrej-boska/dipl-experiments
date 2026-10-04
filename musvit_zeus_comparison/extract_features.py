"""
Feature Pre-Extraction Script for MuSViT.

Extracts and compresses visual features from single stave images using pre-trained MuSViT.
Addresses disk size by applying vertical pooling and FP16 half-precision:
- Full uncompressed patch embeddings: ~12.6 MB per stave (126 GB for 10,000 staves)
- Vertical Mean Pooling (FP16): ~98 KB per stave (under 1 GB for 10,000 staves) -> 128x smaller!
- Vertical 4-Slice Pooling (FP16): ~393 KB per stave (~3.9 GB for 10,000 staves) -> 32x smaller!

Can be run on the GPU cluster before training to eliminate ViT training overhead.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

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


def find_staves(dataset_dir: str | Path) -> list[Path]:
    """Finds all stave images in standard MusiCorpus / OmniOMR layout."""
    base_path = Path(dataset_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    staves = sorted(list(base_path.glob("*/Staves/*/image.jpg")))
    if not staves:
        staves = sorted(list(base_path.glob("*/*/Staves/*/image.jpg")))
    if not staves:
        staves = sorted(list(base_path.rglob("Staves/*/image.jpg")))
    if not staves:
        # Generic fallback for any jpg/png images
        staves = sorted(list(base_path.rglob("*.jpg")) + list(base_path.rglob("*.png")))

    return staves


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

    # Reshape to spatial 2D grid: (1, 64, 64, 768) -> (1, 768, 64, 64)
    grid = patch_embeddings.view(1, grid_dim, grid_dim, dim).permute(0, 3, 1, 2)

    if mode == "vertical_mean":
        # Average pool across the vertical axis (dim 2) -> (1, 768, 1, 64) -> (64, 768)
        pooled = grid.mean(dim=2).squeeze(0).permute(1, 0)
        return pooled

    elif mode == "vertical_4slice":
        # Adaptive average pool across height into 4 bins -> (1, 768, 4, 64) -> (256, 768)
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid, (4, 64))
        # Flatten spatial (4, 64) -> 256
        pooled = pooled.squeeze(0).permute(1, 2, 0).reshape(4 * 64, dim)
        return pooled

    elif mode == "full":
        return patch_embeddings.squeeze(0)  # (4096, 768)

    else:
        raise ValueError(f"Unknown pooling mode '{mode}'. Choose 'vertical_mean', 'vertical_4slice', or 'full'.")


def extract_and_cache(
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "PRAIG/musvit",
    pool_mode: str = "vertical_mean",
    precision: str = "float16",  # 'float16' or 'float32'
    device: str | None = None,
    token: str | None = None,
    skip_existing: bool = True,
):
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    staves = find_staves(dataset_path)
    print(f"Found {len(staves)} stave images in '{dataset_dir}'.")
    if not staves:
        print("No stave images found! Exiting.")
        return

    model = load_musvit(model_name=model_name, device=device, token=token)
    transform = get_transform(image_size=1024)

    dtype = torch.float16 if precision == "float16" else torch.float32

    manifest = []
    skipped_count = 0
    saved_count = 0

    print(f"Extracting features with pooling='{pool_mode}', precision='{precision}'...")
    for img_path in tqdm(staves, desc="Extracting features"):
        # Generate stable cache filename based on relative path or hash
        rel_path = img_path.relative_to(dataset_path) if dataset_path in img_path.parents else img_path.name
        rel_str = str(rel_path).replace("\\", "_").replace("/", "_").replace(".jpg", "").replace(".png", "")
        feat_filename = f"{rel_str}_{pool_mode}_{precision}.pt"
        feat_path = output_path / feat_filename

        manifest_entry = {
            "image_path": str(img_path.resolve()),
            "feature_path": str(feat_path.resolve()),
            "relative_path": str(rel_path),
        }

        # Check for matching transcription (MusicXML or LMX)
        musicxml_path = img_path.parent / "transcription.musicxml"
        if musicxml_path.exists():
            manifest_entry["musicxml_path"] = str(musicxml_path.resolve())

        lmx_path = img_path.parent / "transcription.lmx"
        # Check if already cached under direct path or canonical stave key
        existing_feat = None
        if skip_existing:
            if feat_path.is_file():
                existing_feat = feat_path
            elif "Staves" in img_path.parts:
                parts = img_path.parts
                staves_idx = parts.index("Staves")
                if staves_idx > 0 and staves_idx + 1 < len(parts):
                    stave_key = f"{parts[staves_idx - 1]}_Staves_{parts[staves_idx + 1]}_{img_path.stem}"
                    cand = output_path / f"{stave_key}_{pool_mode}_{precision}.pt"
                    if cand.is_file():
                        existing_feat = cand

        if existing_feat is not None:
            manifest_entry["feature_path"] = str(existing_feat.resolve())
            manifest.append(manifest_entry)
            skipped_count += 1
            continue

        manifest.append(manifest_entry)

        try:
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                last_hidden_state = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                patch_embeddings = last_hidden_state[:, 1:, :]  # drop [CLS] -> (1, 4096, 768)

                pooled = pool_embeddings(patch_embeddings, mode=pool_mode).to(dtype=dtype, device="cpu")

            torch.save(pooled, feat_path)
            saved_count += 1

        except Exception as e:
            print(f"\n[Warning] Failed on {img_path}: {e}")

    # Save manifest JSON for fast dataset lookup
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_size_mb = sum(p.stat().st_size for p in output_path.glob("*.pt")) / (1024 * 1024)
    print("\nExtraction Complete!")
    print(f"  - Saved new features: {saved_count}")
    print(f"  - Skipped (already cached): {skipped_count}")
    print(f"  - Total cache size: {total_size_mb:.2f} MB")
    print(f"  - Manifest written to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-extract and compress MuSViT features for single staves.")
    parser.add_argument("--dataset-dir", type=str, default="UFAL.OmniOMR", help="Path to input dataset (default: UFAL.OmniOMR).")
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
        dataset_dir=args.dataset_dir,
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
