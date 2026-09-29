"""
MusViT Embedding Extractor for OmniOMR Staves.

This script extracts feature embeddings from music score staves in the dataset
using the pre-trained MuSViT (Music Score Vision Transformer) foundation model
(PRAIG/musvit on Hugging Face).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from torchvision import transforms as T

try:
    from transformers import AutoModel, ViTModel
except ImportError:
    AutoModel = None
    ViTModel = None


def get_default_transform(image_size: int = 1024) -> T.Compose:
    """
    Returns the standard image transformation pipeline expected by MuSViT.
    Resizes image to (image_size, image_size) and converts to normalized tensor [0, 1].
    """
    return T.Compose([
        T.Resize([image_size, image_size]),
        T.ToTensor(),
    ])


def find_staves(dataset_dir: Union[str, Path] = "OmniOMR.Small") -> list[Path]:
    """
    Locates all stave image files in the OmniOMR dataset directory.
    Searches for 'image.jpg' inside each stave folder under 'Staves'.
    """
    base_path = Path(dataset_dir)
    if not base_path.exists():
        # Check relative to repository root if called from inside a subdirectory
        repo_root = Path(__file__).resolve().parent.parent
        alt_path = repo_root / dataset_dir
        if alt_path.exists():
            base_path = alt_path
        else:
            raise FileNotFoundError(
                f"OmniOMR dataset directory not found at '{dataset_dir}' or '{alt_path}'"
            )

    stave_images = sorted(list(base_path.glob("*/Staves/*/image.jpg")))
    if not stave_images:
        # Fallback to recursive glob in case of nested directory variations
        stave_images = sorted(list(base_path.rglob("Staves/*/image.jpg")))

    if not stave_images:
        raise FileNotFoundError(f"No stave images found in '{base_path}'")

    return stave_images


def load_stave_image(
    source: Union[int, str, Path],
    dataset_dir: Union[str, Path] = "OmniOMR.Small",
) -> tuple[Image.Image, Path]:
    """
    Loads a stave image given an index in the dataset or a direct file/folder path.

    Returns:
        (PIL.Image, Path): The loaded RGB PIL Image and its file path.
    """
    if isinstance(source, int):
        all_staves = find_staves(dataset_dir)
        if source < 0 or source >= len(all_staves):
            raise IndexError(
                f"Stave index {source} out of range (0 to {len(all_staves) - 1})"
            )
        img_path = all_staves[source]
    else:
        path = Path(source)
        if path.is_dir():
            candidates = list(path.glob("*.jpg")) + list(path.glob("*.png"))
            if not candidates:
                raise FileNotFoundError(f"No image files found inside directory '{path}'")
            img_path = candidates[0]
        elif path.is_file():
            img_path = path
        else:
            # Try resolving relative to dataset_dir or repo root
            repo_root = Path(__file__).resolve().parent.parent
            if (repo_root / source).exists():
                return load_stave_image(repo_root / source, dataset_dir)
            raise FileNotFoundError(f"Stave image or directory not found at '{source}'")

    image = Image.open(img_path).convert("RGB")
    return image, img_path


def load_musvit_model(
    model_name: str = "PRAIG/musvit",
    device: Optional[str] = None,
    token: Optional[str] = None,
):
    """
    Loads the MuSViT model from Hugging Face Hub.

    Args:
        model_name: Model identifier on Hugging Face (e.g. 'PRAIG/musvit' or 'PRAIG/musvit-light')
        device: Device to place the model on ('cuda', 'cpu', etc.). Defaults to auto-detection.
        token: Hugging Face access token (or set via HF_TOKEN environment variable).

    Returns:
        (torch.nn.Module, str): The model in evaluation mode and the active device string.
    """
    if ViTModel is None and AutoModel is None:
        raise ImportError(
            "The 'transformers' package is required to load MusViT.\n"
            "Please install it using: pip install transformers"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_token = token or os.environ.get("HF_TOKEN")

    print(f"Loading MusViT model '{model_name}' on device '{device}'...")
    try:
        try:
            model = ViTModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token,
            )
        except Exception:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=hf_token,
            )
    except Exception as e:
        err_msg = str(e)
        if "401" in err_msg or "gated" in err_msg.lower() or "log in" in err_msg.lower():
            raise RuntimeError(
                f"Access denied when loading '{model_name}'. Hugging Face requires authentication for this repository.\n\n"
                "You can authenticate using one of the following methods:\n"
                "  1. Run 'huggingface-cli login' in your terminal and paste your HF token.\n"
                "  2. Set environment variable: $env:HF_TOKEN=\"your_token_here\" (PowerShell) or export HF_TOKEN=\"...\" (Bash).\n"
                "  3. Pass the token directly to the script: python musvit/main.py --token \"your_token_here\"\n"
            ) from e
        raise

    model.to(device)
    model.eval()
    return model, device


def extract_embeddings(
    stave_source: Union[int, str, Path, Image.Image],
    model: Optional[torch.nn.Module] = None,
    transform: Optional[T.Compose] = None,
    device: Optional[str] = None,
    token: Optional[str] = None,
    model_name: str = "PRAIG/musvit",
    dataset_dir: Union[str, Path] = "OmniOMR.Small",
) -> dict[str, Union[torch.Tensor, Path, tuple[int, int]]]:
    """
    Extracts embeddings for a single stave using MusViT.

    Args:
        stave_source: Stave index (int), image path (str/Path), or PIL Image.
        model: Optional preloaded model. If None, it will be loaded.
        transform: Optional torchvision transform. If None, default (1024x1024) is used.
        device: 'cuda' or 'cpu'. Defaults to auto-detection.
        token: Optional Hugging Face token.
        model_name: Hugging Face model identifier if model is not provided.
        dataset_dir: Root path to OmniOMR dataset.

    Returns:
        Dictionary containing:
            - 'last_hidden_state': Full sequence embeddings [1, 4097, 768]
            - 'cls_token': [CLS] token embedding representing the whole stave [1, 768]
            - 'patch_embeddings': Spatial patch embeddings [1, 4096, 768]
            - 'spatial_grid': Spatial patch embeddings reshaped to 2D grid [1, 64, 64, 768]
            - 'mean_patch_embedding': Mean over all patch embeddings [1, 768]
            - 'original_size': (width, height) of original stave image
            - 'image_path': Path to stave image if loaded from file, else None
    """
    img_path = None
    if isinstance(stave_source, Image.Image):
        image = stave_source.convert("RGB")
        orig_size = image.size
    else:
        image, img_path = load_stave_image(stave_source, dataset_dir=dataset_dir)
        orig_size = image.size

    if transform is None:
        transform = get_default_transform(image_size=1024)

    if model is None:
        model, device = load_musvit_model(model_name=model_name, device=device, token=token)
    elif device is None:
        device = next(model.parameters()).device.type

    # Prepare batch tensor: (1, 3, 1024, 1024)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    # outputs.last_hidden_state shape: (Batch=1, SeqLen=4097, HiddenDim=768)
    if hasattr(outputs, "last_hidden_state"):
        last_hidden_state = outputs.last_hidden_state
    else:
        last_hidden_state = outputs[0]

    cls_token = last_hidden_state[:, 0, :]  # (1, 768)
    patch_embeddings = last_hidden_state[:, 1:, :]  # (1, 4096, 768)

    # ViT divides 1024x1024 image into 16x16 patches -> 64x64 grid
    grid_dim = int(patch_embeddings.shape[1] ** 0.5)
    spatial_grid = patch_embeddings.reshape(
        patch_embeddings.shape[0], grid_dim, grid_dim, patch_embeddings.shape[2]
    )  # (1, 64, 64, 768)

    mean_patch = patch_embeddings.mean(dim=1)  # (1, 768)

    result = {
        "last_hidden_state": last_hidden_state,
        "cls_token": cls_token,
        "patch_embeddings": patch_embeddings,
        "spatial_grid": spatial_grid,
        "mean_patch_embedding": mean_patch,
        "original_size": orig_size,
        "image_path": img_path,
    }

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        result["pooler_output"] = outputs.pooler_output

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MusViT embeddings from a stave in the OmniOMR.Small dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="OmniOMR.Small",
        help="Path to the OmniOMR.Small dataset directory (default: 'OmniOMR.Small').",
    )
    parser.add_argument(
        "--stave-idx",
        type=int,
        default=0,
        help="0-based index of the stave to extract embeddings for (default: 0).",
    )
    parser.add_argument(
        "--stave-path",
        type=str,
        default=None,
        help="Direct path to a stave image or stave directory (overrides --stave-idx).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="PRAIG/musvit",
        help="Hugging Face model identifier (default: 'PRAIG/musvit', or 'PRAIG/musvit-light').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on ('cuda', 'cpu'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save extracted embeddings to disk as a PyTorch .pt file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Custom output file path to save embeddings (implies --save).",
    )
    parser.add_argument(
        "--token",
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face access token (can also be provided via HF_TOKEN environment variable).",
    )
    args = parser.parse_args()

    # Determine input stave source
    source: Union[int, str]
    if args.stave_path:
        source = args.stave_path
        print(f"Target stave path: {source}")
    else:
        source = args.stave_idx
        try:
            all_staves = find_staves(args.dataset_dir)
            print(f"Found {len(all_staves)} staves in dataset '{args.dataset_dir}'.")
            print(f"Selected stave index: {args.stave_idx} -> {all_staves[args.stave_idx]}")
        except Exception as e:
            print(f"Error locating staves in '{args.dataset_dir}': {e}", file=sys.stderr)
            sys.exit(1)

    # Run extraction
    print("\nExtracting MusViT embeddings...")
    try:
        embeddings = extract_embeddings(
            stave_source=source,
            model_name=args.model_name,
            device=args.device,
            token=args.token,
            dataset_dir=args.dataset_dir,
        )
    except Exception as e:
        print(f"\nFailed to extract embeddings: {e}", file=sys.stderr)
        sys.exit(1)

    # Report results
    img_path = embeddings["image_path"]
    orig_size = embeddings["original_size"]
    last_hidden_state: torch.Tensor = embeddings["last_hidden_state"]
    cls_token: torch.Tensor = embeddings["cls_token"]
    patch_embeddings: torch.Tensor = embeddings["patch_embeddings"]
    spatial_grid: torch.Tensor = embeddings["spatial_grid"]

    print("\n" + "=" * 60)
    print("MusViT Embedding Extraction Summary")
    print("=" * 60)
    if img_path:
        print(f"Stave Image:        {img_path}")
    print(f"Original Dimension: {orig_size[0]} x {orig_size[1]} (width x height)")
    print(f"Model Input Size:   1024 x 1024")
    print("-" * 60)
    print(f"Last Hidden State:  {tuple(last_hidden_state.shape)} (batch, tokens, hidden_dim)")
    print(f"CLS Token:          {tuple(cls_token.shape)} (global stave embedding)")
    print(f"Patch Tokens:       {tuple(patch_embeddings.shape)} (64x64 spatial tokens)")
    print(f"Spatial Grid:       {tuple(spatial_grid.shape)} (batch, height, width, hidden_dim)")
    print(f"CLS Norm:           {cls_token.norm().item():.4f}")
    print(f"CLS Mean / Std:     {cls_token.mean().item():.4f} / {cls_token.std().item():.4f}")
    print("=" * 60)

    # Save to disk if requested
    if args.save or args.output_file:
        if args.output_file:
            out_path = Path(args.output_file)
        else:
            out_dir = Path(__file__).resolve().parent / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            stave_id = f"stave_{args.stave_idx}" if not args.stave_path else Path(args.stave_path).stem
            out_path = out_dir / f"{stave_id}_musvit_embeddings.pt"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "last_hidden_state": last_hidden_state.cpu(),
                "cls_token": cls_token.cpu(),
                "patch_embeddings": patch_embeddings.cpu(),
                "spatial_grid": spatial_grid.cpu(),
                "image_path": str(img_path) if img_path else None,
                "original_size": orig_size,
                "model_name": args.model_name,
            },
            out_path,
        )
        print(f"\nSaved embeddings to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
