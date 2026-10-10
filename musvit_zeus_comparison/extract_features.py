"""
Feature Pre-Extraction Script for MuSViT.

Extracts visual features from single stave images using pre-trained MuSViT,
preparing staves as the MuSViT documentation prescribes:
- The stave is resized to the full canvas width and a fixed height (1024 x 64 by default)
  and pasted at the top of a white 1024 x 1024 canvas, the input size MuSViT was pre-trained on.
- Only the patch rows covering the stave are kept (64 / 16 = 4 rows of 64 patch columns),
  the rows of the white padding below are cut away.

Follows the Zeus repository workflow:
- Reads samples directly from the pickled splits (samples.{train,dev,test}.pickle) of the selected datasets.
- Stores the (rows, columns, dim) patch grid of each stave under
  feature_cache/<dataset>/stave<height>_<precision>/<sample_name>.pt
  (~393 KB per stave in FP16 for height 64, ~3.9 GB for 10,000 staves).
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

try:
    from musvit_zeus_comparison.dataset import (
        ZeusDataset,
        dataset_name,
        discover_datasets,
        feature_cache_path,
        feature_variant,
        prepare_stave_band,
        resolve_dataset_folder,
    )
    from musvit_zeus_comparison.models import load_musvit, stave_canvas, stave_patch_grid
except ImportError:
    from dataset import (
        ZeusDataset,
        dataset_name,
        discover_datasets,
        feature_cache_path,
        feature_variant,
        prepare_stave_band,
        resolve_dataset_folder,
    )
    from models import load_musvit, stave_canvas, stave_patch_grid


class StaveBands(torch.utils.data.Dataset):
    """Decodes and resizes stave images in DataLoader workers, so that the GPU does not wait for them."""
    def __init__(self, images: list[bytes], canvas_size: int, stave_height: int):
        self.images = images
        self.canvas_size = canvas_size
        self.stave_height = stave_height

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor | str]:
        """The index with the stave band, or with the error message if the image cannot be decoded."""
        try:
            return idx, prepare_stave_band(Image.open(io.BytesIO(self.images[idx])), self.canvas_size, self.stave_height)
        except OSError as e:
            return idx, str(e)


def keep_as_list(batch: list) -> list:
    return batch


def resolve_pickle_files(specs: list[str], dataset_dir: str | Path) -> list[Path]:
    """All pickled splits of the given datasets (names or folders), or the given .pickle files."""
    pickle_files = []
    for spec in specs:
        if Path(spec).suffix == ".pickle":
            if not Path(spec).is_file():
                raise FileNotFoundError(f"Zeus dataset pickle not found at: '{spec}'")
            pickle_files.append(Path(spec))
            continue
        folder = resolve_dataset_folder(spec, dataset_dir)
        found = sorted(folder.glob("samples.*.pickle"))
        if not found:
            raise FileNotFoundError(f"No pickled splits (samples.*.pickle) in dataset folder '{folder}'.")
        pickle_files.extend(found)
    return sorted(set(pickle_files))


def extract_and_cache(
    datasets: list[str],
    dataset_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "PRAIG/musvit",
    stave_height: int = 64,
    precision: str = "float16",  # 'float16' or 'float32'
    batch_size: int = 8,
    num_workers: int = 4,
    tf32: bool = False,
    device: str | None = None,
    token: str | None = None,
    skip_existing: bool = True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pickle_files = resolve_pickle_files(datasets, dataset_dir)
    print(f"Found {len(pickle_files)} Zeus dataset pickle file(s):")
    for pf in pickle_files:
        print(f"  - {pf}")

    variant = feature_variant(stave_height, precision)

    # Collect (output path, sample) pairs; a sample may appear in several splits of its dataset
    # (e.g. samples.all.pickle), so deduplicate by output path, which is unique per dataset and sample
    todo: dict[Path, bytes] = {}
    seen: set[Path] = set()
    skipped_count = 0
    for pickle_path in pickle_files:
        name = dataset_name(pickle_path)
        for sample in ZeusDataset.load_from_pickle_file(pickle_path).samples:
            feat_path = feature_cache_path(output_dir, name, sample.sample_name, variant)
            if feat_path in seen:
                continue
            seen.add(feat_path)
            if skip_existing and feat_path.is_file():
                skipped_count += 1
                continue
            todo[feat_path] = sample.image
    print(f"Stave samples to extract: {len(todo):,} (already cached: {skipped_count:,})")
    if not todo:
        return

    model = load_musvit(model_name=model_name, device=device, token=token)
    print(f"MuSViT loaded on {device}.")
    canvas_size, patch_size = model.config.image_size, model.config.patch_size
    if stave_height % patch_size != 0 or not 0 < stave_height <= canvas_size:
        raise ValueError(
            f"--stave-height must be a multiple of the patch size ({patch_size}) "
            f"between {patch_size} and {canvas_size}, got {stave_height}."
        )
    patch_rows = stave_height // patch_size
    dtype = torch.float16 if precision == "float16" else torch.float32
    if tf32:
        # Faster matrix multiplications on Ampere+ GPUs, with a 10-bit mantissa like the FP16 storage
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    saved_count = 0
    failed: list[Path] = []
    paths_todo = list(todo)
    loader = torch.utils.data.DataLoader(
        StaveBands(list(todo.values()), canvas_size, stave_height),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=keep_as_list,
    )

    print(
        f"Extracting features of staves resized to {canvas_size}x{stave_height} on a {canvas_size}x{canvas_size} "
        f"canvas ({patch_rows} patch rows), precision='{precision}'..."
    )
    for batch in tqdm(loader, desc="Extracting features"):
        bands, paths = [], []
        for idx, band in batch:
            if isinstance(band, str):
                print(f"\n[Warning] Could not decode the image for '{paths_todo[idx]}': {band}")
                failed.append(paths_todo[idx])
                continue
            bands.append(band)
            paths.append(paths_todo[idx])
        if not bands:
            continue

        with torch.inference_mode():
            canvas = stave_canvas(torch.stack(bands).to(device, non_blocking=True), canvas_size)
            last_hidden_state = model(canvas).last_hidden_state
            grids = stave_patch_grid(last_hidden_state, patch_rows, canvas_size // patch_size).to(dtype=dtype, device="cpu")

        for feat_path, feat in zip(paths, grids, strict=True):
            feat_path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename, so an interrupted run never leaves a truncated file that a resumed run would skip.
            # Clone, so that only this sample's slice of the batch tensor is serialized.
            tmp_path = feat_path.with_name(feat_path.name + ".tmp")
            torch.save(feat.clone(), tmp_path)
            tmp_path.replace(feat_path)
            saved_count += 1

    total_size_mb = sum(p.stat().st_size for p in Path(output_dir).rglob("*.pt")) / (1024 * 1024)
    print("\nExtraction Complete!")
    print(f"  - Saved new features: {saved_count:,}")
    print(f"  - Skipped (already cached): {skipped_count:,}")
    print(f"  - Total cache size: {total_size_mb:.2f} MB")
    print(f"  - Output directory: {output_dir}")
    if failed:
        print(f"  - Failed to decode {len(failed):,} image(s); training on these samples will fail:")
        for path in failed:
            print(f"      {path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-extract and compress MuSViT features from Zeus dataset pickles.")
    parser.add_argument("--dataset-dir", type=str, default="datasets", help="Directory containing the datasets as subfolders.")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Datasets to extract: names of subfolders of --dataset-dir, dataset folders, or .pickle files "
             "(default: all datasets in --dataset-dir).",
    )
    parser.add_argument("--output-dir", type=str, default="feature_cache", help="Path to save pre-extracted features.")
    parser.add_argument("--model-name", type=str, default="PRAIG/musvit", help="HuggingFace model ID.")
    parser.add_argument(
        "--stave-height",
        type=int,
        default=64,
        help="Height staves are resized to on the 1024x1024 canvas; a multiple of the 16px patch size "
             "(default: 64 = 4 patch rows, as in the MuSViT documentation).",
    )
    parser.add_argument("--precision", type=str, default="float16", choices=["float16", "float32"], help="Tensor precision.")
    parser.add_argument("--batch-size", type=int, default=8, help="Images per MuSViT forward pass.")
    parser.add_argument("--num-workers", type=int, default=4, help="Workers decoding and resizing images ahead of the GPU.")
    parser.add_argument("--tf32", action="store_true", help="Use TF32 matrix multiplications on Ampere+ GPUs: several times faster, slightly less precise (10-bit mantissa, like the FP16 storage).")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu'). Auto-detected if not given.")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token if required.")
    parser.add_argument("--no-skip", action="store_true", help="Force re-extraction of existing cached files.")

    args = parser.parse_args()
    datasets = args.datasets or discover_datasets(args.dataset_dir)
    if not datasets:
        raise FileNotFoundError(f"No datasets with pickled splits found in '{args.dataset_dir}'.")

    extract_and_cache(
        datasets=datasets,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        stave_height=args.stave_height,
        precision=args.precision,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tf32=args.tf32,
        device=args.device,
        token=args.token,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
