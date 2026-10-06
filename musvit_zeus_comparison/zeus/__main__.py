"""
Command-line entry point of the embedded Zeus modules, for preparing datasets
without the original Zeus package:

    python -m musvit_zeus_comparison.zeus musicorpus --input <mc_dataset> --output <zeus_dataset> --take-staves
    python -m musvit_zeus_comparison.zeus pickle <zeus_dataset>/samples.train.txt [...]
"""

import argparse
from pathlib import Path

from .data.zeus_dataset import ZeusDataset
from .musicorpus.convert_musicorpus_to_zeus import convert_musicorpus_to_zeus


def musicorpus(args: argparse.Namespace):
    if not (args.take_staves or args.take_grandstaves):
        raise SystemExit("Nothing to convert: pass --take-staves and/or --take-grandstaves.")
    convert_musicorpus_to_zeus(
        input_path=Path(args.input),
        output_path=Path(args.output),
        take_staves=args.take_staves,
        take_grandstaves=args.take_grandstaves,
        re_crop=False,
        normalize_image_height=None,
    )


def pickle(args: argparse.Namespace):
    for samples_file in map(Path, args.samples_files):
        pickle_path = samples_file.with_suffix(".pickle")
        print(f"Pickling '{samples_file}' -> '{pickle_path}'")
        dataset = ZeusDataset.load_from_samples_file(
            samples_file,
            image_suffix=args.image_suffix,
            show_progress_bar=True,
            benevolent=args.benevolent,
        )
        dataset.write_to_pickle_file(pickle_path)
        print(f"Pickled {len(dataset.samples):,} samples.")


def main():
    parser = argparse.ArgumentParser(prog="python -m musvit_zeus_comparison.zeus", description=__doc__.split("\n\n")[0].strip())
    subparsers = parser.add_subparsers(dest="command", required=True)

    mc = subparsers.add_parser("musicorpus", help="Convert a MusiCorpus dataset to a Zeus dataset folder.")
    mc.add_argument("--input", required=True, help="Path to the MusiCorpus dataset, e.g. datasets/UFAL.OmniOMR.")
    mc.add_argument("--output", required=True, help="Output Zeus dataset folder, e.g. datasets/omniomr. Must not exist yet.")
    mc.add_argument("--take-staves", action="store_true", help="Include staves as samples.")
    mc.add_argument("--take-grandstaves", action="store_true", help="Include grandstaves as samples.")
    mc.set_defaults(func=musicorpus)

    pk = subparsers.add_parser("pickle", help="Bundle the samples of samples files into .pickle files next to them.")
    pk.add_argument("samples_files", nargs="+", help="Samples files to pickle, e.g. datasets/omniomr/samples.train.txt.")
    pk.add_argument("--image-suffix", default="", help="Suffix of the image file names of the samples (default: none).")
    pk.add_argument("--benevolent", action="store_true", help="Skip samples with missing images instead of failing.")
    pk.set_defaults(func=pickle)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
