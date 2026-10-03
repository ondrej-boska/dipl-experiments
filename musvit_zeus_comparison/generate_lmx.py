"""
Script to generate official Linearized MusicXML (.lmx) transcriptions from MusicXML.

Follows the official Zeus dataset conversion specification (zeus.musicorpus.convert_musicxml):
1. Applies Zeus-style invisible header normalization (clef, key, time signature).
2. Encodes normalized part to LMX tokens using the official 'linearized-musicxml' package.
3. Automatically catches and skips unconvertible samples (e.g. unpitched percussion staves).
"""

import argparse
import sys
from pathlib import Path

from lmx.musicxml.io.read_musicxml_tree_from_file import read_musicxml_tree_from_file
from lmx.musicxml.omitted_staff_header.normalize_invisible_header_clef import (
    normalize_invisible_header_clef,
)
from lmx.musicxml.omitted_staff_header.normalize_invisible_key_signature import (
    normalize_invisible_key_signature,
)
from lmx.musicxml.omitted_staff_header.normalize_invisible_time_signature import (
    normalize_invisible_time_signature,
)
from lmx.musicxml.pitch.Clef import F_CLEF, G_CLEF
from lmx.tokenization.Encoder import Encoder


def convert_musicxml_file_to_lmx(mxml_path: Path, verbose: bool = False) -> str:
    """
    Normalizes a single-staff MusicXML file and encodes it to an LMX string.
    Matches zeus.musicorpus.convert_musicxml in the official Zeus implementation.
    """
    musicxml_tree = read_musicxml_tree_from_file(mxml_path)
    part_elements = musicxml_tree.findall("part")
    if not part_elements:
        raise ValueError(f"No <part> found in {mxml_path}")

    part_element = part_elements[0]
    is_grandstaff = part_element.findtext("measure/attributes/staves", "1") == "2"

    # Normalize invisible headers matching Zeus semantics
    musicxml_tree.getroot().remove(part_element)
    part_element = normalize_invisible_header_clef(
        part_element=part_element,
        desired_clef=[G_CLEF, F_CLEF] if is_grandstaff else G_CLEF,
        when_clef_visible="dont-normalize",
    )
    part_element = normalize_invisible_key_signature(
        part_element=part_element,
        desired_key=0,
        when_key_visible="dont-normalize",
    )
    part_element = normalize_invisible_time_signature(
        part_element=part_element,
        desired_time=None,
        when_time_visible="dont-normalize",
    )
    musicxml_tree.getroot().append(part_element)

    # Encode to LMX tokens.
    # Pass errout=sys.stderr if verbose; otherwise errout=None buffers warnings
    # into an internal StringIO so non-fatal encoder warnings (e.g. transpose) don't pollute the CLI.
    encoder = Encoder(errout=sys.stderr if verbose else None)
    encoder.process_part(part_element)
    return " ".join(encoder.output_tokens)


def generate_lmx_dataset(
    dataset_dir: str | Path,
    force: bool = False,
    quiet: bool = False,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """
    Converts all transcription.musicxml in dataset_dir to transcription.lmx.

    Returns:
        (converted_count, skipped_count, error_count)
    """
    base_path = Path(dataset_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    # Find all transcription.musicxml files
    xml_files = sorted(list(base_path.glob("*/Staves/*/transcription.musicxml")))
    if not xml_files:
        xml_files = sorted(list(base_path.rglob("transcription.musicxml")))
    if not xml_files:
        xml_files = sorted(list(base_path.rglob("*.musicxml")))

    if not xml_files:
        print(f"No MusicXML files found in '{dataset_dir}'.")
        return 0, 0, 0

    print(f"Found {len(xml_files)} MusicXML files in '{dataset_dir}'.")

    converted = 0
    skipped = 0
    errors = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(xml_files, desc="Generating LMX", disable=quiet)
    except ImportError:
        iterator = xml_files

    for xml_path in iterator:
        lmx_path = xml_path.with_name("transcription.lmx")
        if not force and lmx_path.exists() and lmx_path.stat().st_size > 0:
            skipped += 1
            continue

        try:
            lmx_str = convert_musicxml_file_to_lmx(xml_path, verbose=verbose)
            lmx_path.write_text(lmx_str, encoding="utf-8")
            converted += 1
        except Exception as e:
            # Zeus convention: skip unconvertible samples (e.g. unpitched percussion or corrupt staves)
            errors += 1
            if not quiet:
                print(f"\n[Warning] Skipped unconvertible sample {xml_path}: {e}")

    print("\nLMX Generation Complete:")
    print(f"  - Converted: {converted}")
    print(f"  - Skipped (already existed): {skipped}")
    print(f"  - Incompatible / errors (skipped): {errors}")

    return converted, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="Generate official Linearized MusicXML (.lmx) transcriptions from MusicXML."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="UFAL.OmniOMR",
        help="Path to dataset containing Staves and transcription.musicxml files (default: UFAL.OmniOMR).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing transcription.lmx files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bar.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show non-fatal LMX encoder warnings (e.g. ignored attributes).",
    )

    args = parser.parse_args()
    generate_lmx_dataset(
        dataset_dir=args.dataset_dir,
        force=args.force,
        quiet=args.quiet,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

