"""
Script to generate official Linearized MusicXML (.lmx) transcriptions from MusicXML.

Traverses dataset staves (e.g. OmniOMR.Small), applies Zeus-style invisible header
normalization (invisible G-clef, key=0, time=None), and compiles MusicXML into .lmx
using the official 'linearized-musicxml' package.

Once generated, dataset.py and extract_features.py automatically detect and use these
.lmx files for 100% token parity with Zeus benchmarks.
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


def convert_musicxml_file_to_lmx(mxml_path: Path) -> str:
    """
    Normalizes a single-staff MusicXML file and encodes it to an LMX string.
    Follows Zeus specification in zeus.musicorpus.convert_musicxml.
    """
    musicxml_tree = read_musicxml_tree_from_file(mxml_path)
    part_elements = musicxml_tree.findall("part")
    if not part_elements:
        raise ValueError(f"No <part> found in {mxml_path}")

    part_element = part_elements[0]
    is_grandstaff = part_element.findtext("measure/attributes/staves", "1") == "2"

    # Pre-clean known MuseScore artifacts that break LMX OnsetVisitor / normalization:
    # 1. Figured bass (<figured-bass>) is ignored by LMX Encoder anyway, but causes
    #    NotImplementedError in OnsetVisitor if it has duration tags.
    for m in part_element.findall("measure"):
        for fb in list(m.findall("figured-bass")):
            m.remove(fb)

    # 2. Redundant invisible clefs, keys, or time signatures repeated by MuseScore
    #    in subsequent measures (e.g. during measure repeats). LMX normalization
    #    strictly forbids invisible headers anywhere other than measure 0 (head).
    measures = part_element.findall("measure")
    for m in measures[1:]:
        for attr in list(m.findall("attributes")):
            for tag in ("clef", "key", "time"):
                for elem in list(attr.findall(tag)):
                    if elem.attrib.get("print-object") == "no":
                        attr.remove(elem)
            if len(attr) == 0:
                m.remove(attr)

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

    # Encode to LMX tokens
    encoder = Encoder(errout=sys.stderr)
    encoder.process_part(part_element)
    return " ".join(encoder.output_tokens)


def generate_lmx_dataset(
    dataset_dir: str | Path,
    force: bool = False,
    quiet: bool = False,
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
            lmx_str = convert_musicxml_file_to_lmx(xml_path)
            lmx_path.write_text(lmx_str, encoding="utf-8")
            converted += 1
        except Exception as e:
            errors += 1
            if not quiet:
                print(f"\n[Warning] Failed on {xml_path}: {e}")

    print("\nLMX Generation Complete:")
    print(f"  - Converted: {converted}")
    print(f"  - Skipped (already existed): {skipped}")
    print(f"  - Errors: {errors}")

    return converted, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="Generate official Linearized MusicXML (.lmx) transcriptions from MusicXML."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="OmniOMR.Small",
        help="Path to dataset containing Staves and transcription.musicxml files (default: OmniOMR.Small).",
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

    args = parser.parse_args()
    generate_lmx_dataset(
        dataset_dir=args.dataset_dir,
        force=args.force,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
