"""
Writes the predictions of a model as a predicted MusiCorpus dataset, for MusicXML-level evaluation (OMR-NED) with
omr-benchmark against the gold MusiCorpus dataset the Zeus dataset was converted from.

A Zeus sample converted from MusiCorpus (zeus/musicorpus) is named after its place in the MusiCorpus dataset,
`samples/<page>/<Kind>_<name>`, e.g. `samples/<page>/Staves_3`. Its prediction goes to that place in the predicted
dataset, beside where the gold dataset has the staff's image and transcription:

    <predictions dir>/
      prediction.json                       # one record per written split: model, checkpoint, counts, failures
      <page>/Staves/<n>/transcription.lmx   # the predicted LMX
      <page>/Staves/<n>/transcription.musicxml  # the MusicXML decoded from it, as Zeus does

A prediction the LMX decoder cannot turn into MusicXML keeps its LMX but gets no MusicXML, so omr-benchmark scores it
as missing (OMR-NED 1), as it would a failed Musibot execution.

The models are trained on the LMX of the *normalized* gold: the conversion to Zeus replaces invisible header clefs
with a G clef (F and G for grandstaves) and invisible key signatures with C major (zeus/musicorpus/convert_musicxml.py),
since a staff crop does not show them. Against the original MusiCorpus gold, every pitch of such a staff is wrong even
in a perfect prediction. So the gold to evaluate against is the normalized MusicXML the conversion wrote beside the
LMX, exported as a MusiCorpus dataset with `normalized-gold`:

    python -m musvit_zeus_comparison.predictions normalized-gold datasets/omniomr datasets/gold/omniomr
    omr-benchmark evaluate datasets/gold/omniomr experiment_results/zeus_predictions/omniomr --split test \
        -b staff-omr-ned -o benchmark_results/omniomr-test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

try:
    from musvit_zeus_comparison.zeus.musicxml.lmx_to_musicxml import LmxDecodingError, lmx_to_musicxml
except ImportError:
    from zeus.musicxml.lmx_to_musicxml import LmxDecodingError, lmx_to_musicxml


LMX_FILE = "transcription.lmx"
MUSICXML_FILE = "transcription.musicxml"

_SAMPLE_NAME = re.compile(r"^samples/(?P<page>[^/]+)/(?P<kind>Staves|Grandstaves|Systems)_(?P<name>[^/]+)$")


def musicorpus_location(sample_name: str) -> Path | None:
    """
    Path of a Zeus sample within its MusiCorpus dataset, e.g. 'samples/<page>/Staves_3' -> '<page>/Staves/3';
    None for samples not converted from MusiCorpus.
    """
    match = _SAMPLE_NAME.match(sample_name)
    if match is None:
        return None
    return Path(match["page"]) / match["kind"] / match["name"]


def write_predictions(
    sample_names: list[str],
    predictions: list[str],
    output_dir: Path,
    run_info: dict,
) -> dict:
    """
    Writes the predicted LMX of each sample and the MusicXML decoded from it into the predicted MusiCorpus dataset
    `output_dir`, overwriting earlier predictions of the same samples, and records the run in its prediction.json.
    Returns the record.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    failures: list[dict] = []
    unplaced: list[str] = []
    for sample_name, lmx in zip(sample_names, predictions, strict=True):
        location = musicorpus_location(sample_name)
        if location is None:
            unplaced.append(sample_name)
            continue
        sample_dir = output_dir / location
        sample_dir.mkdir(parents=True, exist_ok=True)
        # Written before the conversion, so that a prediction the decoder rejects can still be looked at
        (sample_dir / LMX_FILE).write_text(lmx + "\n", encoding="utf-8")
        written += 1
        try:
            musicxml = lmx_to_musicxml(lmx)
        except LmxDecodingError as e:
            # No MusicXML, rather than a stale one from an earlier run: omr-benchmark scores the staff as missing
            (sample_dir / MUSICXML_FILE).unlink(missing_ok=True)
            failures.append({"sample": location.as_posix(), "error": str(e)})
            continue
        (sample_dir / MUSICXML_FILE).write_text(musicxml, encoding="utf-8")

    record = {
        **run_info,
        "written": datetime.now().isoformat(timespec="seconds"),
        "samples": len(sample_names),
        "lmx_written": written,
        "musicxml_written": written - len(failures),
        "not_musicorpus_samples": len(unplaced),
        "musicxml_failed": failures,
    }
    _record_run(output_dir, record)

    print(f"  Wrote {written:,} predictions to '{output_dir}' ({written - len(failures):,} with MusicXML).")
    if failures:
        print(f"  Warning: {len(failures):,} predicted LMX could not be decoded into MusicXML "
              f"and score as missing, e.g. '{failures[0]['sample']}': {failures[0]['error']}")
    if unplaced:
        print(f"  Warning: {len(unplaced):,} sample(s) are not named after a MusiCorpus location and were not "
              f"written, e.g. '{unplaced[0]}'.")
    return record


def _record_run(output_dir: Path, record: dict):
    """
    Appends the record to prediction.json, in the format of omr-benchmark predict, so that a predicted dataset
    written by several evaluations (e.g. the validation and the test split) keeps them all.
    """
    path = output_dir / "prediction.json"
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"] if path.is_file() else []
    runs.append(record)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"runs": runs}, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def export_normalized_gold(zeus_dataset_dir: Path, output_dir: Path):
    """
    Exports the normalized gold MusicXML of a Zeus dataset folder converted from MusiCorpus (the loose
    `samples/<page>/<Kind>_<name>.musicxml` files, not the pickles) as a MusiCorpus dataset, with a splits.json made
    of the pages of its `samples.<split>.txt` files. Staves the conversion skipped (LMX it could not encode) are not
    in it, so they are not scored.
    """
    splits: dict[str, list[str]] = {}
    for samples_file in sorted(zeus_dataset_dir.glob("samples.*.txt")):
        split = samples_file.name[len("samples."):-len(".txt")]
        if split == "all":
            continue
        locations = [musicorpus_location(line.strip()) for line in samples_file.read_text(encoding="utf-8").splitlines()]
        splits[split] = sorted({location.parts[0] for location in locations if location is not None})
    if not splits:
        raise FileNotFoundError(f"'{zeus_dataset_dir}' has no samples.<split>.txt files; is it a Zeus dataset folder?")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for musicxml in sorted((zeus_dataset_dir / "samples").glob("*/*.musicxml")):
        location = musicorpus_location(f"samples/{musicxml.parent.name}/{musicxml.stem}")
        if location is None:
            continue
        (output_dir / location).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(musicxml, output_dir / location / MUSICXML_FILE)
        copied += 1
    (output_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    counts = ", ".join(f"{split}: {len(pages):,}" for split, pages in splits.items())
    print(f"Exported {copied:,} normalized gold transcriptions to '{output_dir}' (pages per split: {counts}).")


def main():
    parser = argparse.ArgumentParser(description="Predicted MusiCorpus datasets and their normalized gold.")
    commands = parser.add_subparsers(dest="command", required=True)
    gold = commands.add_parser(
        "normalized-gold",
        help="Export the normalized gold MusicXML of a Zeus dataset folder as a MusiCorpus dataset, the gold to "
             "evaluate the predictions against with omr-benchmark.",
    )
    gold.add_argument("zeus_dataset", help="Zeus dataset folder made by 'zeus musicorpus', e.g. datasets/omniomr.")
    gold.add_argument("output", help="MusiCorpus dataset folder to write, e.g. datasets/gold/omniomr.")
    args = parser.parse_args()
    export_normalized_gold(Path(args.zeus_dataset), Path(args.output))


if __name__ == "__main__":
    main()
