import json
from pathlib import Path

import music21 as m21
import pytest


def write_score(path: Path, tiny_notation: str):
    """A small MusicXML file from music21 tinyNotation, e.g. '4/4 c4 d e f'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    m21.converter.parse(f"tinyNotation: {tiny_notation}").write("musicxml", fp=path)


STAFF_A = "4/4 c4 d e f g1"
STAFF_B = "4/4 e4 f g a b1"
STAFF_C = "3/4 c'4 b a g2."


@pytest.fixture
def datasets(tmp_path: Path) -> tuple[Path, Path]:
    """
    A gold dataset of two pages and a predicted one:
    - page-1: page and staff 1 identical, staff 2 wrong, staff 3 missing, staff 4 (gold) has no transcription
    - page-2: page prediction unreadable, staff 1 identical, an extra predicted staff 9
    """
    gold, pred = tmp_path / "TEST.Gold", tmp_path / "pred"
    (gold / "musicorpus.json").parent.mkdir(parents=True)
    (gold / "musicorpus.json").write_text("{}", encoding="utf-8")
    (gold / "splits.json").write_text(json.dumps({"train": ["page-2"], "validation": [], "test": ["page-1"]}))

    write_score(gold / "page-1/transcription.musicxml", STAFF_A)
    write_score(gold / "page-1/Staves/1/transcription.musicxml", STAFF_A)
    write_score(gold / "page-1/Staves/2/transcription.musicxml", STAFF_B)
    write_score(gold / "page-1/Staves/3/transcription.musicxml", STAFF_C)
    (gold / "page-1/Staves/4").mkdir(parents=True)
    write_score(gold / "page-2/transcription.musicxml", STAFF_B)
    write_score(gold / "page-2/Staves/1/transcription.musicxml", STAFF_C)

    write_score(pred / "page-1/transcription.musicxml", STAFF_A)
    write_score(pred / "page-1/Staves/1/transcription.musicxml", STAFF_A)
    write_score(pred / "page-1/Staves/2/transcription.musicxml", STAFF_C)
    (pred / "page-2").mkdir(parents=True)
    (pred / "page-2/transcription.musicxml").write_text("<score-partwise><oops", encoding="utf-8")
    write_score(pred / "page-2/Staves/1/transcription.musicxml", STAFF_C)
    write_score(pred / "page-2/Staves/9/transcription.musicxml", STAFF_A)
    return gold, pred
