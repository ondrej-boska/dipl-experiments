"""
MusiCorpus datasets, read with the `musicorpus` package (see musicorpus/docs/python-api.md and the specification).

A benchmark compares a gold dataset with a predicted one. Both are MusiCorpus dataset folders: page folders named by
page, each optionally with `Staves/`, `Grandstaves/` and `Systems/` subdivision folders, and transcriptions as
`transcription.<format>` at the page or subdivision level. A gold sample is paired with the predicted sample of the
same page, subdivision kind and subdivision name.

A predicted dataset need not have a `musicorpus.json`: it is the output of a model rather than a published dataset
(`omr-benchmark predict` copies the gold splits files and writes its provenance to `prediction.json` instead).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from musicorpus import Dataset, Page, Sample, Subdivision

SPLIT_ALIASES = {"dev": "validation", "val": "validation"}
"""Other names for the specification's split names, as used in training scripts."""


def natural_key(text: str) -> list[int | str]:
    """Sorts 'Staves/10' after 'Staves/9' (musicorpus sorts subdivision names lexicographically)."""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", text)]


def open_dataset(path: str | Path, require_manifest: bool) -> Dataset:
    """
    Opens a dataset folder. A gold dataset should be a MusiCorpus dataset with a `musicorpus.json`; one without it
    (e.g. an excerpt of a dataset) is read anyway, with a warning.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset folder '{path}' does not exist.")
    if Dataset.is_dataset(path):
        return Dataset.load(path)
    if require_manifest:
        print(f"Warning: '{path}' has no musicorpus.json; reading it as a MusiCorpus dataset anyway.", file=sys.stderr)
    return Dataset(path)


def select_pages(dataset: Dataset, split: str | None = None, splits_variant: str | None = None) -> list[str]:
    """
    Names of the pages to benchmark, in natural order: every page folder, or the pages of one split of
    `splits.json` (or of `splits.<variant>.json`). A split may name pages that are not on disk (e.g. in an excerpt of
    a dataset); those are left out with a warning.
    """
    if split is None:
        return sorted(dataset.page_names, key=natural_key)

    splits = dataset.splits(splits_variant)
    name = split if split in splits else SPLIT_ALIASES.get(split, split)
    if name not in splits:
        raise KeyError(
            f"There is no split '{split}' in '{dataset.splits_path(splits_variant)}'; "
            f"it has: {', '.join(splits.split_names())}."
        )
    pages = [page for page in splits[name] if dataset.page(page).exists()]
    absent = len(splits[name]) - len(pages)
    if absent:
        print(f"Warning: {absent} of the {len(splits[name])} pages of split '{name}' are not in '{dataset.path}' "
              "and are left out.", file=sys.stderr)
    return sorted(pages, key=natural_key)


def samples_of(page: Page, kind: str | None) -> list[Sample]:
    """The page itself (kind None), or its subdivisions of one kind ('Staves', ...), in natural order."""
    if kind is None:
        return [page]
    return sorted(page.subdivisions(kind), key=lambda s: natural_key(s.path.name))


def sample_id(sample: Sample) -> str:
    """The sample's path within its page: '' for the page itself, e.g. 'Staves/3' for a staff."""
    return f"{sample.kind}/{sample.name}" if isinstance(sample, Subdivision) else ""


def counterpart(sample: Sample, dataset: Dataset) -> Sample:
    """The sample at the same place in another dataset, whether or not it exists there."""
    page = dataset.page(sample.page_name)
    if isinstance(sample, Subdivision):
        return Subdivision(path=page.path / sample.kind / sample.name, page=page, kind=sample.kind)
    return page
