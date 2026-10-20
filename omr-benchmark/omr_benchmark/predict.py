"""
Making a predicted dataset: running a Musibot pipeline (or a single model, as its implicit pipeline) over the pages of
a gold dataset, and writing what it produces into a new dataset folder of the same layout.

The unit decides what one Musibot execution gets, and must fit the pipeline's signature:
- 'page': the page image (`image.jpg`), for page-level pipelines, which find the staves themselves.
- 'staff', 'grandstaff', 'system': one subdivision image (e.g. `Staves/3/image.jpg`) per execution, for models that
  read one staff (`Staves/{s}/image.jpg`), like Zeus.

Every produced file with one of the kept suffixes is written to the same path in the predicted dataset; the gold
splits files are copied, and each run is recorded in `prediction.json`.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from musicorpus import Dataset, Sample

from omr_benchmark.dataset import counterpart, sample_id, samples_of

UNIT_KINDS = {"page": None, "staff": "Staves", "grandstaff": "Grandstaves", "system": "Systems"}

DEFAULT_SUFFIXES = (".musicxml", ".lmx")


@dataclass(frozen=True)
class JobKey:
    page: str
    sample: str
    """Path of the sample within the page: '' for the page itself, e.g. 'Staves/3'."""


def _has_output(sample: Sample, suffixes: tuple[str, ...], recursive: bool) -> bool:
    if not sample.path.is_dir():
        return False
    files = sample.path.rglob("*") if recursive else sample.path.iterdir()
    return any(f.is_file() and f.name.endswith(suffixes) for f in files)


def _jobs(gold: Dataset, pred: Dataset, pages: list[str], kind: str | None, suffixes: tuple[str, ...],
          skip_existing: bool, counters: dict) -> Iterator:
    """Pulled lazily by the client, so that only the images of the pages in flight are in memory."""
    from musibot.client import BatchJob

    for page in pages:
        for sample in samples_of(gold.page(page), kind):
            image = sample.image_path()
            if image is None:
                counters["without_image"] += 1
                continue
            if skip_existing and _has_output(counterpart(sample, pred), suffixes, recursive=kind is None):
                counters["skipped"] += 1
                continue
            prefix = f"{sample_id(sample)}/" if kind else ""
            counters["submitted"] += 1
            yield BatchJob(key=JobKey(page, sample_id(sample)), input={f"{prefix}{image.name}": image.read_bytes()})


def _copy_splits(gold: Dataset, pred_root: Path):
    for splits_file in gold.path.glob("splits*.json"):
        target = pred_root / splits_file.name
        if not target.exists():
            shutil.copyfile(splits_file, target)


def predict(
    gold: Dataset,
    pred_root: Path,
    pages: list[str],
    pipeline: tuple[str, str],
    unit: str,
    api_url: str,
    api_token: str,
    suffixes: tuple[str, ...] = DEFAULT_SUFFIXES,
    concurrency: int = 4,
    timeout_seconds: float = 600.0,
    skip_existing: bool = False,
    run_settings: dict | None = None,
    transport=None,
) -> dict:
    """Runs the pipeline on every sample of `pages`; `transport` is an httpx transport for tests."""
    try:
        from musibot.client import MusibotClient
    except ImportError as e:
        raise SystemExit(
            "omr-benchmark predict needs the Musibot python client; see 'Installation' in the omr-benchmark README."
        ) from e

    kind = UNIT_KINDS[unit]
    pred_root.mkdir(parents=True, exist_ok=True)
    pred = Dataset(pred_root)
    _copy_splits(gold, pred_root)

    def keep(file) -> bool:
        return file.path.endswith(suffixes)

    counters = {"submitted": 0, "skipped": 0, "without_image": 0, "completed": 0}
    failures: list[dict] = []
    started = datetime.now()
    start = time.time()

    with MusibotClient(musibot_api_url=api_url, api_token=api_token, transport=transport) as client:
        jobs = _jobs(gold, pred, pages, kind, suffixes, skip_existing, counters)
        for result in client.process_pages(
            jobs, pipeline=pipeline, output=keep, concurrency=concurrency, timeout_seconds=timeout_seconds
        ):
            key: JobKey = result.key
            where = f"{key.page}/{key.sample}".rstrip("/")
            if result.failed:
                failures.append({"page": key.page, "sample": key.sample, "error": str(result.error)})
                print(f"failed {where}: {result.error}", file=sys.stderr, flush=True)
                continue
            for path, content in result.files.items():
                target = pred_root / key.page / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            counters["completed"] += 1
            print(f"done {where} ({len(result.files)} file(s), {result.seconds:.1f}s)", flush=True)

    run = {
        "gold": str(gold.path),
        "pipeline": {"name": pipeline[0], "version": pipeline[1]},
        "unit": unit,
        **(run_settings or {}),
        "started": started.isoformat(timespec="seconds"),
        "seconds": round(time.time() - start, 1),
        **counters,
        "failed": failures,
    }
    _record_run(pred_root, run)
    return run


def _record_run(pred_root: Path, run: dict):
    """Appends the run to prediction.json, so that a dataset made in several runs (--skip-existing) keeps them all."""
    path = pred_root / "prediction.json"
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"] if path.is_file() else []
    runs.append(run)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    os.replace(tmp, path)
