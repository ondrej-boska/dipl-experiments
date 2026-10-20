"""
Running benchmarks: pairing the gold samples with the predicted ones, and scoring every pair.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from musicorpus import Dataset

from omr_benchmark.benchmarks import Benchmark
from omr_benchmark.dataset import counterpart, natural_key, sample_id, samples_of
from omr_benchmark.metrics import ItemResult, Metric, status_counts, summed_details


@dataclass(frozen=True)
class Item:
    """One gold transcription and the predicted one at the same place, if there is any."""

    page: str
    sample: str
    """Path of the sample within the page: '' for the page itself, e.g. 'Staves/3'."""
    gold: Path
    pred: Path | None


@dataclass
class BenchmarkResult:
    benchmark: Benchmark
    items: list[Item]
    results: list[ItemResult]
    gold_without_transcription: int = 0
    """Gold samples of the benchmark's level that have no transcription in its format; not items."""
    unmatched_predictions: list[str] = field(default_factory=list)
    """Predicted transcriptions with no gold counterpart, e.g. a staff the gold page does not have; not scored."""
    seconds: float = 0.0

    def summary(self) -> dict:
        return {
            "benchmark": self.benchmark.name,
            "metric": self.benchmark.metric.name,
            "items": len(self.items),
            **status_counts(self.results),
            **self.benchmark.metric.aggregate(self.results),
            "gold_without_transcription": self.gold_without_transcription,
            "unmatched_predictions": len(self.unmatched_predictions),
            "edit_breakdown": summed_details(self.results),
            "seconds": round(self.seconds, 1),
        }


def collect_items(benchmark: Benchmark, gold: Dataset, pred: Dataset, pages: list[str]) -> BenchmarkResult:
    """The benchmark's items over the given pages, before scoring."""
    items: list[Item] = []
    gold_without_transcription = 0
    unmatched: list[str] = []
    for page in pages:
        gold_ids = set()
        for gold_sample in samples_of(gold.page(page), benchmark.kind):
            gold_path = gold_sample.transcription_path(benchmark.format)
            if gold_path is None:
                gold_without_transcription += 1
                continue
            gold_ids.add(sample_id(gold_sample))
            pred_path = counterpart(gold_sample, pred).transcription_path(benchmark.format)
            items.append(Item(page, sample_id(gold_sample), gold_path, pred_path))

        pred_page = pred.page(page)
        if pred_page.exists():
            for pred_sample in samples_of(pred_page, benchmark.kind):
                if sample_id(pred_sample) not in gold_ids and pred_sample.transcription_path(benchmark.format):
                    unmatched.append(f"{page}/{sample_id(pred_sample)}".rstrip("/"))
    return BenchmarkResult(benchmark, items, [], gold_without_transcription, unmatched)


def _compute(metric: Metric, item: Item) -> ItemResult:
    return metric.compute(item.gold, item.pred)


def score_items(metric: Metric, items: list[Item], jobs: int, label: str) -> list[ItemResult]:
    """Scores the items in `jobs` processes (1: in this process), in their order, printing the progress."""
    results: list[ItemResult] = []
    report_every = max(1, len(items) // 20)

    def progress(done: int):
        if done % report_every == 0 or done == len(items):
            print(f"  {label}: {done}/{len(items)}", file=sys.stderr, flush=True)

    if jobs <= 1:
        for item in items:
            results.append(_compute(metric, item))
            progress(len(results))
        return results

    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for result in pool.map(_compute, [metric] * len(items), items, chunksize=4):
            results.append(result)
            progress(len(results))
    return results


def run_benchmark(benchmark: Benchmark, gold: Dataset, pred: Dataset, pages: list[str], jobs: int) -> BenchmarkResult:
    start = time.time()
    result = collect_items(benchmark, gold, pred, pages)
    result.results = score_items(benchmark.metric, result.items, jobs, f"{pred.name} {benchmark.name}")
    result.seconds = time.time() - start
    result.unmatched_predictions.sort(key=natural_key)
    return result
