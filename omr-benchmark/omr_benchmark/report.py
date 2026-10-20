"""
Writing benchmark results:

    <output dir>/
      summary.md                 # one table: every predicted dataset x benchmark
      summary.json               # the same with everything else: run settings, edit breakdowns, timings
                                 # (also all that `omr-benchmark combine` writes, for its combined runs)
      <prediction>/<benchmark>.csv   # one row per item, worst first
      <prediction>/<benchmark>.unmatched.txt   # predicted transcriptions without a gold counterpart, if any
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from omr_benchmark.evaluate import BenchmarkResult


def write_items_csv(result: BenchmarkResult, path: Path):
    metric = result.benchmark.metric
    rows = sorted(
        zip(result.items, result.results, strict=True),
        key=lambda pair: (-pair[1].values.get("omr_ned", 0.0), pair[0].page, pair[0].sample),
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "sample", "status", *metric.columns, "gold", "pred", "error"])
        for item, item_result in rows:
            values = [item_result.values.get(column, "") for column in metric.columns]
            values = [f"{v:.6f}" if isinstance(v, float) else v for v in values]
            writer.writerow([
                item.page, item.sample, item_result.status, *values,
                str(item.gold), str(item.pred or ""), item_result.error or "",
            ])


def write_prediction_results(results: list[BenchmarkResult], directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for result in results:
        write_items_csv(result, directory / f"{result.benchmark.name}.csv")
        unmatched_path = directory / f"{result.benchmark.name}.unmatched.txt"
        if result.unmatched_predictions:
            unmatched_path.write_text("\n".join(result.unmatched_predictions) + "\n", encoding="utf-8")
        elif unmatched_path.exists():
            unmatched_path.unlink()


def summary_table(summaries: dict[str, list[dict]]) -> str:
    """Markdown table of the primary values, one row per predicted dataset and benchmark."""
    lines = [
        "| Prediction | Benchmark | Items | OK | Missing | Unreadable | Corpus OMR-NED | Mean OMR-NED |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for prediction, benchmarks in summaries.items():
        for s in benchmarks:
            lines.append(
                f"| {prediction} | {s['benchmark']} | {s['items']} | {s['ok']} | {s['missing']} | {s['unreadable']} "
                f"| {s['corpus_omr_ned']:.4f} | {s['mean_omr_ned']:.4f} |"
            )
    return "\n".join(lines)


def write_summary(output_dir: Path, run: dict, summaries: dict[str, list[dict]]):
    split = run["split"] + (f" ({run['splits_variant']})" if run["splits_variant"] else "") if run["split"] else "all pages"
    gold = f"Gold: `{run['gold']}`, {split}, {run['pages']} pages. {run['finished']}."
    return _write_summary_files(output_dir, run, summaries, gold)


def write_combined_summary(output_dir: Path, run: dict, summaries: dict[str, list[dict]]):
    """The summary of `omr-benchmark combine`, whose run lists the runs it combines."""
    gold = "Combined runs:\n\n" + "\n".join(
        f"- `{r['gold']}`, {r['split'] or 'all pages'}, {r['pages']} pages (`{path}`)"
        for path, r in zip(run["combined"], run["runs"], strict=True)
    ) + f"\n\n{run['finished']}."
    return _write_summary_files(output_dir, run, summaries, gold)


def _write_summary_files(output_dir: Path, run: dict, summaries: dict[str, list[dict]], gold: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps({"run": run, "predictions": summaries}, indent=2), encoding="utf-8"
    )

    notes = []
    for prediction, benchmarks in summaries.items():
        for s in benchmarks:
            if s["gold-error"]:
                notes.append(f"- {prediction} / {s['benchmark']}: {s['gold-error']} gold file(s) could not be read "
                             "and are not scored.")
            if s["unmatched_predictions"]:
                notes.append(f"- {prediction} / {s['benchmark']}: {s['unmatched_predictions']} predicted "
                             "transcription(s) have no gold counterpart and are not scored.")
    text = (
        f"# Benchmark results\n\n"
        f"{gold}\n\n"
        f"{summary_table(summaries)}\n\n"
        "OMR-NED: 0 is a perfect transcription, 1 a completely wrong one. *Corpus* divides the summed edit distances "
        "by the summed symbol counts, *Mean* averages the items. Missing and unreadable predictions are scored as "
        "empty transcriptions.\n"
    )
    if notes:
        text += "\n" + "\n".join(notes) + "\n"
    (output_dir / "summary.md").write_text(text, encoding="utf-8")
    return text
