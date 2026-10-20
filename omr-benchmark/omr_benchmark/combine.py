"""
Combining the results of several `omr-benchmark evaluate` runs, e.g. one per gold dataset, into the results of one
benchmark over all their items. The corpus OMR-NED of the combination then weighs each dataset by its symbols, and
the mean OMR-NED by its items, which an average of the per-dataset values would not.

The combination needs no rescoring: the summary.json of a run holds the sums its aggregates are made of. The
predicted datasets of the runs are matched by their position, so each run must list the models in the same order.
"""

from __future__ import annotations

import sys
from collections import Counter

from omr_benchmark.benchmarks import ALL_BENCHMARKS
from omr_benchmark.metrics import GOLD_ERROR, MISSING, OK, UNREADABLE


def combine_runs(runs: list[dict], names: list[str] | None = None) -> dict[str, list[dict]]:
    """
    Combines the summary.json contents of several runs. Returns summaries in the layout of summary.json
    "predictions": per combined model, one summary per benchmark that every run has.
    """
    n_models = {len(run["predictions"]) for run in runs}
    if len(n_models) != 1:
        raise ValueError(
            f"The runs evaluate different numbers of predicted datasets ({', '.join(map(str, sorted(n_models)))}); "
            "each must evaluate the same models, in the same order."
        )
    n_models = n_models.pop()
    if names is not None and len(names) != n_models:
        raise ValueError(f"{len(names)} name(s) given for {n_models} combined predicted dataset(s).")

    versions = {run["run"].get("musicdiff_version") for run in runs}
    if len(versions) > 1:
        print(f"Warning: the runs were scored with different musicdiff versions ({', '.join(map(str, versions))}), "
              "whose OMR-NED may differ.", file=sys.stderr)

    combined: dict[str, list[dict]] = {}
    for model in range(n_models):
        labels, summaries = zip(*(list(run["predictions"].items())[model] for run in runs), strict=True)
        label = names[model] if names is not None else " + ".join(labels)
        in_every_run = set.intersection(*({s["benchmark"] for s in u} for u in summaries))
        benchmarks = [s["benchmark"] for s in summaries[0] if s["benchmark"] in in_every_run]
        dropped = {s["benchmark"] for u in summaries for s in u} - in_every_run
        if dropped:
            print(f"Warning: {label}: {', '.join(sorted(dropped))} not in every run, left out.", file=sys.stderr)
        combined[label] = [
            combine_benchmark([next(s for s in u if s["benchmark"] == name) for u in summaries])
            for name in benchmarks
        ]
    return combined


def combine_benchmark(summaries: list[dict]) -> dict:
    """Combines the summaries of one benchmark, one per run."""
    name = summaries[0]["benchmark"]
    if name not in ALL_BENCHMARKS:
        raise ValueError(f"Unknown benchmark '{name}'; see 'omr-benchmark list'.")
    metric = ALL_BENCHMARKS[name].metric
    breakdown: Counter[str] = Counter()
    for s in summaries:
        breakdown.update(s["edit_breakdown"])
    return {
        "prediction": " + ".join(s["prediction"] for s in summaries),
        "benchmark": name,
        "metric": summaries[0]["metric"],
        **{key: sum(s[key] for s in summaries) for key in ("items", OK, MISSING, UNREADABLE, GOLD_ERROR)},
        **metric.combine(summaries, [s["items"] - s[GOLD_ERROR] for s in summaries]),
        "gold_without_transcription": sum(s["gold_without_transcription"] for s in summaries),
        "unmatched_predictions": sum(s["unmatched_predictions"] for s in summaries),
        "edit_breakdown": dict(breakdown.most_common()),
        "seconds": round(sum(s["seconds"] for s in summaries), 1),
    }
