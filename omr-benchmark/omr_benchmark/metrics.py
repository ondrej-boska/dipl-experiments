"""
Metrics: how one gold file and one predicted file are compared, and how the comparisons add up.

A metric scores one item (a gold file and the predicted file at the same path, which may be missing) and aggregates
the items of a benchmark. A missing or unreadable prediction is scored, not skipped, as an empty prediction: a model
must not score better by failing. A gold file that cannot be read is reported and left out, since it says nothing
about the model.

Adding a metric means writing a class with the `Metric` interface here and using it in benchmarks.py.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

OK = "ok"
MISSING = "missing"  # no predicted file at the gold file's path
UNREADABLE = "unreadable"  # the predicted file exists but could not be parsed; scored as an empty prediction
GOLD_ERROR = "gold-error"  # the gold file could not be parsed or is empty; not scored


@dataclass
class ItemResult:
    """The comparison of one gold file with its prediction."""

    status: str
    values: dict[str, float | int] = field(default_factory=dict)
    """Per-item numbers, the metric's `columns`."""
    details: dict[str, int] = field(default_factory=dict)
    """A breakdown, e.g. edit operations by kind; summed over the benchmark."""
    error: str | None = None


class Metric(Protocol):
    name: str
    description: str
    columns: tuple[str, ...]
    """Names of the per-item values, in the order of the per-item CSV."""
    primary: str
    """The aggregate value that summary tables show."""

    def compute(self, gold: Path, pred: Path | None) -> ItemResult: ...

    def aggregate(self, results: list[ItemResult]) -> dict[str, float | int]: ...


# ==============================================================================
# OMR-NED
# ==============================================================================

_converter21_registered = False


def _parse_score(path: Path, accept_syntax_errors: bool):
    """The first score of a file readable by music21 (MusicXML, MEI, Humdrum, ...), as musicdiff reads it."""
    global _converter21_registered
    import converter21
    import music21 as m21

    if not _converter21_registered:
        converter21.register()
        _converter21_registered = True
    score = m21.converter.parse(path, forceSource=True, acceptSyntaxErrors=accept_syntax_errors)
    if isinstance(score, m21.stream.Opus):
        score = score.scores[0]
    return score


@dataclass(frozen=True)
class OmrNed:
    """
    OMR normalized edit distance (OMR-NED, Martinez-Sevilla et al., 2025), computed by musicdiff: the edit distance
    between the annotated scores, divided by the number of symbols of both (so 0 is identical and 1 is nothing right).
    Over a benchmark, `corpus_omr_ned` divides the summed edit distances by the summed symbol counts, as musicdiff's
    own batch evaluation does, and `mean_omr_ned` averages the per-item values.
    """

    detail: int | None = None
    """musicdiff DetailLevel; None for its default."""

    name: str = "OMR-NED"
    description: str = "OMR normalized edit distance (musicdiff); 0 is a perfect transcription, 1 a completely wrong one."
    columns: tuple[str, ...] = ("gold_symbols", "pred_symbols", "edit_distance", "omr_ned")
    primary: str = "corpus_omr_ned"

    def _detail(self):
        from musicdiff import DetailLevel

        return DetailLevel.Default if self.detail is None else self.detail

    def compute(self, gold: Path, pred: Path | None) -> ItemResult:
        import music21 as m21
        from musicdiff.annotation import AnnScore
        from musicdiff.comparison import Comparison
        from musicdiff.visualization import Visualization

        try:
            gold_score = _parse_score(gold, accept_syntax_errors=False)
        except Exception as e:
            return ItemResult(GOLD_ERROR, error=f"Could not read the gold file: {e}")
        if len(gold_score.parts) == 0:
            return ItemResult(GOLD_ERROR, error="The gold file has no parts.")

        status, error = OK, None
        if pred is None:
            status, pred_score = MISSING, m21.stream.Score()
        else:
            try:
                pred_score = _parse_score(pred, accept_syntax_errors=True)
            except Exception as e:
                status, error, pred_score = UNREADABLE, f"Could not read the prediction: {e}", m21.stream.Score()

        detail = self._detail()
        ann_gold, ann_pred = AnnScore(gold_score, detail), AnnScore(pred_score, detail)
        ops, edit_distance = Comparison.annotated_scores_diff(ann_pred, ann_gold)
        gold_symbols, pred_symbols = ann_gold.notation_size(), ann_pred.notation_size()
        breakdown = Visualization.get_edit_distances_dict(ops, ann_pred.num_syntax_errors_fixed, detail)
        return ItemResult(
            status,
            values={
                "gold_symbols": gold_symbols,
                "pred_symbols": pred_symbols,
                "edit_distance": edit_distance,
                "omr_ned": Visualization.get_omr_ned(edit_distance, pred_symbols, gold_symbols),
            },
            details={name.removesuffix(" OMR-ED"): count for name, count in breakdown.items() if count},
            error=error,
        )

    def aggregate(self, results: list[ItemResult]) -> dict[str, float | int]:
        scored = [r for r in results if r.status != GOLD_ERROR]
        gold_symbols = sum(r.values["gold_symbols"] for r in scored)
        pred_symbols = sum(r.values["pred_symbols"] for r in scored)
        edit_distance = sum(r.values["edit_distance"] for r in scored)
        both = gold_symbols + pred_symbols
        return {
            "corpus_omr_ned": edit_distance / both if both else 0.0,
            "mean_omr_ned": statistics.fmean(r.values["omr_ned"] for r in scored) if scored else 0.0,
            "gold_symbols": gold_symbols,
            "pred_symbols": pred_symbols,
            "edit_distance": edit_distance,
        }


def status_counts(results: list[ItemResult]) -> dict[str, int]:
    counts = Counter(r.status for r in results)
    return {status: counts.get(status, 0) for status in (OK, MISSING, UNREADABLE, GOLD_ERROR)}


def summed_details(results: list[ItemResult]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for r in results:
        if r.status != GOLD_ERROR:
            total.update(r.details)
    return dict(total.most_common())
