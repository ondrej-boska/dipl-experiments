"""
The benchmarks, and how to name them.

A benchmark is a metric applied at one level of a MusiCorpus page: the page itself or one kind of its subdivisions
(staves, grandstaves, systems), on one transcription format. Its items are the gold samples of that level that have
a transcription; each is compared with the predicted transcription at the same place.

Adding a benchmark means writing it here and naming it in `ALL_BENCHMARKS`; the CLI resolves whatever names this
module publishes.
"""

from __future__ import annotations

from dataclasses import dataclass

from omr_benchmark.metrics import Metric, OmrNed


@dataclass(frozen=True)
class Benchmark:
    name: str
    """As it appears on the command line and in the reports."""
    description: str
    kind: str | None
    """The subdivision kind the items are ('Staves', 'Grandstaves', 'Systems'), or None for whole pages."""
    format: str
    """The transcription format compared: the suffix of `transcription.<format>`."""
    metric: Metric


PAGE_OMR_NED = Benchmark(
    name="page-omr-ned",
    description="OMR-NED of the page-level transcription.musicxml.",
    kind=None,
    format="musicxml",
    metric=OmrNed(),
)

STAFF_OMR_NED = Benchmark(
    name="staff-omr-ned",
    description="OMR-NED of each Staves/<n>/transcription.musicxml.",
    kind="Staves",
    format="musicxml",
    metric=OmrNed(),
)

GRANDSTAFF_OMR_NED = Benchmark(
    name="grandstaff-omr-ned",
    description="OMR-NED of each Grandstaves/<n>/transcription.musicxml.",
    kind="Grandstaves",
    format="musicxml",
    metric=OmrNed(),
)

SYSTEM_OMR_NED = Benchmark(
    name="system-omr-ned",
    description="OMR-NED of each Systems/<n>/transcription.musicxml.",
    kind="Systems",
    format="musicxml",
    metric=OmrNed(),
)

ALL_BENCHMARKS = {b.name: b for b in (PAGE_OMR_NED, STAFF_OMR_NED, GRANDSTAFF_OMR_NED, SYSTEM_OMR_NED)}

DEFAULT_BENCHMARKS = ("page-omr-ned", "staff-omr-ned")


def resolve(names: list[str]) -> list[Benchmark]:
    unknown = [name for name in names if name not in ALL_BENCHMARKS]
    if unknown:
        raise KeyError(f"Unknown benchmark(s): {', '.join(unknown)}. Available: {', '.join(ALL_BENCHMARKS)}.")
    return [ALL_BENCHMARKS[name] for name in names]
