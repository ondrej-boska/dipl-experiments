"""
The omr-benchmark command line:

    omr-benchmark list
    omr-benchmark evaluate GOLD PRED [PRED ...] [--benchmarks ...] [--split test] [--jobs 8] [--output-dir DIR]
    omr-benchmark predict GOLD PRED --pipeline NAME VERSION --unit staff [--split test] [--api-url URL]
    omr-benchmark combine SUMMARY SUMMARY [...] [--names NAME ...] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from omr_benchmark import __version__
from omr_benchmark.benchmarks import ALL_BENCHMARKS, DEFAULT_BENCHMARKS, resolve
from omr_benchmark.combine import combine_runs
from omr_benchmark.dataset import open_dataset, select_pages
from omr_benchmark.evaluate import run_benchmark
from omr_benchmark.predict import DEFAULT_SUFFIXES, UNIT_KINDS, predict
from omr_benchmark.report import summary_table, write_combined_summary, write_prediction_results, write_summary


def _unique_labels(paths: list[Path]) -> list[str]:
    """Folder names of the predicted datasets, made unique (they name the result folders)."""
    labels: list[str] = []
    for path in paths:
        label, n = path.resolve().name, 2
        while label in labels:
            label, n = f"{path.resolve().name}-{n}", n + 1
        labels.append(label)
    return labels


def evaluate_command(args: argparse.Namespace):
    benchmarks = resolve(args.benchmarks)
    gold = open_dataset(args.gold, require_manifest=True)
    pages = select_pages(gold, args.split, args.splits_variant)
    pred_paths = [Path(p) for p in args.pred]
    labels = _unique_labels(pred_paths)
    output_dir = Path(args.output_dir or Path("benchmark_results") / gold.name)
    print(f"Gold: {gold.path} ({len(pages)} pages). Benchmarks: {', '.join(b.name for b in benchmarks)}.")

    import musicdiff  # noqa: F401  (fail before any work if the metric dependencies are missing)
    from importlib.metadata import version

    summaries: dict[str, list[dict]] = {}
    for label, pred_path in zip(labels, pred_paths, strict=True):
        pred = open_dataset(pred_path, require_manifest=False)
        results = [run_benchmark(b, gold, pred, pages, args.jobs) for b in benchmarks]
        write_prediction_results(results, output_dir / label)
        summaries[label] = [{"prediction": str(pred_path), **r.summary()} for r in results]

    run = {
        "gold": str(gold.path),
        "split": args.split,
        "splits_variant": args.splits_variant,
        "pages": len(pages),
        "benchmarks": [b.name for b in benchmarks],
        "omr_benchmark_version": __version__,
        "musicdiff_version": version("musicdiff"),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    print()
    print(write_summary(output_dir, run, summaries))
    print(f"Results written to '{output_dir}'.")


def predict_command(args: argparse.Namespace):
    api_url = args.api_url or os.environ.get("MUSIBOT_API_URL")
    api_token = args.api_token or os.environ.get("MUSIBOT_API_TOKEN")
    if not api_url or not api_token:
        raise SystemExit("Give the Musibot API with --api-url and --api-token, or MUSIBOT_API_URL and MUSIBOT_API_TOKEN.")

    gold = open_dataset(args.gold, require_manifest=True)
    pages = select_pages(gold, args.split, args.splits_variant)
    print(f"Running {args.pipeline[0]} {args.pipeline[1]} on {len(pages)} pages of {gold.path}, one {args.unit} "
          f"per execution.")
    run = predict(
        gold,
        Path(args.pred),
        pages,
        pipeline=(args.pipeline[0], args.pipeline[1]),
        unit=args.unit,
        api_url=api_url,
        api_token=api_token,
        suffixes=tuple(args.keep),
        concurrency=args.concurrency,
        timeout_seconds=args.timeout,
        skip_existing=args.skip_existing,
        run_settings={"split": args.split, "splits_variant": args.splits_variant},
    )
    print(f"{run['completed']} completed, {len(run['failed'])} failed, {run['skipped']} skipped (already predicted), "
          f"{run['without_image']} without an image, in {run['seconds']}s. Recorded in '{args.pred}/prediction.json'.")


def combine_command(args: argparse.Namespace):
    paths = [Path(p) for p in args.summaries]
    runs = []
    for path in paths:
        if path.is_dir():
            path = path / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"There is no '{path}'.")
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    try:
        summaries = combine_runs(runs, args.names)
    except ValueError as e:
        raise SystemExit(f"omr-benchmark: {e}")
    for label, benchmarks in summaries.items():
        print(f"{label}: {benchmarks[0]['prediction'] if benchmarks else 'no benchmark in every run'}")

    print()
    if args.output_dir is None:
        print(summary_table(summaries))
        return
    run = {
        "combined": [str(p) for p in paths],
        "runs": [r["run"] for r in runs],
        "omr_benchmark_version": __version__,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    print(write_combined_summary(Path(args.output_dir), run, summaries))
    print(f"Results written to '{args.output_dir}'.")


def list_command(_: argparse.Namespace):
    for name, benchmark in ALL_BENCHMARKS.items():
        default = " (default)" if name in DEFAULT_BENCHMARKS else ""
        print(f"{name}{default}: {benchmark.description}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="omr-benchmark", description="Benchmarks of OMR models on MusiCorpus datasets.")
    parser.add_argument("--version", action="version", version=f"omr-benchmark {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_page_selection(p: argparse.ArgumentParser):
        p.add_argument("--split", help="Only the pages of this split of the gold splits.json: train, validation "
                                       "(or dev, val), test, or any other split the file has. Default: all pages.")
        p.add_argument("--splits-variant", help="Read the split from splits.<variant>.json instead.")

    evaluate = commands.add_parser("evaluate", help="Compare predicted datasets with a gold dataset.")
    evaluate.add_argument("gold", help="The gold MusiCorpus dataset folder.")
    evaluate.add_argument("pred", nargs="+", help="Predicted dataset folder(s), with the gold dataset's layout.")
    evaluate.add_argument("--benchmarks", "-b", nargs="+", default=list(DEFAULT_BENCHMARKS),
                          help=f"Benchmarks to run (default: {' '.join(DEFAULT_BENCHMARKS)}); see 'omr-benchmark list'.")
    add_page_selection(evaluate)
    evaluate.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1,
                          help="Processes comparing transcriptions (default: all CPUs; 1 compares in this process).")
    evaluate.add_argument("--output-dir", "-o", help="Where to write the results (default: benchmark_results/<gold name>).")
    evaluate.set_defaults(run=evaluate_command)

    predict_parser = commands.add_parser("predict", help="Make a predicted dataset by running a Musibot pipeline.")
    predict_parser.add_argument("gold", help="The gold MusiCorpus dataset folder, whose images are recognized.")
    predict_parser.add_argument("pred", help="The predicted dataset folder to write.")
    predict_parser.add_argument("--pipeline", nargs=2, metavar=("NAME", "VERSION"), required=True,
                                help="The Musibot pipeline, or model (its implicit pipeline), and its version.")
    predict_parser.add_argument("--unit", choices=list(UNIT_KINDS), default="page",
                                help="What one execution gets, to fit the pipeline's signature: the page image "
                                     "(default), or one staff, grandstaff or system image.")
    add_page_selection(predict_parser)
    predict_parser.add_argument("--api-url", help="The Musibot API (default: MUSIBOT_API_URL).")
    predict_parser.add_argument("--api-token", help="The Musibot API token (default: MUSIBOT_API_TOKEN).")
    predict_parser.add_argument("--keep", nargs="+", default=list(DEFAULT_SUFFIXES),
                                help=f"Suffixes of the produced files to write (default: {' '.join(DEFAULT_SUFFIXES)}).")
    predict_parser.add_argument("--concurrency", type=int, default=4,
                                help="Executions in flight; size it to the Musibot workers (default: 4).")
    predict_parser.add_argument("--timeout", type=float, default=600.0, help="Seconds per execution (default: 600).")
    predict_parser.add_argument("--skip-existing", action="store_true",
                                help="Skip samples that already have a prediction, e.g. to continue an interrupted run.")
    predict_parser.set_defaults(run=predict_command)

    combine = commands.add_parser(
        "combine",
        help="Combine evaluate runs, e.g. one per gold dataset, into one benchmark over all their items.",
        description="Combines the results of several evaluate runs, e.g. one per gold dataset, as if their items "
                    "had been one benchmark: the corpus OMR-NED weighs each run by its symbols and the mean OMR-NED by "
                    "its items. The predicted datasets of the runs are matched by position, so every run must list "
                    "the same models in the same order.",
    )
    combine.add_argument("summaries", nargs="+", help="The summary.json of each run, or its output folder.")
    combine.add_argument("--names", nargs="+", help="Names of the combined predicted datasets, in order "
                                                    "(default: their names in the runs, joined with ' + ').")
    combine.add_argument("--output-dir", "-o", help="Where to write summary.md and summary.json (default: only print).")
    combine.set_defaults(run=combine_command)

    list_parser = commands.add_parser("list", help="List the benchmarks.")
    list_parser.set_defaults(run=list_command)

    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (FileNotFoundError, KeyError) as e:
        print(f"omr-benchmark: {e.args[0] if e.args else e}", file=sys.stderr)
        sys.exit(1)
