# omr-benchmark

Benchmarks for OMR models on [MusiCorpus](../musicorpus/spec/musicorpus-specification.md) datasets. A benchmark compares a **gold** dataset with a **predicted** one. Both are MusiCorpus dataset folders, and the predicted one usually comes from running a [Musibot](../musibot/README.md) model or pipeline over the gold pages.

```bash
# 1. Predict: run a Musibot model over the test pages, one staff per execution
omr-benchmark predict datasets/UFAL.OmniOMR predictions/zeus-baseline-2026-10-15 \
    --pipeline zeus-baseline 2026-10-15-153411-e180 --unit staff --split test

# 2. Evaluate: compare one or more predicted datasets with the gold one
omr-benchmark evaluate datasets/UFAL.OmniOMR predictions/zeus-baseline-2026-10-15 predictions/musvit-e2e-2026-10-15 \
    --split test -o benchmark_results/omniomr-test
```

The two steps are independent. `evaluate` reads any folder laid out like the gold dataset, however it was made, so a model that is not in Musibot can be benchmarked by writing its outputs into such a folder.


## Installation

Python 3.12 or newer, which musicdiff requires:

```bash
pip install -e "omr-benchmark"            # evaluate
pip install -e "omr-benchmark[predict]"   # and predict, with the Musibot python client
```

`musicorpus` and the Musibot client are on no package index, so pip installs them from git. In this repository, `pip install ./musicorpus` installs the local checkout instead.


## Benchmarks

| Benchmark | Items |
| --- | --- |
| `page-omr-ned` (default) | The page-level `transcription.musicxml` of every page. |
| `staff-omr-ned` (default) | Every `Staves/<n>/transcription.musicxml`. |
| `grandstaff-omr-ned` | Every `Grandstaves/<n>/transcription.musicxml`. |
| `system-omr-ned` | Every `Systems/<n>/transcription.musicxml`. |

Select them with `--benchmarks`/`-b`, and list them with `omr-benchmark list`.

**OMR-NED** is the OMR normalized edit distance of Martínez-Sevilla et al., computed by [musicdiff](https://github.com/gregchapman-dev/musicdiff) with its default detail level. It is the edit distance between the two annotated scores divided by the number of symbols in both, so 0 is a perfect transcription and 1 a completely wrong one. Each benchmark reports two aggregates:
- **Corpus OMR-NED:** the summed edit distances divided by the summed symbol counts, which is musicdiff's own batch score. Long pages and staves weigh more.
- **Mean OMR-NED:** the mean of the per-item values, where every page or staff counts the same.


## How gold and predicted transcriptions are paired

- **Items** are the gold samples of the benchmark's level (pages, or staves, ...) that have a transcription. Each is compared with the predicted transcription at the same page, subdivision kind and folder name, e.g. `<page>/Staves/3/transcription.musicxml` in both datasets.
- **A missing or unreadable prediction** counts as an empty transcription, with OMR-NED 1. It is never skipped, so a model can't score better by failing. The per-item CSV marks these items `missing` or `unreadable`.
- **A gold file that cannot be read** is reported as `gold-error` and left out, since it says nothing about the model.
- **Predicted transcriptions without a gold counterpart**, e.g. a staff `Staves/13` on a page whose gold has 12 staves, are listed in `<benchmark>.unmatched.txt` and counted in the summary, but not scored.
- **`--split`** restricts both steps to the pages of one split of the gold `splits.json`: `train`, `validation` (also `dev` or `val`), `test`, or any other split the file defines. `--splits-variant NAME` reads `splits.NAME.json` instead. Pages a split names that aren't on disk are left out with a warning, as in an excerpt of a dataset.
- **A gold dataset should have a `musicorpus.json`.** One without it, such as an excerpt, is read anyway with a warning. A predicted dataset doesn't need one.

Staff-level pairing is by folder name. That is exact for staff-level models, which get the gold staff crops. A page-level pipeline that finds the staves itself numbers them on its own, and its staves are compared with the gold staves of the same number, whether or not they're the same staff on the page.


## Results

```
<output dir>/
  summary.md                        # one table: every predicted dataset x benchmark
  summary.json                      # the same, plus run settings, edit breakdowns by kind, timings
  <prediction>/<benchmark>.csv      # one row per item, worst first
  <prediction>/<benchmark>.unmatched.txt
```

The edit breakdown in `summary.json` sums musicdiff's edit operations over the benchmark (wrong notes, missing measures, wrong clefs, ...), which shows *what* a model gets wrong. Comparing transcriptions is CPU-bound and runs in `--jobs` processes, all CPUs by default.


## Predicting with Musibot

`predict` runs a Musibot pipeline, or a single model through its implicit pipeline, over the gold images and writes the produced `.musicxml` and `.lmx` files (`--keep`) into the predicted dataset at the same paths. `--unit` decides what one execution gets and must fit the pipeline's signature:
- `page` (default): the page image, for page-level pipelines.
- `staff`, `grandstaff`, `system`: one subdivision image per execution, e.g. `Staves/3/image.jpg`, for models like Zeus that declare `Staves/{s}/image.jpg`.

The API is given by `--api-url` and `--api-token`, or by `MUSIBOT_API_URL` and `MUSIBOT_API_TOKEN`. A failed execution is logged and skipped, and its item then scores as `missing`. `--skip-existing` continues an interrupted run. The gold splits files are copied into the predicted dataset, and every run is appended to its `prediction.json` (pipeline, version, counts, failures), so a predicted dataset records how it was made.


## Adding a benchmark or a metric

A benchmark is a metric applied at one level of the page to one transcription format. To add one at another level or for another format, add a `Benchmark` to `ALL_BENCHMARKS` in [benchmarks.py](omr_benchmark/benchmarks.py). A new metric is a class with the `Metric` interface in [metrics.py](omr_benchmark/metrics.py):
- `compute(gold, pred)` scores one item, where `pred` is `None` when the prediction is missing.
- `aggregate(results)` sums up a benchmark.

The CLI, the reports and the parallel scoring pick both up without further changes.


## Tests

```bash
pip install -e "omr-benchmark[dev,predict]"
pytest omr-benchmark/tests
```

The `predict` tests drive the client against the fake Musibot server of the client's own tests, so they need the `musibot` repository checked out beside this folder. Otherwise they're skipped.
