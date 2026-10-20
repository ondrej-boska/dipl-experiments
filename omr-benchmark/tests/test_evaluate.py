import csv
import json

import pytest

from omr_benchmark.benchmarks import PAGE_OMR_NED, STAFF_OMR_NED
from omr_benchmark.cli import main
from omr_benchmark.dataset import open_dataset, select_pages
from omr_benchmark.evaluate import run_benchmark
from omr_benchmark.metrics import GOLD_ERROR, MISSING, OK, UNREADABLE, OmrNed


def by_sample(result):
    return {f"{item.page}/{item.sample}".rstrip("/"): r for item, r in zip(result.items, result.results, strict=True)}


def test_staff_items_are_paired_and_scored(datasets):
    gold, pred = (open_dataset(p, require_manifest=False) for p in datasets)
    result = run_benchmark(STAFF_OMR_NED, gold, pred, select_pages(gold), jobs=1)
    items = by_sample(result)

    assert set(items) == {"page-1/Staves/1", "page-1/Staves/2", "page-1/Staves/3", "page-2/Staves/1"}
    assert items["page-1/Staves/1"].status == OK and items["page-1/Staves/1"].values["omr_ned"] == 0.0
    assert items["page-1/Staves/2"].status == OK and 0.0 < items["page-1/Staves/2"].values["omr_ned"] <= 1.0
    # A missing prediction is scored as an empty one: every gold symbol is an edit
    missing = items["page-1/Staves/3"]
    assert missing.status == MISSING and missing.values["omr_ned"] == 1.0
    assert missing.values["edit_distance"] == missing.values["gold_symbols"]

    assert result.gold_without_transcription == 1  # page-1/Staves/4
    assert result.unmatched_predictions == ["page-2/Staves/9"]


def test_unreadable_prediction_counts_as_empty(datasets):
    gold, pred = (open_dataset(p, require_manifest=False) for p in datasets)
    items = by_sample(run_benchmark(PAGE_OMR_NED, gold, pred, select_pages(gold), jobs=1))
    assert items["page-1"].status == OK and items["page-1"].values["omr_ned"] == 0.0
    assert items["page-2"].status == UNREADABLE and items["page-2"].values["omr_ned"] == 1.0
    assert items["page-2"].error


def test_aggregate_divides_summed_edits_by_summed_symbols(datasets):
    gold, pred = (open_dataset(p, require_manifest=False) for p in datasets)
    result = run_benchmark(STAFF_OMR_NED, gold, pred, select_pages(gold), jobs=1)
    summary = result.summary()
    values = [r.values for r in result.results]
    edits = sum(v["edit_distance"] for v in values)
    symbols = sum(v["gold_symbols"] + v["pred_symbols"] for v in values)
    assert summary["corpus_omr_ned"] == pytest.approx(edits / symbols)
    assert summary["mean_omr_ned"] == pytest.approx(sum(v["omr_ned"] for v in values) / len(values))
    assert (summary["items"], summary["ok"], summary["missing"], summary["unreadable"]) == (4, 3, 1, 0)


def test_unreadable_gold_is_reported_not_scored(tmp_path):
    gold = tmp_path / "gold.musicxml"
    gold.write_text("not music", encoding="utf-8")
    result = OmrNed().compute(gold, None)
    assert result.status == GOLD_ERROR
    assert OmrNed().aggregate([result])["corpus_omr_ned"] == 0.0


def test_split_selection(datasets):
    gold = open_dataset(datasets[0], require_manifest=True)
    assert select_pages(gold) == ["page-1", "page-2"]
    assert select_pages(gold, "test") == ["page-1"]
    assert select_pages(gold, "dev") == []  # alias of 'validation'
    with pytest.raises(KeyError):
        select_pages(gold, "holdout")


def test_split_pages_not_on_disk_are_left_out(datasets, capsys):
    gold_path = datasets[0]
    (gold_path / "splits.json").write_text('{"train": [], "validation": [], "test": ["page-1", "page-404"]}')
    assert select_pages(open_dataset(gold_path, require_manifest=True), "test") == ["page-1"]
    assert "1 of the 2 pages" in capsys.readouterr().err


def test_parallel_scoring_matches_sequential(datasets):
    gold, pred = (open_dataset(p, require_manifest=False) for p in datasets)
    sequential = run_benchmark(STAFF_OMR_NED, gold, pred, select_pages(gold), jobs=1)
    parallel = run_benchmark(STAFF_OMR_NED, gold, pred, select_pages(gold), jobs=2)
    assert [r.values for r in parallel.results] == [r.values for r in sequential.results]


def test_cli_writes_reports(datasets, tmp_path, capsys):
    gold, pred = datasets
    out = tmp_path / "results"
    main(["evaluate", str(gold), str(pred), str(gold), "--split", "test", "-j", "1", "-o", str(out)])

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["run"]["pages"] == 1
    assert set(summary["predictions"]) == {"pred", "TEST.Gold"}  # the gold dataset as a perfect prediction
    assert all(s["corpus_omr_ned"] == 0.0 for s in summary["predictions"]["TEST.Gold"])
    assert "| pred | staff-omr-ned | 3 |" in (out / "summary.md").read_text(encoding="utf-8")

    with open(out / "pred" / "staff-omr-ned.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["sample"] for row in rows][0] == "Staves/3"  # worst first
    assert rows[0]["status"] == MISSING


def test_combine_matches_one_run_over_all_items(datasets, tmp_path, capsys):
    gold, pred = datasets
    for split in ("test", "train"):
        main(["evaluate", str(gold), str(pred), str(gold), "--split", split, "-j", "1", "-o", str(tmp_path / split)])
    main(["evaluate", str(gold), str(pred), str(gold), "-j", "1", "-o", str(tmp_path / "all")])
    out = tmp_path / "combined"
    main(["combine", str(tmp_path / "test"), str(tmp_path / "train" / "summary.json"), "--names", "pred", "gold",
          "-o", str(out)])

    combined = json.loads((out / "summary.json").read_text(encoding="utf-8"))["predictions"]
    whole = json.loads((tmp_path / "all" / "summary.json").read_text(encoding="utf-8"))["predictions"]
    assert set(combined) == {"pred", "gold"}
    for name, label in (("pred", "pred"), ("gold", "TEST.Gold")):
        for c, w in zip(combined[name], whole[label], strict=True):
            for key in ("benchmark", "items", "ok", "missing", "unreadable", "gold_symbols", "edit_distance",
                        "unmatched_predictions", "gold_without_transcription", "edit_breakdown"):
                assert c[key] == w[key], key
            assert c["corpus_omr_ned"] == pytest.approx(w["corpus_omr_ned"])
            assert c["mean_omr_ned"] == pytest.approx(w["mean_omr_ned"])
    assert "| pred | staff-omr-ned | 4 |" in (out / "summary.md").read_text(encoding="utf-8")


def test_combine_needs_the_same_models(datasets, tmp_path, capsys):
    gold, pred = datasets
    main(["evaluate", str(gold), str(pred), "-b", "staff-omr-ned", "-j", "1", "-o", str(tmp_path / "one")])
    main(["evaluate", str(gold), str(pred), str(gold), "-b", "staff-omr-ned", "-j", "1", "-o", str(tmp_path / "two")])
    with pytest.raises(SystemExit, match="same models"):
        main(["combine", str(tmp_path / "one"), str(tmp_path / "two")])
