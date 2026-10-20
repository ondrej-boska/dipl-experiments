"""
`omr-benchmark predict` against the fake Musibot server of the python client's own tests, which answers over an
httpx mock transport. Skipped unless the client is installed and the musibot repository is checked out beside this
package (../musibot).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from omr_benchmark.dataset import open_dataset, select_pages
from omr_benchmark.predict import predict

pytest.importorskip("musibot.client")
FAKE_SERVER = Path(__file__).resolve().parents[2] / "musibot/components/python-client/tests/fake_server.py"
if not FAKE_SERVER.is_file():
    pytest.skip("the musibot repository is not checked out beside omr-benchmark", allow_module_level=True)
# Loaded by path: its package is also called `tests`
_spec = importlib.util.spec_from_file_location("musibot_fake_server", FAKE_SERVER)
fake_server = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fake_server  # dataclasses look their module up
_spec.loader.exec_module(fake_server)
API_HOST, FakeServer = fake_server.API_HOST, fake_server.FakeServer


class StaffModelServer(FakeServer):
    """A Musibot running a staff model: each execution writes a transcription beside its input image."""

    def _start_execution(self, request, page_id):
        for path in json.loads(request.content)["input"]:
            folder = path.rpartition("/")[0]
            if "Staves/2/" in path:
                self.outcome = "failed"  # this staff fails
            else:
                self.outcome = "completed"
                self.objects[f"{page_id}/{folder}/transcription.musicxml"] = f"<{folder}/>".encode()
                self.objects[f"{page_id}/{folder}/debug.png"] = b"not kept"
        return super()._start_execution(request, page_id)


@pytest.fixture
def gold(tmp_path: Path) -> Path:
    gold = tmp_path / "TEST.Gold"
    for staff in ("1", "2", "3"):
        (gold / "page-1" / "Staves" / staff).mkdir(parents=True)
        (gold / "page-1" / "Staves" / staff / "image.jpg").write_bytes(b"jpeg " + staff.encode())
    (gold / "page-1" / "Staves" / "4").mkdir()  # no image: not submitted
    (gold / "musicorpus.json").write_text("{}")
    (gold / "splits.json").write_text(json.dumps({"train": [], "validation": [], "test": ["page-1"]}))
    return gold


def run(gold: Path, pred: Path, server: FakeServer, **kwargs) -> dict:
    dataset = open_dataset(gold, require_manifest=True)
    return predict(
        dataset, pred, select_pages(dataset), pipeline=("zeus-baseline", "v1"), unit="staff",
        api_url=API_HOST, api_token="secret", concurrency=2, transport=server.transport(), **kwargs,
    )


def test_predict_writes_one_transcription_per_staff(gold, tmp_path):
    pred = tmp_path / "pred"
    server = StaffModelServer(auto_finish=True)
    record = run(gold, pred, server)

    assert (pred / "page-1/Staves/1/transcription.musicxml").read_bytes() == b"<Staves/1/>"
    assert (pred / "page-1/Staves/3/transcription.musicxml").read_bytes() == b"<Staves/3/>"
    assert not (pred / "page-1/Staves/2").exists()  # failed
    assert not (pred / "page-1/Staves/1/debug.png").exists()  # not a kept suffix
    assert (pred / "splits.json").read_text() == (gold / "splits.json").read_text()

    # one execution per staff, each with exactly its own image
    assert sorted(body["input"] for body in server.started) == [
        ["Staves/1/image.jpg"], ["Staves/2/image.jpg"], ["Staves/3/image.jpg"]
    ]
    assert (record["submitted"], record["completed"], record["without_image"]) == (3, 2, 1)
    assert [(f["page"], f["sample"]) for f in record["failed"]] == [("page-1", "Staves/2")]
    assert json.loads((pred / "prediction.json").read_text())["runs"][0]["pipeline"]["name"] == "zeus-baseline"


def test_skip_existing_submits_only_what_is_missing(gold, tmp_path):
    pred = tmp_path / "pred"
    run(gold, pred, StaffModelServer())
    server = StaffModelServer()
    record = run(gold, pred, server, skip_existing=True)

    assert [body["input"] for body in server.started] == [["Staves/2/image.jpg"]]
    assert record["skipped"] == 2
    assert len(json.loads((pred / "prediction.json").read_text())["runs"]) == 2
