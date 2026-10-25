"""Resumability tests for bench.run_bench against a fake server and
MockLocalClient -- no GPU, no Docker, no real binaries needed."""

import json

import pytest

from quantlab import bench
from quantlab.client import MockLocalClient

FAKE_TASKS = [
    {"id": "t1", "category": "math", "task": "What is 2 + 2?",
     "grader": "numeric", "expected": "4"},
    {"id": "t2", "category": "math", "task": "What is 3 + 3?",
     "grader": "numeric", "expected": "6"},
    {"id": "t3", "category": "math", "task": "What is 4 + 4?",
     "grader": "numeric", "expected": "8"},
    {"id": "t4", "category": "math", "task": "What is 5 + 5?",
     "grader": "numeric", "expected": "10"},
]


class FakeLlamaServer:
    def __init__(self, rung, ngl):
        self.rung = rung
        self.ngl = ngl
        self.process = None

    def start(self, known_pids=frozenset()):
        pass

    def stop(self):
        pass

    def client(self):
        return MockLocalClient()


@pytest.fixture
def fake_bench(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "LlamaServer", FakeLlamaServer)
    monkeypatch.setattr(bench, "_ngl_for", lambda rung: 0)
    monkeypatch.setattr(bench, "load_tasks",
                        lambda slice_name, task_ids=None: [dict(t) for t in FAKE_TASKS])
    return tmp_path / "grid.jsonl"


def _rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def test_resume_no_duplicates_and_full_coverage(fake_bench):
    out_path = fake_bench

    bench.run_bench(rungs=["Q4_K_M"], out_path=out_path, stop_after=2)
    assert len(_rows(out_path)) == 2

    bench.run_bench(rungs=["Q4_K_M"], out_path=out_path)
    rows = _rows(out_path)
    assert len(rows) == 4

    ids = [r["task_id"] for r in rows]
    assert sorted(ids) == sorted(t["id"] for t in FAKE_TASKS)
    assert len(ids) == len(set(ids))  # no duplicates
    assert all(r["error"] is None for r in rows)
    assert all(r["mean_logprob"] is not None for r in rows)


def test_resumed_run_does_not_repeat_already_done_tasks(fake_bench):
    out_path = fake_bench
    bench.run_bench(rungs=["Q4_K_M"], out_path=out_path, stop_after=1)
    first_task_id = _rows(out_path)[0]["task_id"]

    bench.run_bench(rungs=["Q4_K_M"], out_path=out_path, stop_after=1)
    rows = _rows(out_path)
    assert len(rows) == 2
    assert rows[1]["task_id"] != first_task_id  # a different task was picked


def test_interrupting_and_resuming_three_times_still_covers_everything(fake_bench):
    out_path = fake_bench
    for _ in range(4):
        bench.run_bench(rungs=["Q4_K_M"], out_path=out_path, stop_after=1)
    rows = _rows(out_path)
    assert len(rows) == 4
    ids = [r["task_id"] for r in rows]
    assert sorted(ids) == sorted(t["id"] for t in FAKE_TASKS)
    assert len(ids) == len(set(ids))


def test_rerun_after_full_completion_does_nothing(fake_bench):
    out_path = fake_bench
    bench.run_bench(rungs=["Q4_K_M"], out_path=out_path)
    assert len(_rows(out_path)) == 4

    bench.run_bench(rungs=["Q4_K_M"], out_path=out_path)
    assert len(_rows(out_path)) == 4  # nothing new written, all already done


def test_multiple_rungs_each_get_full_coverage(fake_bench):
    out_path = fake_bench
    bench.run_bench(rungs=["Q4_K_M", "Q8_0"], out_path=out_path)
    rows = _rows(out_path)
    assert len(rows) == 8
    for rung in ("Q4_K_M", "Q8_0"):
        rung_ids = sorted(r["task_id"] for r in rows if r["rung"] == rung)
        assert rung_ids == sorted(t["id"] for t in FAKE_TASKS)
