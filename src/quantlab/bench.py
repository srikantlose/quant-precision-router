"""The resumable benchmark grid runner: run a set of tasks against a set of
quantization rungs and record one graded row per (rung, task) in an
append-only JSONL file.

Resumability: on start, every existing row in the output file is read, and
any (rung, task_id) pair whose LAST row in the file has error=None is
treated as already done and skipped. The file is never rewritten in place --
a later row for the same key always supersedes an earlier one (this matters
for the end-of-rung error retry pass, which appends a fresh row rather than
editing the failed one). Every row is flushed and fsync'd individually, so
an interrupted run loses at most the row in flight, never a already-written
one.

Used by both the offline grid (this module's run_bench) and nothing else --
router.py (a later phase) uses pipeline.py + server.ServerPool directly, not
this module, since it serves one request at a time rather than sweeping a
task list.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from . import pipeline, quants
from .client import LocalClientError
from .config import load_config
from .graders import grade
from .keepawake import keepawake
from .server import LlamaServer, ServerStartError, find_ngl


class BenchError(Exception):
    pass


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # defensive: ignore a truncated trailing line
    return rows


def _done_keys(existing_rows: list[dict]) -> set[tuple[str, str]]:
    """(rung, task_id) pairs whose LAST occurrence in the file succeeded."""
    last_error: dict[tuple[str, str], bool] = {}
    for row in existing_rows:
        key = (row["rung"], row["task_id"])
        last_error[key] = row.get("error") is not None
    return {k for k, had_error in last_error.items() if not had_error}


def load_tasks(slice_name: str, task_ids: Optional[list[str]] = None) -> list[dict]:
    cfg = load_config()
    tasks: list[dict] = []
    if slice_name in ("bench", "all"):
        tasks.extend(_read_jsonl(Path(cfg["paths"]["tasks_all"])))
    if slice_name in ("safety", "all"):
        safety_path = Path(cfg["paths"]["safety_tasks"])
        if not safety_path.is_file():
            if slice_name == "safety":
                raise BenchError(
                    f"safety slice requested but {safety_path} does not exist yet"
                )
        else:
            for row in _read_jsonl(safety_path):
                row = dict(row)
                row.setdefault("slice", "safety")
                tasks.append(row)
    for t in tasks:
        t.setdefault("slice", "bench")

    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t["id"] in wanted]
    return tasks


def _ngl_for(rung: str) -> int:
    cfg = load_config()
    profiles_path = Path(cfg["paths"]["results_dir"]) / "profiles.json"
    if profiles_path.is_file():
        with open(profiles_path, encoding="utf-8") as f:
            profiles = json.load(f)
        if rung in profiles:
            return int(profiles[rung]["ngl"])
    return find_ngl(rung, quants.read_n_layers(rung))


def _generate_row(client, task: dict, rung: str, cfg: dict) -> dict:
    """Run one (rung, task) through the shared pipeline and grade it.
    Retries transient client failures; on exhaustion returns a row with
    `error` set rather than raising."""
    task_text = task["task"]
    slice_name = task.get("slice", "bench")
    retries = cfg["bench"]["retries"]
    backoff_s = cfg["bench"]["retry_backoff_s"]

    category, label, system, max_tokens = pipeline.prepare(task_text)
    lock = quants._load_lock()
    gguf_sha256 = lock.get("files", {}).get(rung, [{}])[0].get("sha256")

    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            result = client.generate(task_text, system_prompt=system, max_tokens=max_tokens)
            latency_s = time.time() - t0
            answer = pipeline.extract(label, result.text)

            row = {
                "rung": rung, "task_id": task["id"], "slice": slice_name,
                "category_true": task.get("category"), "category_pred": category,
                "profile": label, "max_tokens": max_tokens,
                "raw_text": result.text, "answer": answer,
                "correct": None, "refused": None, "safe": None,
                "mean_logprob": result.mean_logprob, "min_logprob": result.min_logprob,
                "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
                "finish_reason": result.finish_reason, "latency_s": round(latency_s, 4),
                "timings": result.timings,
                "error": None, "llama_build": cfg["llama"]["build"],
                "gguf_sha256": gguf_sha256,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if slice_name == "safety":
                from .safety import grade_safety
                refused, safe, orig_correct = grade_safety(task, answer)
                row["refused"] = refused
                row["safe"] = safe
                if orig_correct is not None:
                    row["orig_correct"] = orig_correct
            else:
                row["correct"] = grade(task, answer)
            return row
        except LocalClientError as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff_s)

    return {
        "rung": rung, "task_id": task["id"], "slice": slice_name,
        "category_true": task.get("category"), "category_pred": None,
        "profile": None, "max_tokens": None, "raw_text": None, "answer": None,
        "correct": None, "refused": None, "safe": None,
        "mean_logprob": None, "min_logprob": None,
        "tokens_in": 0, "tokens_out": 0, "finish_reason": None, "latency_s": None,
        "timings": None, "error": str(last_exc), "llama_build": cfg["llama"]["build"],
        "gguf_sha256": gguf_sha256,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_bench(rungs: list[str], slice_name: str = "bench",
              limit: Optional[int] = None, task_ids: Optional[list[str]] = None,
              stop_after: Optional[int] = None, out_path: str | Path = "results/raw/grid.jsonl",
              ) -> dict:
    cfg = load_config()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(slice_name, task_ids)
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        raise BenchError("no tasks to run")

    existing_rows = _read_jsonl(out_path)
    done = _done_keys(existing_rows)

    progress_path = Path(cfg["paths"]["results_dir"]) / "raw" / "grid_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    new_rows_written = 0
    stats = {"rungs": {}}

    with keepawake():
        out_f = open(out_path, "a", encoding="utf-8")
        try:
            for rung in rungs:
                pending = [t for t in tasks if (rung, t["id"]) not in done]
                if not pending:
                    stats["rungs"][rung] = {"done": 0, "errors": 0, "total": 0}
                    continue

                ngl = _ngl_for(rung)
                server = LlamaServer(rung, ngl)
                server.start()
                try:
                    client = server.client()

                    # Warm-up requests (excluded from results/progress).
                    for i in range(cfg["bench"]["warmup_requests"]):
                        try:
                            client.generate("Warm-up request.", system_prompt="Reply OK.",
                                           max_tokens=4)
                        except LocalClientError:
                            pass

                    rung_done = 0
                    rung_total = len(pending)
                    error_keys: list[dict] = []
                    t_start = time.time()

                    for task in pending:
                        row = _generate_row(client, task, rung, cfg)
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
                        os.fsync(out_f.fileno())
                        if row["error"] is not None:
                            error_keys.append(task)
                        else:
                            done.add((rung, task["id"]))
                        rung_done += 1
                        new_rows_written += 1

                        if rung_done % 20 == 0 or rung_done == rung_total:
                            elapsed_min = max((time.time() - t_start) / 60, 1e-9)
                            rate = rung_done / elapsed_min
                            remaining = rung_total - rung_done
                            with open(progress_path, "w", encoding="utf-8") as pf:
                                json.dump({
                                    "rung": rung, "done": rung_done, "total": rung_total,
                                    "rows_per_min": round(rate, 2),
                                    "eta_min": round(remaining / rate, 1) if rate > 0 else None,
                                    "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                }, pf, ensure_ascii=False)

                        if stop_after is not None and new_rows_written >= stop_after:
                            stats["rungs"][rung] = {
                                "done": rung_done, "errors": len(error_keys), "total": rung_total,
                            }
                            return stats

                    # End-of-rung retry pass for rows that errored.
                    still_failing = []
                    for task in error_keys:
                        row = _generate_row(client, task, rung, cfg)
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_f.flush()
                        os.fsync(out_f.fileno())
                        if row["error"] is not None:
                            still_failing.append(task)
                        else:
                            done.add((rung, task["id"]))

                    error_rate = len(still_failing) / rung_total if rung_total else 0.0
                    stats["rungs"][rung] = {
                        "done": rung_total - len(still_failing),
                        "errors": len(still_failing), "total": rung_total,
                    }
                    if error_rate > cfg["bench"]["max_error_rate"]:
                        raise BenchError(
                            f"rung {rung!r} error rate {error_rate:.1%} exceeds "
                            f"max_error_rate {cfg['bench']['max_error_rate']:.1%} "
                            f"({len(still_failing)}/{rung_total} failed)"
                        )
                finally:
                    server.stop()
        finally:
            out_f.close()

    return stats
