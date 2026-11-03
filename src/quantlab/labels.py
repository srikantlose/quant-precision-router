"""Required-precision labels (plan Sec 9.2).

req(t) is the cheapest rung whose own correctness and a bounded number of
exceptions among costlier non-reference rungs together justify calling the
task "solved" there. It is derived purely from measured (correct,
mean_logprob) pairs on the bench slice -- never from the tier, and never
from agreement with a lower rung (that would reintroduce the circularity
flagged in review).

tau is fit on the train split's F16-correct rows only; test tasks are
labelled with that same tau, never a separately-fit one (D19: no leakage).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .quants import LADDER

REFERENCE = "F16"


class LabelsError(Exception):
    pass


def compute_tau(rows: list[dict], quantile: float) -> float:
    """quantile-th percentile of mean_logprob over F16-correct rows."""
    margins = [r["mean_logprob"] for r in rows if r["mean_logprob"] is not None]
    if not margins:
        raise LabelsError("no F16-correct rows with a non-null mean_logprob to fit tau on")
    return float(np.quantile(margins, quantile))


def solved(row: dict, tau: float) -> bool:
    if not row.get("correct"):
        return False
    margin = row.get("mean_logprob")
    return margin is None or margin >= tau


def required_precision(s: list[bool], max_exc: int = 1) -> int | None:
    """s is indexed by LADDER position (len 7); idx 6 = F16."""
    for b in range(6):
        if s[b] and sum(1 for x in s[b + 1:6] if not x) <= max_exc:
            return b
    return 6 if s[6] else None


def _load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_labels(
    grid_path: str | Path = "results/raw/grid.jsonl",
    tasks_path: str | Path = "data/tasks/all.jsonl",
    out_path: str | Path = "results/labels.json",
) -> dict[str, Any]:
    cfg = load_config()
    max_exc = cfg["labels"]["max_exceptions_above"]
    main_quantile = cfg["labels"]["tau_quantile"]
    sensitivity_quantiles = cfg["labels"]["tau_sensitivity"]

    tasks = {t["id"]: t for t in _load_jsonl(tasks_path)}
    grid = _load_jsonl(grid_path)
    bench = [r for r in grid if r["slice"] == "bench"]

    n_null_logprob = sum(1 for r in bench if r["mean_logprob"] is None)

    by_task: dict[str, dict[str, dict]] = {}
    for r in bench:
        by_task.setdefault(r["task_id"], {})[r["rung"]] = r

    train_f16_correct = [
        rows[REFERENCE] for tid, rows in by_task.items()
        if REFERENCE in rows and rows[REFERENCE].get("correct")
        and tasks.get(tid, {}).get("split") == "train"
    ]

    def _label_all(tau: float) -> dict[str, Any]:
        req_by_task: dict[str, int | None] = {}
        unsolvable = {"train": 0, "test": 0}
        dist: dict[str, dict[str, int]] = {}
        for tid, rows in by_task.items():
            task_meta = tasks.get(tid)
            if task_meta is None:
                continue
            split = task_meta["split"]
            category = task_meta["category"]
            s = [solved(rows[rung], tau) if rung in rows else False for rung in LADDER]
            req = required_precision(s, max_exc=max_exc)
            req_by_task[tid] = req
            if req is None:
                unsolvable[split] += 1
            cat_dist = dist.setdefault(category, {})
            key = "unsolvable" if req is None else str(req)
            cat_dist[key] = cat_dist.get(key, 0) + 1
        return {"req": req_by_task, "unsolvable": unsolvable, "distribution": dist}

    tau = compute_tau(train_f16_correct, main_quantile)
    main_result = _label_all(tau)

    sensitivity: dict[str, Any] = {}
    for q in sensitivity_quantiles:
        tau_q = compute_tau(train_f16_correct, q)
        res_q = _label_all(tau_q)
        sensitivity[str(q)] = {
            "tau": tau_q,
            "unsolvable": res_q["unsolvable"],
            "distribution": res_q["distribution"],
        }

    result = {
        "tau": tau,
        "tau_quantile": main_quantile,
        "n_train_f16_correct_rows": len(train_f16_correct),
        "n_null_mean_logprob": n_null_logprob,
        "max_exceptions_above": max_exc,
        "req": main_result["req"],
        "unsolvable": main_result["unsolvable"],
        "distribution": main_result["distribution"],
        "sensitivity": sensitivity,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
