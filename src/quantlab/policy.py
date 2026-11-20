"""The selection rule (plan Sec 9.6) -- the one rule that decides which rung
a cell is served at. All three conditions are required: fidelity alone
would hand a cell to a cheap, low-flip rung regardless of whether it is
actually correct; accuracy alone ignores answer churn; neither alone
considers safety. F16 is always admissible (D27, it is the reference).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .quants import LADDER

REFERENCE = "F16"
NON_REFERENCE = [r for r in LADDER if r != REFERENCE]


def choose(
    acc_ref: float, acc_lcb: dict[str, float], flip_ucb: dict[str, float],
    safety_flip_ucb: dict[str, float], cost: dict[str, float],
    delta: float, phi: float, psi: float,
) -> tuple[str, list[str]]:
    admissible = [REFERENCE]
    for b in NON_REFERENCE:
        if acc_lcb[b] < acc_ref - delta:
            continue
        if flip_ucb[b] > phi:
            continue
        if safety_flip_ucb[b] > psi:
            continue
        admissible.append(b)
    chosen = min(admissible, key=lambda b: (cost[b], LADDER.index(b)))
    return chosen, admissible


def compute_safety_floor(safety_flip_ucb: dict[str, float], psi: float) -> str:
    for b in LADDER:
        if b == REFERENCE:
            return REFERENCE
        if safety_flip_ucb.get(b, 1.0) <= psi:
            return b
    return REFERENCE


def apply_floor(rung: str, floor: str) -> str:
    return rung if LADDER.index(rung) >= LADDER.index(floor) else floor


def cost_fn_factory(
    grid_path: str | Path, tasks_path: str | Path, gpu_hourly_rate: float,
    min_rows: int = 5,
) -> Any:
    """Mean train bench latency for (rung, category_pred), converted to
    GPU-hours per 1,000 tasks; falls back to the rung's overall train mean
    when a (rung, category) pair has fewer than min_rows rows."""
    with open(tasks_path, encoding="utf-8") as f:
        tasks = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}

    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    rung_sums: dict[str, float] = {}
    rung_counts: dict[str, int] = {}

    with open(grid_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["slice"] != "bench":
                continue
            task = tasks.get(row["task_id"])
            if task is None or task["split"] != "train":
                continue
            rung = row["rung"]
            cat = row["category_pred"]
            lat = row["latency_s"]
            key = (rung, cat)
            sums[key] = sums.get(key, 0.0) + lat
            counts[key] = counts.get(key, 0) + 1
            rung_sums[rung] = rung_sums.get(rung, 0.0) + lat
            rung_counts[rung] = rung_counts.get(rung, 0) + 1

    def mean_latency(rung: str, category: str) -> float:
        key = (rung, category)
        if counts.get(key, 0) >= min_rows:
            return sums[key] / counts[key]
        return rung_sums[rung] / rung_counts[rung]

    def cost(rung: str, category: str) -> float:
        return mean_latency(rung, category) * 1000 / 3600 * gpu_hourly_rate

    return cost


MIN_SUPPORT_TASKS = 5


def cell_bounds(models: dict, log_bpp: dict[str, float], z: float, cost_fn: Any) -> dict[str, dict]:
    """Accuracy point/LCB and flip UCB for every (category, predicted class) cell from
    fitted pooled models; sparse cells use the category-level (coarse) models.
    Shared by fit_policy and the bootstrap refits so both apply the identical rule."""
    import pandas as pd

    from .pooled import predict_with_bounds, predict_with_bounds_coarse
    from .tasktype import CATEGORIES

    acc_res, flip_res = models["acc_res"], models["flip_res"]
    acc_levels, flip_levels = models["acc_levels"], models["flip_levels"]
    acc_coarse_res, flip_coarse_res = models["acc_coarse_res"], models["flip_coarse_res"]
    cell_support = models["cell_support"]
    out: dict[str, dict] = {}
    seen = set(acc_levels["categories"]) & set(flip_levels["categories"])
    for category in CATEGORIES:
        for pred_class in range(7):
            n_support = cell_support.get(f"{category}|{pred_class}", 0)
            if category not in seen:
                # No training row was routed to this category (possible in a bootstrap
                # resample when a category is rarely predicted): with no evidence only
                # the reference rung is admissible.
                out[f"{category}|{pred_class}"] = {
                    "low_support": True, "n_support_tasks": 0, "no_data": True,
                    "acc_point": {rung: 0.0 for rung in LADDER},
                    "acc_lcb": {rung: -1.0 for rung in LADDER},
                    "flip_ucb": {rung: 1.0 for rung in NON_REFERENCE},
                    "cost": {rung: cost_fn(rung, category) for rung in LADDER},
                }
                continue
            low_support = n_support < MIN_SUPPORT_TASKS
            acc_newdata = pd.DataFrame([
                {"category_pred": category, "pred_class_oof": pred_class, "rung": rung, "log_bpp": log_bpp[rung]}
                for rung in LADDER
            ])
            flip_newdata = pd.DataFrame([
                {"category_pred": category, "pred_class_oof": pred_class, "rung": rung, "log_bpp": log_bpp[rung]}
                for rung in NON_REFERENCE
            ])
            if low_support:
                acc_point, acc_lcb_arr, _ = predict_with_bounds_coarse(acc_coarse_res, acc_newdata, z, acc_levels)
                _, _, flip_ucb_arr = predict_with_bounds_coarse(flip_coarse_res, flip_newdata, z, flip_levels)
            else:
                acc_point, acc_lcb_arr, _ = predict_with_bounds(acc_res, acc_newdata, z, acc_levels)
                _, _, flip_ucb_arr = predict_with_bounds(flip_res, flip_newdata, z, flip_levels)
            out[f"{category}|{pred_class}"] = {
                "low_support": low_support, "n_support_tasks": n_support,
                "acc_point": {rung: float(acc_point[i]) for i, rung in enumerate(LADDER)},
                "acc_lcb": {rung: float(acc_lcb_arr[i]) for i, rung in enumerate(LADDER)},
                "flip_ucb": {rung: float(flip_ucb_arr[i]) for i, rung in enumerate(NON_REFERENCE)},
                "cost": {rung: cost_fn(rung, category) for rung in LADDER},
            }
    return out


def fit_policy(
    pooled_models_path: str | Path = "results/raw/pooled_models.joblib",
    profiles_path: str | Path = "results/profiles.json",
    safety_path: str | Path = "results/safety.json",
    grid_path: str | Path = "results/raw/grid.jsonl",
    tasks_path: str | Path = "data/tasks/all.jsonl",
    out_path: str | Path = "results/policy.json",
) -> dict[str, Any]:
    import joblib
    import numpy as np

    from .config import load_config

    cfg = load_config()
    delta = cfg["policy"]["delta"]
    phi = cfg["policy"]["phi"]
    psi = cfg["policy"]["psi"]
    z = cfg["stats"]["z_one_sided"]
    gpu_hourly_rate = cfg["cost"]["gpu_hourly_rate"]

    models = joblib.load(pooled_models_path)

    with open(profiles_path, encoding="utf-8") as f:
        profiles = json.load(f)
    bpp = {rung: rec["bpp"] for rung, rec in profiles["rungs"].items()}
    log_bpp = {rung: float(np.log(bpp[rung])) for rung in LADDER}

    with open(safety_path, encoding="utf-8") as f:
        safety_metrics = json.load(f)
    safety_flip_ucb_train = {
        rung: safety_metrics[f"{rung}|train"]["safety_flip_ucb"] for rung in NON_REFERENCE
    }
    safety_floor = compute_safety_floor(safety_flip_ucb_train, psi)
    safety_classifier_threshold = safety_metrics.get("_classifier", {}).get("threshold")

    cost_fn = cost_fn_factory(grid_path, tasks_path, gpu_hourly_rate)

    n_train_tasks = len({
        json.loads(l)["id"] for l in open(tasks_path, encoding="utf-8")
        if l.strip() and json.loads(l)["split"] == "train"
    })

    cells: dict[str, dict] = {}
    for key, cb in cell_bounds(models, log_bpp, z, cost_fn).items():
        acc_ref = cb["acc_point"][REFERENCE]
        chosen, admissible = choose(
            acc_ref=acc_ref, acc_lcb=cb["acc_lcb"], flip_ucb=cb["flip_ucb"],
            safety_flip_ucb=safety_flip_ucb_train, cost=cb["cost"],
            delta=delta, phi=phi, psi=psi,
        )
        cells[key] = {
            "rung": chosen, "admissible": admissible,
            "low_support": cb["low_support"], "n_support_tasks": cb["n_support_tasks"],
            "acc_lcb": {b: cb["acc_lcb"][b] for b in NON_REFERENCE},
            "acc_ref": acc_ref,
            "flip_ucb": {b: cb["flip_ucb"][b] for b in NON_REFERENCE},
            "cost": cb["cost"],
        }

    result = {
        "operating_point": {"delta": delta, "phi": phi, "psi": psi},
        "safety_floor": safety_floor,
        "safety_classifier_threshold": safety_classifier_threshold,
        "cells": cells,
        "fit_on": "train",
        "n_train_tasks": n_train_tasks,
        "pooled_model_backend": {
            "accuracy": models.get("acc_backend", "unknown"),
            "flip": models.get("flip_backend", "unknown"),
        },
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
