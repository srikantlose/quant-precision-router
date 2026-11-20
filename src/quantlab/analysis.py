"""`quantlab analyze`: every P11 step, in order, from the raw grid to the figures.

Each step writes its own file (plan P11) so it can be inspected or re-run alone:

  labels        results/labels.json                         (labels.py, Sec 9.2)
  embeddings    results/raw/embeddings.npy + ids            (features.py, Sec 9.3)
  predictor     results/predictor_metrics.json, raw/pred_class.json, raw/predictor.joblib
  confusion     category confusion matrix -> predictor_metrics.json
  pooled        results/pooled_models.json, raw/pooled_models.joblib
  safety        results/safety.json, raw/safety_classifier.joblib
  policy        results/policy.json
  evaluate      results/baselines.json                      (baselines + swap simulation)
  sweep         results/sweep.csv
  differential  results/differential.json
  bootstrap     results/bootstrap.json
  figures       results/figures/*.png

Everything is deterministic (seed 42). The refusal audit spends API calls and is a
separate command (`quantlab audit-refusals`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

STEPS = ["labels", "embeddings", "predictor", "confusion", "pooled", "safety", "policy",
         "evaluate", "sweep", "differential", "bootstrap", "figures"]


def _jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def step_embeddings() -> None:
    from .features import compute_embeddings
    pairs = [(t["id"], t["task"]) for t in _jsonl("data/tasks/all.jsonl")]
    pairs += [(t["id"], t["task"]) for t in _jsonl("data/safety/safety.jsonl")]
    compute_embeddings(pairs)


def step_confusion() -> None:
    from .tasktype import CATEGORIES
    tasks = _jsonl("data/tasks/all.jsonl")
    pred = json.loads(Path("results/raw/pred_class.json").read_text(encoding="utf-8"))["category_pred"]
    cats = sorted(CATEGORIES)
    mx = {t: {p: 0 for p in cats} for t in cats}
    for t in tasks:
        mx[t["category"]][pred[t["id"]]] += 1
    n = len(tasks)
    path = Path("results/predictor_metrics.json")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["category_confusion"] = {
        "rows_true_cols_pred": mx,
        "accuracy": sum(mx[c][c] for c in cats) / n,
        "per_category_recall": {c: (mx[c][c] / sum(mx[c].values())) if sum(mx[c].values()) else None for c in cats},
        "n": n,
        "note": "Some seed 'classification' tasks are sentiment prompts and are detected as sentiment; seeds are kept as given.",
    }
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def step_safety() -> None:
    import joblib

    from .config import load_config
    from .safety import compute_safety_metrics, fit_safety_classifier
    cfg = load_config()
    result = compute_safety_metrics("results/raw/grid.jsonl", "data/safety/safety.jsonl", cfg["stats"]["z_one_sided"])
    clf = fit_safety_classifier("data/safety/safety.jsonl", "data/tasks/all.jsonl", "results/raw/embeddings.npy",
                                cfg["safety"]["classifier_target_recall"], cfg["predictor"]["folds"])
    result["_classifier"] = clf["metrics"]
    joblib.dump({"model": clf["model"], "threshold": clf["threshold"]}, "results/raw/safety_classifier.joblib")
    Path("results/safety.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def runners() -> dict[str, Callable[[], object]]:
    from .bootstrap import run_bootstrap
    from .differential import differential_test
    from .evaluate import evaluate, sweep
    from .figures import make_figures
    from .labels import compute_labels
    from .policy import fit_policy
    from .pooled import fit_pooled_models
    from .predictor import fit_predictor
    return {
        "labels": compute_labels, "embeddings": step_embeddings, "predictor": fit_predictor,
        "confusion": step_confusion, "pooled": fit_pooled_models, "safety": step_safety, "policy": fit_policy,
        "evaluate": evaluate, "sweep": sweep, "differential": differential_test, "bootstrap": run_bootstrap,
        "figures": make_figures,
    }


def run_analysis(steps: list[str] | None = None, start: str | None = None, log: Callable[[str], None] = print) -> None:
    chosen = steps or (STEPS[STEPS.index(start):] if start else STEPS)
    unknown = [s for s in chosen if s not in STEPS]
    if unknown:
        raise ValueError(f"unknown analysis step(s): {', '.join(unknown)} (choose from {', '.join(STEPS)})")
    run = runners()
    for s in STEPS:
        if s in chosen:
            t0 = time.time()
            run[s]()
            log(f"  {s:<13} done in {time.time() - t0:6.1f} s")
