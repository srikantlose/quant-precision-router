"""Pooled accuracy and flip models across the whole ladder (plan Sec 9.5).

Rows are clustered by task_id (a GEE with an exchangeable working
correlation, falling back to a GLM with cluster-robust SEs if GEE fails to
converge) because every task contributes one correlated row per rung --
treating them as independent, as an ordinary GLM would, understates the
standard errors and overstates confidence in the selection rule.

The complexity regressor is always the cross-fitted pred_class_oof. This
function refuses to fit on anything else: the observed label or the
heuristic tier would make a rung's own admissibility partly a function of
whether it happened to get a task right, the circularity flagged in review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit

ALLOWED_COMPLEXITY_COL = "pred_class_oof"


def _category_term(categories: list[str]) -> str:
    return f"C(category_pred, levels={list(categories)!r})"


def _rhs_formula(categories: list[str], complexity: str) -> str:
    term = _category_term(categories)
    return f"{term} + {complexity} + log_bpp + {term}:log_bpp + {complexity}:log_bpp"


class PooledModelError(Exception):
    pass


def _finite_and_converged(res) -> bool:
    params = np.asarray(res.params)
    if not np.all(np.isfinite(params)):
        return False
    try:
        cov = np.asarray(res.cov_params())
    except Exception:
        return False
    if not np.all(np.isfinite(cov)):
        return False
    converged = getattr(res, "converged", True)
    return bool(converged)


def _fit_with_fallback(formula: str, df: pd.DataFrame, force_fallback: bool) -> tuple[Any, str]:
    if not force_fallback:
        try:
            model = smf.gee(
                formula, groups="task_id", data=df,
                family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
            )
            res = model.fit(maxiter=200)
            if _finite_and_converged(res):
                return res, "gee"
        except Exception:
            pass

    groups = pd.factorize(df["task_id"])[0]
    model = smf.glm(formula, data=df, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return res, "glm_cluster"


def fit_pooled_model(
    df: pd.DataFrame, response_col: str, complexity_col: str = ALLOWED_COMPLEXITY_COL,
    force_fallback: bool = False,
) -> tuple[Any, str, list[str]]:
    if complexity_col != ALLOWED_COMPLEXITY_COL:
        raise ValueError(
            f"complexity_col must be {ALLOWED_COMPLEXITY_COL!r}; got {complexity_col!r} -- "
            "the pooled models may never regress on the observed label, tier, or "
            "required_precision (that reintroduces circularity, see plan Sec 0.7.2)"
        )

    categories = sorted(df["category_pred"].unique())
    formula = f"{response_col} ~ " + _rhs_formula(categories, complexity_col)
    res, backend = _fit_with_fallback(formula, df, force_fallback)
    return res, backend, categories


def fit_coarse_pooled_model(
    df: pd.DataFrame, response_col: str, force_fallback: bool = False,
) -> tuple[Any, str, list[str]]:
    """Same response, but no pred_class_oof term at all -- used only as a
    fallback for (category, pred_class) cells with too few tasks to support
    the fully-interacted model (Sec 9.5's formula extrapolates wildly past
    the edge of its data in those cells; this model never extrapolates
    along pred_class because it was never fit on it)."""
    categories = sorted(df["category_pred"].unique())
    term = _category_term(categories)
    formula = f"{response_col} ~ {term} + log_bpp + {term}:log_bpp"
    res, backend = _fit_with_fallback(formula, df, force_fallback)
    return res, backend, categories


def predict_with_bounds(
    res, newdata: pd.DataFrame, z: float, categories: list[str],
    complexity_col: str = ALLOWED_COMPLEXITY_COL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuilds the model matrix for newdata from the RHS formula, with
    categorical levels pinned to exactly the categories seen at fit time,
    rather than reusing the fitted model's patsy design_info object --
    patsy's DesignInfo explicitly refuses to pickle, so a design_info
    captured at fit time cannot survive a joblib round trip into another
    process. A freshly parsed formula with the same pinned levels
    reproduces the identical columns deterministically instead."""
    rhs = _rhs_formula(categories, complexity_col)
    return _predict_with_rhs(res, newdata, z, rhs)


def predict_with_bounds_coarse(
    res, newdata: pd.DataFrame, z: float, categories: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    term = _category_term(categories)
    rhs = f"{term} + log_bpp + {term}:log_bpp"
    return _predict_with_rhs(res, newdata, z, rhs)


def _predict_with_rhs(res, newdata: pd.DataFrame, z: float, rhs: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = patsy.dmatrix(rhs, newdata, return_type="dataframe")
    X = X.reindex(columns=res.params.index)
    params = np.asarray(res.params)
    cov = np.asarray(res.cov_params())
    eta = X.values @ params
    se = np.sqrt(np.einsum("ij,jk,ik->i", X.values, cov, X.values))
    point = expit(eta)
    lcb = expit(eta - z * se)
    ucb = expit(eta + z * se)
    return point, lcb, ucb


def _load_bpp(profiles_path: str | Path = "results/profiles.json") -> dict[str, float]:
    with open(profiles_path, encoding="utf-8") as f:
        profiles = json.load(f)
    return {rung: rec["bpp"] for rung, rec in profiles["rungs"].items()}


def _load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_accuracy_df(
    grid_path: str | Path, tasks_path: str | Path, pred_path: str | Path,
    bpp: dict[str, float],
) -> pd.DataFrame:
    """Rows = train bench tasks x 7 rungs, unsolvable tasks included (D26).
    pred_class_oof comes from the predictor's cross-fitted output -- a task
    in the test split has no OOF value and is excluded here (pooled models
    are fit on train only, D19)."""
    tasks = {t["id"]: t for t in _load_jsonl(tasks_path)}
    with open(pred_path, encoding="utf-8") as f:
        pred = json.load(f)
    pred_oof = {k: v for k, v in pred["pred_class_oof_train"].items()}
    cat_pred = pred["category_pred"]

    grid = _load_jsonl(grid_path)
    rows = []
    for r in grid:
        if r["slice"] != "bench":
            continue
        tid = r["task_id"]
        task = tasks.get(tid)
        if task is None or task["split"] != "train":
            continue
        if tid not in pred_oof:
            continue
        rows.append({
            "task_id": tid, "rung": r["rung"],
            "category_pred": cat_pred.get(tid, task["category"]),
            "pred_class_oof": pred_oof[tid],
            "log_bpp": np.log(bpp[r["rung"]]),
            "correct": bool(r["correct"]),
        })
    return pd.DataFrame(rows)


def build_flip_df(
    grid_path: str | Path, tasks_path: str | Path, pred_path: str | Path,
    bpp: dict[str, float], reference: str = "F16",
) -> pd.DataFrame:
    """Rows = train bench tasks x 6 non-reference rungs. flipped = correct(b)
    XOR correct(reference); symmetric, as in the survey's fidelity metric."""
    tasks = {t["id"]: t for t in _load_jsonl(tasks_path)}
    with open(pred_path, encoding="utf-8") as f:
        pred = json.load(f)
    pred_oof = pred["pred_class_oof_train"]
    cat_pred = pred["category_pred"]

    grid = _load_jsonl(grid_path)
    by_task: dict[str, dict[str, dict]] = {}
    for r in grid:
        if r["slice"] != "bench":
            continue
        by_task.setdefault(r["task_id"], {})[r["rung"]] = r

    rows = []
    for tid, rungs in by_task.items():
        task = tasks.get(tid)
        if task is None or task["split"] != "train" or tid not in pred_oof:
            continue
        if reference not in rungs:
            continue
        ref_correct = bool(rungs[reference]["correct"])
        for rung, row in rungs.items():
            if rung == reference:
                continue
            flipped = bool(row["correct"]) != ref_correct
            rows.append({
                "task_id": tid, "rung": rung,
                "category_pred": cat_pred.get(tid, task["category"]),
                "pred_class_oof": pred_oof[tid],
                "log_bpp": np.log(bpp[rung]),
                "flipped": flipped,
            })
    return pd.DataFrame(rows)


def fit_pooled_models(
    grid_path: str | Path = "results/raw/grid.jsonl",
    tasks_path: str | Path = "data/tasks/all.jsonl",
    pred_path: str | Path = "results/raw/pred_class.json",
    profiles_path: str | Path = "results/profiles.json",
    out_path: str | Path = "results/pooled_models.json",
    out_models_path: str | Path = "results/raw/pooled_models.joblib",
) -> dict[str, Any]:
    import joblib

    bpp = _load_bpp(profiles_path)
    acc_df = build_accuracy_df(grid_path, tasks_path, pred_path, bpp)
    flip_df = build_flip_df(grid_path, tasks_path, pred_path, bpp)

    acc_res, acc_backend, acc_categories = fit_pooled_model(acc_df, response_col="correct")
    flip_res, flip_backend, flip_categories = fit_pooled_model(flip_df, response_col="flipped")

    acc_coarse_res, acc_coarse_backend, _ = fit_coarse_pooled_model(acc_df, response_col="correct")
    flip_coarse_res, flip_coarse_backend, _ = fit_coarse_pooled_model(flip_df, response_col="flipped")

    support = (
        acc_df.groupby(["category_pred", "pred_class_oof"])["task_id"]
        .nunique().to_dict()
    )
    support = {f"{cat}|{cls}": int(n) for (cat, cls), n in support.items()}

    joblib.dump({
        "acc_res": acc_res, "flip_res": flip_res,
        "acc_backend": acc_backend, "flip_backend": flip_backend,
        "acc_categories": acc_categories, "flip_categories": flip_categories,
        "acc_coarse_res": acc_coarse_res, "flip_coarse_res": flip_coarse_res,
        "acc_coarse_backend": acc_coarse_backend, "flip_coarse_backend": flip_coarse_backend,
        "cell_support": support,
    }, out_models_path)

    result = {
        "accuracy_model": {
            "backend": acc_backend, "n_rows": len(acc_df),
            "n_tasks": acc_df["task_id"].nunique(),
            "params": {k: float(v) for k, v in zip(acc_res.model.exog_names, acc_res.params)},
            "bse": {k: float(v) for k, v in zip(acc_res.model.exog_names, acc_res.bse)},
        },
        "flip_model": {
            "backend": flip_backend, "n_rows": len(flip_df),
            "n_tasks": flip_df["task_id"].nunique(),
            "params": {k: float(v) for k, v in zip(flip_res.model.exog_names, flip_res.params)},
            "bse": {k: float(v) for k, v in zip(flip_res.model.exog_names, flip_res.bse)},
        },
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return {"summary": result, "acc_res": acc_res, "flip_res": flip_res,
           "acc_backend": acc_backend, "flip_backend": flip_backend}
