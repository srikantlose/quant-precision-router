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

Deviation from Sec 9.5: the accuracy model gives each rung its own level
(C(rung)) instead of a straight line in log_bpp. Measured accuracy is a
knee -- Q2_K collapses, Q4_K_M and above are flat -- and the straight line
over-predicted F16 (0.79 fitted vs 0.74 observed) while under-predicting
Q5/Q6, so the accuracy condition rejected every cheaper rung. The flip
model keeps log_bpp: flip rate falls smoothly with bits, the line fits it
within two points at every rung, and per-rung levels would hit perfect
separation (several category x rung cells have zero flips).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .quants import LADDER

ALLOWED_COMPLEXITY_COL = "pred_class_oof"


def _category_term(levels: dict) -> str:
    return f"C(category_pred, levels={list(levels['categories'])!r})"


def _bits_term(levels: dict) -> str:
    rungs = levels.get("rungs")
    return f"C(rung, levels={list(rungs)!r})" if rungs else "log_bpp"


def _rhs_formula(levels: dict, complexity: str) -> str:
    cat, bits = _category_term(levels), _bits_term(levels)
    return f"{cat} + {complexity} + {bits} + {cat}:{bits} + {complexity}:{bits}"


def _coarse_rhs(levels: dict) -> str:
    cat, bits = _category_term(levels), _bits_term(levels)
    return f"{cat} + {bits} + {cat}:{bits}"


def _levels(df: pd.DataFrame, by_rung: bool) -> dict:
    """The exact factor levels seen at fit time; persisted alongside the model
    so prediction can rebuild an identical design matrix."""
    present = set(df["rung"]) if by_rung else set()
    return {
        "categories": sorted(df["category_pred"].unique()),
        "rungs": [r for r in LADDER if r in present] if by_rung else None,
    }


class PooledModelError(Exception):
    pass


CALIBRATION_TOL = 0.10


def _finite(res) -> bool:
    if not np.all(np.isfinite(np.asarray(res.params))):
        return False
    try:
        cov = np.asarray(res.cov_params())
    except Exception:
        return False
    return bool(np.all(np.isfinite(cov)))


def _numeric_response(df: pd.DataFrame, response_col: str) -> pd.DataFrame:
    """patsy dummy-codes a boolean response into [False, True] columns and the
    binomial family then models the first one, silently fitting P(False) --
    every accuracy and flip prediction came out as its complement until this
    cast was added."""
    return df.assign(**{response_col: df[response_col].astype(float)})


def _calibrated(res, df: pd.DataFrame, response_col: str) -> bool:
    """The fitted rate must track the observed rate at every rung. A fit can
    return finite parameters while modelling the wrong thing, and statsmodels'
    GEE results carry no convergence flag to catch it."""
    fitted = np.asarray(res.predict(df), dtype=float)
    observed = df[response_col].astype(float).to_numpy()
    by_rung = pd.DataFrame({
        "rung": df["log_bpp"].round(6).to_numpy(), "fitted": fitted, "observed": observed,
    }).groupby("rung").agg(fitted=("fitted", "mean"), observed=("observed", "mean"), n=("observed", "size"))
    noise = 3 * np.sqrt(by_rung["observed"] * (1 - by_rung["observed"]) / by_rung["n"])
    return bool(np.all(np.abs(by_rung["fitted"] - by_rung["observed"]) <= CALIBRATION_TOL + noise))


def _fit_with_fallback(
    formula: str, df: pd.DataFrame, response_col: str, force_fallback: bool,
) -> tuple[Any, str]:
    if not force_fallback:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.gee(
                    formula, groups="task_id", data=df,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
                )
                res = model.fit(maxiter=200)
            diverged = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            if not diverged and _finite(res) and _calibrated(res, df, response_col):
                return res, "gee"
        except Exception:
            pass

    groups = pd.factorize(df["task_id"])[0]
    model = smf.glm(formula, data=df, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    if not (_finite(res) and _calibrated(res, df, response_col)):
        raise PooledModelError(
            f"neither GEE nor the GLM-cluster fallback gave a calibrated fit for {response_col!r}"
        )
    return res, "glm_cluster"


def fit_pooled_model(
    df: pd.DataFrame, response_col: str, complexity_col: str = ALLOWED_COMPLEXITY_COL,
    force_fallback: bool = False, by_rung: bool = False,
) -> tuple[Any, str, dict]:
    if complexity_col != ALLOWED_COMPLEXITY_COL:
        raise ValueError(
            f"complexity_col must be {ALLOWED_COMPLEXITY_COL!r}; got {complexity_col!r} -- "
            "the pooled models may never regress on the observed label, tier, or "
            "required_precision (that reintroduces circularity, see plan Sec 0.7.2)"
        )

    df = _numeric_response(df, response_col)
    levels = _levels(df, by_rung)
    formula = f"{response_col} ~ " + _rhs_formula(levels, complexity_col)
    res, backend = _fit_with_fallback(formula, df, response_col, force_fallback)
    return res, backend, levels


def fit_coarse_pooled_model(
    df: pd.DataFrame, response_col: str, force_fallback: bool = False, by_rung: bool = False,
) -> tuple[Any, str, dict]:
    """Same response, but no pred_class_oof term at all -- used only as a
    fallback for (category, pred_class) cells with too few tasks to support
    the fully-interacted model, so it never extrapolates along pred_class
    into cells it has no data for."""
    df = _numeric_response(df, response_col)
    levels = _levels(df, by_rung)
    formula = f"{response_col} ~ " + _coarse_rhs(levels)
    res, backend = _fit_with_fallback(formula, df, response_col, force_fallback)
    return res, backend, levels


def predict_with_bounds(
    res, newdata: pd.DataFrame, z: float, levels: dict,
    complexity_col: str = ALLOWED_COMPLEXITY_COL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuilds the model matrix for newdata from the RHS formula, with
    factor levels pinned to exactly those seen at fit time, rather than
    reusing the fitted model's patsy design_info object -- patsy's
    DesignInfo explicitly refuses to pickle, so a design_info captured at
    fit time cannot survive a joblib round trip into another process. A
    freshly parsed formula with the same pinned levels reproduces the
    identical columns deterministically instead."""
    return _predict_with_rhs(res, newdata, z, _rhs_formula(levels, complexity_col))


def predict_with_bounds_coarse(
    res, newdata: pd.DataFrame, z: float, levels: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _predict_with_rhs(res, newdata, z, _coarse_rhs(levels))


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

    acc_res, acc_backend, acc_levels = fit_pooled_model(acc_df, response_col="correct", by_rung=True)
    flip_res, flip_backend, flip_levels = fit_pooled_model(flip_df, response_col="flipped")

    acc_coarse_res, acc_coarse_backend, _ = fit_coarse_pooled_model(acc_df, response_col="correct", by_rung=True)
    flip_coarse_res, flip_coarse_backend, _ = fit_coarse_pooled_model(flip_df, response_col="flipped")

    support = (
        acc_df.groupby(["category_pred", "pred_class_oof"])["task_id"]
        .nunique().to_dict()
    )
    support = {f"{cat}|{cls}": int(n) for (cat, cls), n in support.items()}

    joblib.dump({
        "acc_res": acc_res, "flip_res": flip_res,
        "acc_backend": acc_backend, "flip_backend": flip_backend,
        "acc_levels": acc_levels, "flip_levels": flip_levels,
        "acc_coarse_res": acc_coarse_res, "flip_coarse_res": flip_coarse_res,
        "acc_coarse_backend": acc_coarse_backend, "flip_coarse_backend": flip_coarse_backend,
        "cell_support": support,
    }, out_models_path)

    result = {
        "accuracy_model": {
            "backend": acc_backend, "bits_term": "C(rung)", "n_rows": len(acc_df),
            "n_tasks": acc_df["task_id"].nunique(),
            "params": {k: float(v) for k, v in zip(acc_res.model.exog_names, acc_res.params)},
            "bse": {k: float(v) for k, v in zip(acc_res.model.exog_names, acc_res.bse)},
        },
        "flip_model": {
            "backend": flip_backend, "bits_term": "log_bpp", "n_rows": len(flip_df),
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
