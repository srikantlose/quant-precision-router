"""Tests for the pooled accuracy/flip models (plan Sec 9.5). The circularity
guard is the single point three review rounds focused on: the pooled models'
complexity regressor must always be the cross-fitted pred_class_oof, never
the label itself or the heuristic tier."""

import numpy as np
import pandas as pd
import pytest

from quantlab.pooled import fit_pooled_model, predict_with_bounds


def _synthetic_df(n_tasks=60, seed=42):
    rng = np.random.default_rng(seed)
    categories = ["math", "code_gen"]
    rows = []
    for i in range(n_tasks):
        cat = categories[i % 2]
        pred_class = rng.integers(0, 7)
        for bpp in [0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
            log_bpp = np.log(bpp)
            logit = -2.0 + 1.5 * log_bpp + 0.3 * pred_class
            p = 1 / (1 + np.exp(-logit))
            correct = rng.random() < p
            rows.append({
                "task_id": f"t{i}", "category_pred": cat, "pred_class_oof": pred_class,
                "log_bpp": log_bpp, "correct": correct,
            })
    return pd.DataFrame(rows)


def test_circularity_guard_rejects_label_columns():
    df = _synthetic_df()
    for bad_col in ["req", "tier", "required_precision"]:
        df_bad = df.rename(columns={"pred_class_oof": bad_col})
        with pytest.raises(ValueError):
            fit_pooled_model(df_bad, response_col="correct", complexity_col=bad_col)


def test_accuracy_model_bounds_ordering():
    df = _synthetic_df()
    res, backend, categories = fit_pooled_model(df, response_col="correct", complexity_col="pred_class_oof")
    newdata = pd.DataFrame([
        {"category_pred": "math", "pred_class_oof": 3, "log_bpp": np.log(2.0)},
        {"category_pred": "code_gen", "pred_class_oof": 5, "log_bpp": np.log(8.0)},
    ])
    point, lcb, ucb = predict_with_bounds(res, newdata, z=1.645, categories=categories)
    assert all(lcb <= point) and all(point <= ucb)
    assert backend in ("gee", "glm_cluster")


def test_glm_fallback_runs_on_degenerate_data():
    # Perfectly separable-ish / tiny data can make GEE convergence flaky;
    # the fallback path must still produce a usable result.
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        for bpp in [0.3, 1.0, 8.0]:
            rows.append({
                "task_id": f"t{i}", "category_pred": "math", "pred_class_oof": i % 3,
                "log_bpp": np.log(bpp), "correct": bool(rng.integers(0, 2)),
            })
    df = pd.DataFrame(rows)
    res, backend, categories = fit_pooled_model(df, response_col="correct", complexity_col="pred_class_oof",
                                                force_fallback=True)
    assert backend == "glm_cluster"
    newdata = pd.DataFrame([{"category_pred": "math", "pred_class_oof": 1, "log_bpp": np.log(1.0)}])
    point, lcb, ucb = predict_with_bounds(res, newdata, z=1.645, categories=categories)
    assert lcb[0] <= point[0] <= ucb[0]
