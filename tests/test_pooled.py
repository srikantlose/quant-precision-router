"""Tests for the pooled accuracy/flip models (plan Sec 9.5). The circularity
guard is the single point three review rounds focused on: the pooled models'
complexity regressor must always be the cross-fitted pred_class_oof, never
the label itself or the heuristic tier."""

import numpy as np
import pandas as pd
import pytest

from quantlab import pooled
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


def test_glm_fallback_path_produces_usable_bounds():
    df = _synthetic_df()
    res, backend, categories = fit_pooled_model(df, response_col="correct", complexity_col="pred_class_oof",
                                                force_fallback=True)
    assert backend == "glm_cluster"
    newdata = pd.DataFrame([{"category_pred": "math", "pred_class_oof": 1, "log_bpp": np.log(1.0)}])
    point, lcb, ucb = predict_with_bounds(res, newdata, z=1.645, categories=categories)
    assert lcb[0] <= point[0] <= ucb[0]


def test_boolean_response_is_not_inverted():
    # patsy dummy-codes a bool response and the binomial family then models
    # P(False); every pooled prediction once came out as its complement.
    df = _synthetic_df()
    assert df["correct"].dtype == bool
    res, _, categories = fit_pooled_model(df, response_col="correct", complexity_col="pred_class_oof")
    lo = pd.DataFrame([{"category_pred": "math", "pred_class_oof": 3, "log_bpp": np.log(0.3)}])
    hi = pd.DataFrame([{"category_pred": "math", "pred_class_oof": 3, "log_bpp": np.log(16.0)}])
    p_lo = predict_with_bounds(res, lo, z=1.645, categories=categories)[0][0]
    p_hi = predict_with_bounds(res, hi, z=1.645, categories=categories)[0][0]
    assert p_hi > p_lo  # the synthetic truth: more bits -> more likely correct
    fitted = np.asarray(res.predict(df.assign(correct=df["correct"].astype(float))))
    assert abs(fitted.mean() - df["correct"].mean()) < 0.03


class _DivergedGEEResult:
    """Finite parameters and covariance, but predictions far from the data --
    the shape a mis-specified fit takes (the real case predicted 0.96 at
    Q8_0 against an observed 0.045 flip rate)."""

    def __init__(self, n_params):
        self.params = pd.Series(np.zeros(n_params))
        self.bse = pd.Series(np.ones(n_params))

    def cov_params(self):
        return np.eye(len(self.params))

    def predict(self, df):
        return np.full(len(df), 0.99)


def test_miscalibrated_gee_fit_falls_back_to_glm(monkeypatch):
    df = _synthetic_df()

    class _FakeGEE:
        def __init__(self, *a, **k):
            pass

        def fit(self, **k):
            return _DivergedGEEResult(5)

    monkeypatch.setattr(pooled.smf, "gee", _FakeGEE)
    res, backend, _ = fit_pooled_model(df, response_col="correct", complexity_col="pred_class_oof")
    assert backend == "glm_cluster"
    fitted = np.asarray(res.predict(df))
    assert abs(fitted.mean() - df["correct"].mean()) < 0.05
