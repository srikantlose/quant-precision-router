"""Bootstrap (plan Sec 9.10).

1. Metric intervals: B = 1,000 resamples of test task ids; every policy's accuracy,
   flip rate and inference cost are recomputed on each resample, giving 95% percentile
   intervals, plus paired intervals for each policy minus the learned policy (paired
   resampling is the fair test of "beats"; overlapping marginal intervals count as
   "not shown to beat" in the report).
2. Refit stability: B = 200 resamples of train task ids (a resampled task brings all
   its rows; duplicates become separate clusters). Each refit repeats the predictor's
   cross-fitting (grouped by original task, so duplicates never straddle folds), the
   pooled accuracy and flip models and the policy, then reports per-cell agreement with
   the full-data policy and the spread of test accuracy and cost. To keep 200 refits
   tractable the predictor's regularisation C is held at the full-data choice instead
   of re-running its inner grid search; everything else is refit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .evaluate import EvalData
from .quants import LADDER

NON_REF = LADDER[:-1]


def _pct(a: np.ndarray) -> dict[str, float]:
    return {"lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5))}


def metric_intervals(D: EvalData, B: int = 1000, seed: int = 42) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    bench = D.ids("bench", "test")
    idx = rng.integers(0, len(bench), size=(B, len(bench)))
    ref = np.array([D.grid[i]["F16"]["correct"] for i in bench], dtype=float)
    f16_lat = np.array([D.grid[i]["F16"]["latency"] for i in bench])
    for name, op in D.ops.items():
        pols = D.policies(op, "test")
        arrs = {}
        for p, assign in pols.items():
            c = np.array([D.grid[i][assign[i]]["correct"] for i in bench], dtype=float)
            lat = np.array([D.grid[i][assign[i]]["latency"] for i in bench])
            arrs[p] = (c, lat)
        res = {}
        lc, ll = arrs["learned"]
        for p, (c, lat) in arrs.items():
            acc_b = c[idx].mean(axis=1)
            flip_b = (c[idx] != ref[idx]).mean(axis=1)
            cost_b = lat[idx].mean(axis=1) / f16_lat[idx].mean(axis=1)
            d_acc = acc_b - lc[idx].mean(axis=1)
            d_cost = cost_b - ll[idx].mean(axis=1) / f16_lat[idx].mean(axis=1)
            res[p] = {"accuracy": float(c.mean()), "accuracy_ci": _pct(acc_b),
                      "flip_rate": float((c != ref).mean()), "flip_rate_ci": _pct(flip_b),
                      "cost_vs_f16": float(lat.mean() / f16_lat.mean()), "cost_vs_f16_ci": _pct(cost_b),
                      "minus_learned_accuracy_ci": _pct(d_acc), "minus_learned_cost_ci": _pct(d_cost)}
        out[name] = res
    return {"B": B, "n_test_bench": len(bench), "by_operating_point": out,
            "cost_note": "cost_vs_f16 is inference time only (recorded answer latency); swap costs are in baselines.json."}


# ------------------------------------------------------------------------- refits
class _RefitData:
    def __init__(self, D: EvalData):
        import joblib

        from .features import one_hot_category, surface_features
        from .policy import cost_fn_factory
        from .tasktype import CATEGORIES

        self.D = D
        R = Path("results")
        tasks = {json.loads(l)["id"]: json.loads(l) for l in open("data/tasks/all.jsonl", encoding="utf-8") if l.strip()}
        ids = json.loads((R / "raw" / "embeddings.ids.json").read_text(encoding="utf-8"))
        emb = np.load(R / "raw" / "embeddings.npy")
        row = {t: k for k, t in enumerate(ids)}
        pred = joblib.load(R / "raw" / "predictor.joblib")
        self.C = float(pred["model"].C)
        self.train = D.ids("bench", "train")
        self.test = D.ids("bench", "test")
        catp = {t: D.items[t].catp for t in self.train + self.test}

        def feats(tids):
            e = np.stack([emb[row[t]] for t in tids])
            s = np.asarray([surface_features(tasks[t]["task"]) for t in tids], dtype=np.float64)
            c = np.asarray([one_hot_category(catp[t], CATEGORIES) for t in tids], dtype=np.float64)
            return e, s, c

        self.tr_emb, self.tr_surf, self.tr_cat = feats(self.train)
        self.te_emb, self.te_surf, self.te_cat = feats(self.test)
        req = D.req
        self.tr_req = np.asarray([req[t] if req.get(t) is not None else -1 for t in self.train])
        self.correct = {t: {r: bool(D.grid[t][r]["correct"]) for r in LADDER} for t in self.train}
        prof = json.loads((R / "profiles.json").read_text(encoding="utf-8"))
        self.log_bpp = {r: float(np.log(prof["rungs"][r]["bpp"])) for r in LADDER}
        self.cost_fn = cost_fn_factory("results/raw/grid.jsonl", "data/tasks/all.jsonl", D.rate)
        self.catp = catp


def _lr(C: float):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(class_weight="balanced", max_iter=5000, C=C)


def _fit_predict(RD: _RefitData, fit_idx, X_emb, X_surf, X_cat, y, P_emb, P_surf, P_cat):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_surf[fit_idx])
    m = _lr(RD.C).fit(np.hstack([X_emb[fit_idx], sc.transform(X_surf[fit_idx]), X_cat[fit_idx]]), y[fit_idx])
    return m.predict(np.hstack([P_emb, sc.transform(P_surf), P_cat]))


def refit_once(RD: _RefitData, b: int, full_maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    from .policy import cell_bounds, choose
    from .pooled import fit_coarse_pooled_model, fit_pooled_model

    D = RD.D
    rng = np.random.default_rng([42, b])
    n = len(RD.train)
    pick = rng.integers(0, n, size=n)
    emb, surf, cat, y = RD.tr_emb[pick], RD.tr_surf[pick], RD.tr_cat[pick], RD.tr_req[pick]
    groups = pick
    oof = np.full(n, -1)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, te_idx in skf.split(emb, y, groups):
        fit_idx = tr_idx[y[tr_idx] >= 0]
        oof[te_idx] = _fit_predict(RD, fit_idx, emb, surf, cat, y, emb[te_idx], surf[te_idx], cat[te_idx])
    test_cls = _fit_predict(RD, np.where(y >= 0)[0], emb, surf, cat, y, RD.te_emb, RD.te_surf, RD.te_cat)

    acc_rows, flip_rows = [], []
    for k, j in enumerate(pick):
        tid = RD.train[j]
        cp = RD.catp[tid]
        for r in LADDER:
            acc_rows.append({"task_id": f"{tid}#{k}", "rung": r, "category_pred": cp, "pred_class_oof": int(oof[k]),
                             "log_bpp": RD.log_bpp[r], "correct": RD.correct[tid][r]})
            if r != "F16":
                flip_rows.append({"task_id": f"{tid}#{k}", "rung": r, "category_pred": cp, "pred_class_oof": int(oof[k]),
                                  "log_bpp": RD.log_bpp[r], "flipped": RD.correct[tid][r] != RD.correct[tid]["F16"]})
    acc_df, flip_df = pd.DataFrame(acc_rows), pd.DataFrame(flip_rows)
    acc_res, acc_backend, acc_levels = fit_pooled_model(acc_df, response_col="correct", by_rung=True)
    flip_res, flip_backend, flip_levels = fit_pooled_model(flip_df, response_col="flipped")
    acc_c, _, _ = fit_coarse_pooled_model(acc_df, response_col="correct", by_rung=True)
    flip_c, _, _ = fit_coarse_pooled_model(flip_df, response_col="flipped")
    support = acc_df.groupby(["category_pred", "pred_class_oof"])["task_id"].nunique().to_dict()
    models = {"acc_res": acc_res, "flip_res": flip_res, "acc_levels": acc_levels, "flip_levels": flip_levels,
              "acc_coarse_res": acc_c, "flip_coarse_res": flip_c,
              "cell_support": {f"{c}|{k}": int(v) for (c, k), v in support.items()}}
    cb = cell_bounds(models, RD.log_bpp, D.z, RD.cost_fn)

    out = {"b": b, "backends": [acc_backend, flip_backend]}
    f16_lat = np.mean([D.grid[t]["F16"]["latency"] for t in RD.test])
    for name, op in D.ops.items():
        cmap = {k: choose(c["acc_point"]["F16"], c["acc_lcb"], c["flip_ucb"], D.safety_ucb, c["cost"],
                          op["delta"], op["phi"], op["psi"])[0] for k, c in cb.items()}
        agree = float(np.mean([cmap[k] == full_maps[name][k] for k in cmap]))
        served = []
        for t, cls in zip(RD.test, test_cls):
            rung = cmap[f"{RD.catp[t]}|{int(cls)}"]
            if D.flagged(t) and LADDER.index(rung) < LADDER.index(D.floor):
                rung = D.floor
            served.append(rung)
        acc = float(np.mean([D.grid[t][r]["correct"] for t, r in zip(RD.test, served)]))
        cost = float(np.mean([D.grid[t][r]["latency"] for t, r in zip(RD.test, served)]) / f16_lat)
        out[name] = {"cell_agreement": agree, "cells_below_f16": sum(1 for r in cmap.values() if r != "F16"),
                     "test_accuracy": acc, "test_cost_vs_f16": cost}
    return out


def refit_stability(B: int = 200, n_jobs: int = 4) -> dict[str, Any]:
    from joblib import Parallel, delayed

    D = EvalData()
    RD = _RefitData(D)
    full = {name: {k: r for k, (r, _) in D.cell_map(op).items()} for name, op in D.ops.items()}
    t0 = time.time()
    reps = Parallel(n_jobs=n_jobs)(delayed(refit_once)(RD, b, full) for b in range(B))
    summary = {}
    for name in D.ops:
        a = np.array([r[name]["cell_agreement"] for r in reps])
        acc = np.array([r[name]["test_accuracy"] for r in reps])
        cost = np.array([r[name]["test_cost_vs_f16"] for r in reps])
        below = np.array([r[name]["cells_below_f16"] for r in reps])
        summary[name] = {"cell_agreement_mean": float(a.mean()), "cell_agreement_ci": _pct(a),
                         "test_accuracy_mean": float(acc.mean()), "test_accuracy_ci": _pct(acc),
                         "test_cost_vs_f16_mean": float(cost.mean()), "test_cost_vs_f16_ci": _pct(cost),
                         "cells_below_f16_mean": float(below.mean()), "cells_below_f16_ci": _pct(below)}
    return {"B": B, "seconds": round(time.time() - t0, 1), "predictor_C_fixed": RD.C, "summary": summary,
            "fallback_backends": sum(1 for r in reps if "glm_cluster" in r["backends"]), "replicates": reps}


def run_bootstrap(B_metric: int = 1000, B_refit: int = 200, n_jobs: int = 4,
                  out_path: str | Path = "results/bootstrap.json") -> dict[str, Any]:
    D = EvalData()
    result = {"metric_intervals": metric_intervals(D, B_metric)}
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")   # save the fast part first
    result["refit_stability"] = refit_stability(B_refit, n_jobs)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
