"""Required-precision predictor (plan Sec 9.4).

Predicts req(t) in {0..6} from the prompt text alone (embedding + surface
features + one-hot category_pred), via cross-fitted out-of-fold predictions
on train so the pooled models (Sec 9.5) never see a task's own label used to
produce its own regressor value -- that circularity was the single most
reviewed point of this project's design.

Unsolvable train tasks (req is None) are excluded from fitting the
classifier itself, but still pass through the same CV fold split so they
also receive a pred_class_oof (D26: they are real traffic for the pooled
accuracy model, just not for predictor training).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .config import load_config
from .features import load_embeddings, one_hot_category, surface_features
from .tasktype import CATEGORIES
from . import pipeline as qpipeline

SEED = 42


class PredictorError(Exception):
    pass


def _build_row_features(text: str, category_pred: str, embeddings: dict[str, np.ndarray],
                        task_id: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    emb = embeddings[task_id]
    surf = surface_features(text)
    cat = one_hot_category(category_pred, CATEGORIES)
    return emb, np.asarray(surf, dtype=np.float64), cat


def _fit_one_model(X_emb: np.ndarray, X_surf: np.ndarray, X_cat: np.ndarray,
                   y: np.ndarray, c_grid: list[float]) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler().fit(X_surf)
    X_surf_scaled = scaler.transform(X_surf)
    X = np.hstack([X_emb, X_surf_scaled, X_cat])

    n_classes = len(np.unique(y))
    n_per_class = min(np.bincount(y.astype(int))) if n_classes > 1 else len(y)
    inner_splits = min(3, max(2, n_per_class))

    base = LogisticRegression(class_weight="balanced", max_iter=5000)
    if n_classes > 1 and n_per_class >= 2:
        search = GridSearchCV(
            base, {"C": c_grid}, scoring="neg_mean_absolute_error",
            cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=SEED),
        )
        search.fit(X, y)
        model = search.best_estimator_
    else:
        model = base.set_params(C=1.0).fit(X, y)
    return model, scaler


def _predict_class(model: LogisticRegression, scaler: StandardScaler,
                   X_emb: np.ndarray, X_surf: np.ndarray, X_cat: np.ndarray) -> np.ndarray:
    X_surf_scaled = scaler.transform(X_surf)
    X = np.hstack([X_emb, X_surf_scaled, X_cat])
    return model.predict(X)


def fit_predictor(
    labels_path: str | Path = "results/labels.json",
    tasks_path: str | Path = "data/tasks/all.jsonl",
    safety_path: str | Path = "data/safety/safety.jsonl",
    embeddings_path: str | Path = "results/raw/embeddings.npy",
    out_metrics_path: str | Path = "results/predictor_metrics.json",
    out_pred_path: str | Path = "results/raw/pred_class.json",
    out_model_path: str | Path = "results/raw/predictor.joblib",
) -> dict[str, Any]:
    cfg = load_config()
    c_grid = cfg["predictor"]["C_grid"]
    folds = cfg["predictor"]["folds"]

    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    req_by_id: dict[str, int | None] = labels["req"]

    tasks = [json.loads(l) for l in open(tasks_path, encoding="utf-8") if l.strip()]
    tasks_by_id = {t["id"]: t for t in tasks}
    safety_tasks = [json.loads(l) for l in open(safety_path, encoding="utf-8") if l.strip()]

    embeddings = load_embeddings(embeddings_path)

    # category_pred for every bench and safety task, from the classifier only
    def cat_pred_for(text: str) -> str:
        category, *_ = qpipeline.prepare(text)
        return category

    train_ids = [tid for tid in req_by_id if tid in tasks_by_id
                and tasks_by_id[tid]["split"] == "train"]
    test_ids = [tid for tid in req_by_id if tid in tasks_by_id
               and tasks_by_id[tid]["split"] == "test"]

    cat_pred_cache: dict[str, str] = {}
    for tid in train_ids + test_ids:
        cat_pred_cache[tid] = cat_pred_for(tasks_by_id[tid]["task"])

    def features_for(tid: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
        task = tasks_by_id[tid]
        return _build_row_features(task["task"], cat_pred_cache[tid], embeddings, tid)

    train_emb = np.stack([features_for(tid)[0] for tid in train_ids])
    train_surf = np.stack([features_for(tid)[1] for tid in train_ids])
    train_cat = np.asarray([features_for(tid)[2] for tid in train_ids])
    train_req = np.asarray([req_by_id[tid] if req_by_id[tid] is not None else -1
                            for tid in train_ids])

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    pred_oof = np.full(len(train_ids), -1, dtype=int)

    for fold_train_idx, fold_test_idx in skf.split(train_emb, train_req):
        labelled_mask = train_req[fold_train_idx] >= 0
        fit_idx = fold_train_idx[labelled_mask]
        if len(np.unique(train_req[fit_idx])) < 1:
            continue
        model, scaler = _fit_one_model(
            train_emb[fit_idx], train_surf[fit_idx], train_cat[fit_idx],
            train_req[fit_idx], c_grid,
        )
        preds = _predict_class(model, scaler, train_emb[fold_test_idx],
                               train_surf[fold_test_idx], train_cat[fold_test_idx])
        pred_oof[fold_test_idx] = preds

    pred_class_oof = {tid: int(pred_oof[i]) for i, tid in enumerate(train_ids)}

    labelled_mask_all = train_req >= 0
    final_model, final_scaler = _fit_one_model(
        train_emb[labelled_mask_all], train_surf[labelled_mask_all],
        train_cat[labelled_mask_all], train_req[labelled_mask_all], c_grid,
    )

    test_emb = np.stack([features_for(tid)[0] for tid in test_ids]) if test_ids else np.empty((0, 384))
    test_surf = np.stack([features_for(tid)[1] for tid in test_ids]) if test_ids else np.empty((0, 13))
    test_cat = np.asarray([features_for(tid)[2] for tid in test_ids]) if test_ids else np.empty((0, len(CATEGORIES)))
    test_preds = _predict_class(final_model, final_scaler, test_emb, test_surf, test_cat) if test_ids else []
    pred_class_test = {tid: int(p) for tid, p in zip(test_ids, test_preds)}

    safety_ids = [t["id"] for t in safety_tasks]
    for t in safety_tasks:
        cat_pred_cache[t["id"]] = cat_pred_for(t["task"])
    safety_feat = [_build_row_features(t["task"], cat_pred_cache[t["id"]], embeddings, t["id"])
                  for t in safety_tasks]
    safety_emb = np.stack([f[0] for f in safety_feat]) if safety_feat else np.empty((0, 384))
    safety_surf = np.stack([f[1] for f in safety_feat]) if safety_feat else np.empty((0, 13))
    safety_cat = np.asarray([f[2] for f in safety_feat]) if safety_feat else np.empty((0, len(CATEGORIES)))
    safety_preds = _predict_class(final_model, final_scaler, safety_emb, safety_surf, safety_cat) if safety_feat else []
    pred_class_safety = {tid: int(p) for tid, p in zip(safety_ids, safety_preds)}

    joblib.dump({"model": final_model, "scaler": final_scaler}, out_model_path)

    # metrics on labelled test tasks
    test_labelled = [tid for tid in test_ids if req_by_id[tid] is not None]
    y_true = np.asarray([req_by_id[tid] for tid in test_labelled])
    y_pred = np.asarray([pred_class_test[tid] for tid in test_labelled])
    if len(y_true) > 0:
        exact_acc = float(np.mean(y_true == y_pred))
        within_one_acc = float(np.mean(np.abs(y_true - y_pred) <= 1))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

        majority_class = int(np.bincount(train_req[train_req >= 0]).argmax())
        majority_pred = np.full_like(y_true, majority_class)
        majority_mae = float(np.mean(np.abs(y_true - majority_pred)))
        majority_acc = float(np.mean(y_true == majority_pred))

        train_by_cat: dict[str, list[int]] = {}
        for i, tid in enumerate(train_ids):
            if train_req[i] >= 0:
                train_by_cat.setdefault(cat_pred_cache[tid], []).append(int(train_req[i]))
        cat_median = {c: int(np.median(v)) for c, v in train_by_cat.items()}
        overall_median = int(np.median(train_req[train_req >= 0]))
        cat_pred_baseline = np.asarray([
            cat_median.get(cat_pred_cache[tid], overall_median) for tid in test_labelled
        ])
        cat_baseline_mae = float(np.mean(np.abs(y_true - cat_pred_baseline)))
        cat_baseline_acc = float(np.mean(y_true == cat_pred_baseline))
    else:
        exact_acc = within_one_acc = mae = macro_f1 = None
        majority_mae = majority_acc = cat_baseline_mae = cat_baseline_acc = None

    metrics = {
        "n_train_labelled": int(labelled_mask_all.sum()),
        "n_train_unsolvable": int((~labelled_mask_all).sum()),
        "n_test_labelled": len(test_labelled),
        "test_exact_accuracy": exact_acc,
        "test_within_one_accuracy": within_one_acc,
        "test_mae": mae,
        "test_macro_f1": macro_f1,
        "baseline_majority_class_mae": majority_mae,
        "baseline_majority_class_accuracy": majority_acc,
        "baseline_per_category_median_mae": cat_baseline_mae,
        "baseline_per_category_median_accuracy": cat_baseline_acc,
    }

    out_metrics_path = Path(out_metrics_path)
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_out = {
        "pred_class_oof_train": pred_class_oof,
        "pred_class_test": pred_class_test,
        "pred_class_safety": pred_class_safety,
        "category_pred": cat_pred_cache,
    }
    out_pred_path = Path(out_pred_path)
    out_pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_out, f, ensure_ascii=False, indent=2)

    return {"metrics": metrics, "predictions": pred_out}
