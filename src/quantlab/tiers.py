"""Heuristic complexity tier scorer (easy/moderate/complex).

This is used ONLY for stratifying the task set and for reporting -- it is
never the routing key (see docs/literature-survey.md: the predicted
required-precision class drives routing; the tier exists so results can be
broken down by intended difficulty without that breakdown leaking into the
policy itself).
"""

from __future__ import annotations

import random
import re
from typing import Any

TIERS = ["easy", "moderate", "complex"]

_WORD_RE = re.compile(r"\S+")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_CONJUNCTION_RE = re.compile(r"\b(and|then|after|before|if|but|while|unless)\b",
                            re.IGNORECASE)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_ASSERT_RE = re.compile(r"\bassert\b")

FEATURE_NAMES = ["n_words", "n_numbers", "n_conjunctions", "n_code_lines",
                 "n_asserts", "n_expected_items"]


def _n_words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _n_numbers(text: str) -> int:
    return len(_NUMBER_RE.findall(text or ""))


def _n_conjunctions(text: str) -> int:
    return len(_CONJUNCTION_RE.findall(text or ""))


def _n_code_lines(text: str) -> int:
    lines = sum(block.count("\n") for block in _FENCE_RE.findall(text or ""))
    for line in (text or "").splitlines():
        if line.startswith("    ") and line.strip():
            lines += 1
    return lines


def _n_asserts(test_code: str | None) -> int:
    return len(_ASSERT_RE.findall(test_code or ""))


def _n_expected_items(expected: Any) -> int:
    if isinstance(expected, list):
        return len(expected)
    if isinstance(expected, dict):
        any_of = expected.get("any_of")
        if isinstance(any_of, list):
            return len(any_of)
        return 1
    return 1


def features_for(task: dict) -> dict[str, float]:
    task_text = task.get("task", "") or ""
    test_code = task.get("test_code") or ""
    return {
        "n_words": _n_words(task_text),
        "n_numbers": _n_numbers(task_text),
        "n_conjunctions": _n_conjunctions(task_text),
        "n_code_lines": _n_code_lines(task_text) + _n_code_lines(test_code),
        "n_asserts": _n_asserts(test_code),
        "n_expected_items": _n_expected_items(task.get("expected")),
    }


def _zscore(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var ** 0.5
    if std == 0:
        return [0.0] * n  # constant feature contributes nothing, not a div/0
    return [(v - mean) / std for v in values]


def compute_scores(tasks: list[dict]) -> dict[str, float]:
    """{task_id: score}, each feature z-scored WITHIN its own category so
    e.g. "number of words" is judged relative to other tasks of the same
    kind, not across categories with very different natural lengths."""
    by_category: dict[str, list[dict]] = {}
    for t in tasks:
        by_category.setdefault(t["category"], []).append(t)

    scores: dict[str, float] = {}
    for cat_tasks in by_category.values():
        feats = [features_for(t) for t in cat_tasks]
        zs = {name: _zscore([f[name] for f in feats]) for name in FEATURE_NAMES}
        for i, t in enumerate(cat_tasks):
            scores[t["id"]] = sum(zs[name][i] for name in FEATURE_NAMES)
    return scores


def tier_of(score: float, t1: float, t2: float) -> str:
    if score < t1:
        return "easy"
    if score < t2:
        return "moderate"
    return "complex"


def fit_thresholds(scores: list[float], requested_tiers: list[str]) -> tuple[float, float]:
    """Choose t1 <= t2 from the observed score values that maximise exact
    agreement between tier_of(score, t1, t2) and the requested tier."""
    candidates = sorted(set(scores))
    if not candidates:
        return (0.0, 0.0)
    best = (candidates[0], candidates[0])
    best_agree = -1
    for i, t1 in enumerate(candidates):
        for t2 in candidates[i:]:
            agree = sum(1 for s, req in zip(scores, requested_tiers)
                       if tier_of(s, t1, t2) == req)
            if agree > best_agree:
                best_agree = agree
                best = (t1, t2)
    return best


def _tertile_thresholds(scores: list[float]) -> tuple[float, float]:
    s = sorted(scores)
    n = len(s)
    if n == 0:
        return (0.0, 0.0)
    return (s[n // 3], s[(2 * n) // 3])


def assign_heuristic_tiers(tasks: list[dict]) -> dict[str, Any]:
    """{task_id: tier_heuristic} for every task, fitted per category on its
    own generated (requested-tier) tasks where available, falling back to
    within-category score tertiles over all of that category's tasks when
    a category has no generated tasks yet (a budget shortfall)."""
    scores = compute_scores(tasks)
    by_category: dict[str, list[dict]] = {}
    for t in tasks:
        by_category.setdefault(t["category"], []).append(t)

    thresholds: dict[str, tuple[float, float]] = {}
    tier_heuristic: dict[str, str] = {}
    fit_info: dict[str, Any] = {}

    for category, cat_tasks in by_category.items():
        generated = [t for t in cat_tasks if t.get("source") == "fireworks" and t.get("tier")]
        if generated:
            t1, t2 = fit_thresholds([scores[t["id"]] for t in generated],
                                    [t["tier"] for t in generated])
            fit_info[category] = {"method": "fitted", "n_generated": len(generated)}
        else:
            t1, t2 = _tertile_thresholds([scores[t["id"]] for t in cat_tasks])
            fit_info[category] = {"method": "tertile_fallback", "n_generated": 0}
        thresholds[category] = (t1, t2)
        for t in cat_tasks:
            tier_heuristic[t["id"]] = tier_of(scores[t["id"]], t1, t2)

    return {"tier_heuristic": tier_heuristic,
           "thresholds": {k: list(v) for k, v in thresholds.items()},
           "fit_info": fit_info}


def cross_validated_kappa(tasks: list[dict], folds: int = 5, seed: int = 42) -> dict[str, Any]:
    """5-fold CV within each category: fit thresholds on 4/5 of the
    generated (requested-tier) tasks, predict the held-out 1/5, pool
    out-of-fold predictions across all categories and report the
    quadratic-weighted Cohen's kappa against the requested tier."""
    from sklearn.metrics import cohen_kappa_score

    scores = compute_scores(tasks)
    by_category: dict[str, list[dict]] = {}
    for t in tasks:
        if t.get("source") == "fireworks" and t.get("tier"):
            by_category.setdefault(t["category"], []).append(t)

    rng = random.Random(seed)
    tier_rank = {"easy": 0, "moderate": 1, "complex": 2}
    y_true: list[int] = []
    y_pred: list[int] = []

    for cat_tasks in by_category.values():
        n = len(cat_tasks)
        order = list(range(n))
        rng.shuffle(order)
        fold_of = {order[pos]: pos % folds for pos in range(n)}
        cat_scores = [scores[t["id"]] for t in cat_tasks]
        cat_tiers = [t["tier"] for t in cat_tasks]

        for k in range(folds):
            train_i = [i for i in range(n) if fold_of[i] != k]
            test_i = [i for i in range(n) if fold_of[i] == k]
            if not train_i or not test_i:
                continue
            t1, t2 = fit_thresholds([cat_scores[i] for i in train_i],
                                    [cat_tiers[i] for i in train_i])
            for i in test_i:
                pred_tier = tier_of(cat_scores[i], t1, t2)
                y_true.append(tier_rank[cat_tiers[i]])
                y_pred.append(tier_rank[pred_tier])

    if not y_true:
        return {"kappa": None, "n": 0}
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return {"kappa": float(kappa), "n": len(y_true)}
