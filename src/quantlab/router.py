"""Routing: decide which rung serves a prompt, then serve it (plan P12).

decide() makes the whole decision without generating anything:

  1. pipeline.prepare() picks the category, prompting profile and token budget
     (the same code path the benchmark grid used, so there is no train/serve skew);
  2. the saved predictor estimates the required-precision class from the prompt;
  3. the policy cell (category, class) is re-evaluated with policy.choose() at the
     requested operating point, using the bounds stored in results/policy.json;
  4. the saved safety classifier scores the prompt; a flagged prompt is never
     served below the safety floor.

route() then generates on the decided rung through the VRAM-budgeted ServerPool,
always with the ORIGINAL prompt, and extracts the answer with pipeline.extract().

Two operating points are offered. "strict" is the fitted policy (policy.json). With
about 40 training tasks per cell its bounds are wide, so it serves F16 almost
everywhere. "balanced" (the dashboard's preset of the same name) accepts a 10-point
accuracy allowance, 30% flips and a 10% per-rung safety-flip bound in each cell, which
lets Q3_K_M through Q8_0 serve. In both, the floor for prompts the safety classifier
flags is computed from the server's own psi (policy.json, 0.05 -> Q5_K_M), so no
request can lower it.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from . import pipeline
from .config import load_config
from .features import SURFACE_FEATURE_NAMES, one_hot_category, surface_features
from .policy import apply_floor, choose, compute_safety_floor
from .quants import LADDER
from .tasktype import CATEGORIES

# Named operating points a caller may pick. A preset's psi is only the per-cell safety
# gate; the floor for flagged prompts is always computed from the server's psi.
TOLERANCES = {
    "strict": None,                                        # the fitted policy as-is
    "balanced": {"delta": 0.10, "phi": 0.30, "psi": 0.10},
}


class RouterError(Exception):
    pass


_ZERO_WIDTH_RE = __import__("re").compile("[​‌‍⁠﻿]")
_SENTENCE_RE = __import__("re").compile(r"(?<=[.!?])\s+")


def normalize_routing_input(text: str, max_words: int = 256) -> str:
    """Routing-input normalisation (plan Sec 9.13), a defence measured in P13 and off by
    default: NFKC, drop zero-width and control characters, collapse whitespace, drop
    exact-duplicate sentences (keep the first), keep the first max_words words. Applied
    to the routing input only; the model always receives the original prompt."""
    import unicodedata
    t = unicodedata.normalize("NFKC", text or "")
    t = _ZERO_WIDTH_RE.sub("", t)
    t = "".join(ch for ch in t if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    t = " ".join(t.split())
    seen, kept = set(), []
    for s in _SENTENCE_RE.split(t):
        if s and s not in seen:
            seen.add(s)
            kept.append(s)
    return " ".join(" ".join(kept).split()[:max_words])


@dataclass
class Decision:
    category: str
    label: str
    system: str
    max_tokens: int
    pred_class: int
    predicted_required_precision: str
    cell: str
    cell_rung: str
    admissible: list[str]
    safety_score: float
    flagged: bool
    floor: str
    rung: str
    why: str
    operating_point: dict[str, float] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)   # every stage's inputs, outputs and timing

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("system")
        return d


def resolve_rung(decided: str, floor: str, flagged: bool, force_quant: Optional[str] = None) -> str:
    """The rung actually served. A forced rung replaces the decision, but for a
    flagged prompt neither the decision nor a forced rung may go below the floor."""
    rung = force_quant or decided
    return apply_floor(rung, floor) if flagged else rung


class Router:
    def __init__(self, pool: Any = None, cfg: Optional[dict] = None, embedder: Any = None):
        self.cfg = cfg or load_config()
        results = Path(self.cfg["paths"]["results_dir"])

        policy = json.loads((results / "policy.json").read_text(encoding="utf-8"))
        self.cells: dict[str, dict] = policy["cells"]
        op = policy["operating_point"]
        self.default_op = {"delta": float(op["delta"]), "phi": float(op["phi"]), "psi": float(op["psi"])}

        safety = json.loads((results / "safety.json").read_text(encoding="utf-8"))
        self.safety_ucb = {r: float(safety[f"{r}|train"]["safety_flip_ucb"]) for r in LADDER}
        self.floor = compute_safety_floor(self.safety_ucb, self.default_op["psi"])

        pred = joblib.load(results / "raw" / "predictor.joblib")
        self._pred_model, self._pred_scaler = pred["model"], pred["scaler"]
        clf = joblib.load(results / "raw" / "safety_classifier.joblib")
        self._clf, self.threshold = clf["model"], float(clf["threshold"])

        self.profiles = json.loads((results / "profiles.json").read_text(encoding="utf-8"))
        self.tau = float(json.loads((results / "labels.json").read_text(encoding="utf-8"))["tau"])
        self.gpu_hourly_rate = float(self.cfg["cost"]["gpu_hourly_rate"])
        self._embedder = embedder
        self._pool = pool
        self._lock = threading.Lock()   # one generation at a time: rungs share one GPU

    # ------------------------------------------------------------------ helpers
    def _embed(self, text: str) -> np.ndarray:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            name = self.cfg["predictor"]["embedding_model"]
            try:
                self._embedder = SentenceTransformer(name, device="cpu", local_files_only=True)
            except Exception:   # not cached yet: allow the one-off download
                self._embedder = SentenceTransformer(name, device="cpu")
        vec = self._embedder.encode([text], batch_size=1, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)[0]

    def pool(self):
        if self._pool is None:
            from .server import ServerPool
            rungs = self.profiles["rungs"]
            self._pool = ServerPool(
                budget_mb=float(self.profiles["budget_mb"]),
                vram_mb={r: float(rungs[r]["vram_mb"]) for r in rungs},
                ngl={r: int(rungs[r]["ngl"]) for r in rungs},
            )
        return self._pool

    def operating_point(self, tolerance: str = "strict") -> dict[str, float]:
        if tolerance not in TOLERANCES:
            raise RouterError(f"unknown tolerance {tolerance!r}")
        op = dict(self.default_op)
        if TOLERANCES[tolerance]:
            op.update(TOLERANCES[tolerance])
        return op

    # ------------------------------------------------------------------ decide / route
    def routing_input(self, prompt: str, normalize: Optional[bool] = None) -> str:
        on = self.cfg["routing"]["normalize"] if normalize is None else normalize
        return normalize_routing_input(prompt, int(self.cfg["routing"]["normalize_max_words"])) if on else prompt

    def decide_many(self, prompts: list[str], tolerance: str = "strict", normalize: Optional[bool] = None,
                    floor_on: bool = True) -> list[dict[str, Any]]:
        """Batch decisions without traces (adversarial search, evaluation): one embedding
        pass and one predictor call for all prompts; the same rule as decide()."""
        if not prompts:
            return []
        inputs = [self.routing_input(p, normalize) for p in prompts]
        prep = [pipeline.prepare(t) for t in inputs]
        if self._embedder is None:
            self._embed("warm-up")
        emb = np.asarray(self._embedder.encode(inputs, batch_size=64, normalize_embeddings=True, show_progress_bar=False),
                         dtype=np.float32)
        surf = self._pred_scaler.transform(np.asarray([surface_features(t) for t in inputs], dtype=np.float64))
        cat = np.asarray([one_hot_category(p[0], CATEGORIES) for p in prep])
        classes = self._pred_model.predict(np.hstack([emb, surf, cat]))
        scores = self._clf.predict_proba(emb)[:, 1]
        op = self.operating_point(tolerance)
        cache: dict[str, str] = {}
        out = []
        for (category, label, _s, _m), cls, score in zip(prep, classes, scores):
            key = f"{category}|{int(cls)}"
            if key not in cache:
                c = self.cells[key]
                cache[key] = choose(c["acc_ref"], c["acc_lcb"], c["flip_ucb"], self.safety_ucb, c["cost"],
                                    op["delta"], op["phi"], op["psi"])[0]
            flagged = float(score) >= self.threshold
            rung = resolve_rung(cache[key], self.floor, flagged and floor_on)
            out.append({"category": category, "pred_class": int(cls), "cell": key, "cell_rung": cache[key],
                        "score": float(score), "flagged": flagged, "rung": rung})
        return out

    def decide(self, prompt: str, tolerance: str = "strict", normalize: Optional[bool] = None,
               floor_on: bool = True) -> Decision:
        ms: dict[str, float] = {}
        tick = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal tick
            now = time.perf_counter()
            ms[name] = round((now - tick) * 1000, 2)
            tick = now

        original = prompt
        prompt = self.routing_input(original, normalize)
        category, label, system, max_tokens = pipeline.prepare(prompt)
        lap("classify")
        emb = self._embed(prompt)
        lap("embed")
        surf_raw = surface_features(prompt)
        surf = self._pred_scaler.transform(np.asarray([surf_raw], dtype=np.float64))
        x = np.hstack([emb[None, :], surf, np.asarray([one_hot_category(category, CATEGORIES)])])
        pred_class = int(self._pred_model.predict(x)[0])
        probs = self._pred_model.predict_proba(x)[0]
        lap("predict")

        op = self.operating_point(tolerance)
        key = f"{category}|{pred_class}"
        cell = self.cells.get(key)
        if cell is None:
            raise RouterError(f"no policy cell {key!r}")
        cell_rung, admissible = choose(
            cell["acc_ref"], cell["acc_lcb"], cell["flip_ucb"], self.safety_ucb, cell["cost"],
            op["delta"], op["phi"], op["psi"],
        )
        acc_bar = cell["acc_ref"] - op["delta"]
        gates = []
        for r in LADDER:
            row = {"rung": r, "cost": float(cell["cost"][r]), "admissible": r in admissible}
            if r != "F16":
                row.update({
                    "acc_lcb": float(cell["acc_lcb"][r]), "acc_ok": cell["acc_lcb"][r] >= acc_bar,
                    "flip_ucb": float(cell["flip_ucb"][r]), "flip_ok": cell["flip_ucb"][r] <= op["phi"],
                    "safety_ucb": self.safety_ucb[r], "safety_ok": self.safety_ucb[r] <= op["psi"],
                })
            gates.append(row)
        lap("policy")

        score = float(self._clf.predict_proba(emb[None, :])[0, 1])
        flagged = score >= self.threshold
        rung = resolve_rung(cell_rung, self.floor, flagged and floor_on)
        lap("safety")

        why = f"{key} -> {cell_rung} (cheapest of {len(admissible)} admissible)"
        if flagged and floor_on:
            why += f"; flagged (score {score:.2f}) -> floor {self.floor}" + ("" if rung == cell_rung else f" -> {rung}")
        else:
            why += f"; not flagged (score {score:.2f})"

        trace = {
            "timing_ms": ms,
            "input": {"chars": len(original), "words": len(original.split()), "normalized": prompt != original,
                      "routing_chars": len(prompt), "routing_words": len(prompt.split())},
            "category": {"label": label, "category": category, "max_tokens": max_tokens, "system_prompt": system},
            "features": {"embedding_dims": int(emb.shape[0]), "surface": dict(zip(SURFACE_FEATURE_NAMES, map(float, surf_raw))),
                         "category_onehot": category, "total": int(x.shape[1]), "embedder": self.cfg["predictor"]["embedding_model"]},
            "predictor": {"probs": {LADDER[int(c)]: float(p) for c, p in zip(self._pred_model.classes_, probs)},
                          "pred_class": pred_class},
            "policy": {"cell": key, "n_support_tasks": cell.get("n_support_tasks"), "low_support": bool(cell.get("low_support")),
                       "acc_ref": float(cell["acc_ref"]), "acc_bar": float(acc_bar), "operating_point": op, "tolerance": tolerance,
                       "gates": gates, "admissible": admissible, "chosen": cell_rung},
            "safety": {"score": score, "threshold": self.threshold, "flagged": flagged, "floor": self.floor,
                       "floor_psi": self.default_op["psi"], "raised": rung != cell_rung, "floor_on": floor_on},
        }
        return Decision(
            category=category, label=label, system=system, max_tokens=max_tokens,
            pred_class=pred_class, predicted_required_precision=LADDER[pred_class],
            cell=key, cell_rung=cell_rung, admissible=admissible,
            safety_score=round(score, 4), flagged=flagged, floor=self.floor, rung=rung,
            why=why, operating_point=op, trace=trace,
        )

    def route(self, prompt: str, tolerance: str = "strict", force_quant: Optional[str] = None,
              decision: Optional[Decision] = None) -> dict[str, Any]:
        d = decision or self.decide(prompt, tolerance)
        rung = resolve_rung(d.rung, d.floor, d.flagged, force_quant)
        with self._lock:
            pool = self.pool()
            before = pool.resident() if hasattr(pool, "resident") else []
            client, swapped, swap_s = pool.get(rung)
            after = pool.resident() if hasattr(pool, "resident") else [rung]
            t0 = time.time()
            result = client.generate(prompt, system_prompt=d.system, max_tokens=d.max_tokens)
            latency = time.time() - t0
        answer = pipeline.extract(d.label, result.text)
        out = d.public()
        timings = result.timings or {}
        conf = math.exp(result.mean_logprob) if result.mean_logprob is not None else None
        out["trace"]["timing_ms"].update({"load": round(swap_s * 1000, 1), "generate": round(latency * 1000, 1)})
        out["trace"]["serving"] = {
            "rung": rung, "forced": force_quant, "resident_before": before, "resident_after": after,
            "evicted": [r for r in before if r not in after], "loaded": bool(swapped), "load_s": round(swap_s, 2),
            "budget_mb": self.profiles["budget_mb"],
            "vram_mb": {r: self.profiles["rungs"][r]["vram_mb"] for r in after},
        }
        out["trace"]["generation"] = {
            "tokens_in": result.tokens_in, "tokens_out": result.tokens_out, "finish_reason": result.finish_reason,
            "tokens_per_s": timings.get("predicted_per_second"), "prompt_tokens_per_s": timings.get("prompt_per_second"),
            "mean_logprob": result.mean_logprob, "confidence": conf, "confidence_bar": math.exp(self.tau),
            "temperature": self.cfg["request"]["temperature"], "seed": self.cfg["request"]["seed"], "max_tokens": d.max_tokens,
        }
        out["trace"]["extraction"] = {"raw": (result.text or "")[:4000], "answer": answer,
                                      "changed": (result.text or "").strip() != answer}
        out.update({
            "answer": answer,
            "chosen_quant": rung,
            "forced": bool(force_quant),
            "latency_s": round(latency, 3),
            "tokens_out": result.tokens_out,
            "est_cost_gpu_h": round(latency / 3600 * self.gpu_hourly_rate, 8),
            "swap_occurred": bool(swapped),
            "swap_cost_s": round(swap_s, 2),
            "vram_mb": self.profiles["rungs"][rung]["vram_mb"],
            "classifier_threshold": round(self.threshold, 4),
        })
        return out

    def summary(self) -> dict[str, Any]:
        """Policy overview for GET /policy: the rung each cell serves at each tolerance."""
        out = {"operating_points": {t: self.operating_point(t) for t in TOLERANCES},
               "safety_floor": self.floor, "classifier_threshold": self.threshold, "cells": {}}
        for key, cell in self.cells.items():
            out["cells"][key] = {}
            for t in TOLERANCES:
                op = self.operating_point(t)
                out["cells"][key][t] = choose(cell["acc_ref"], cell["acc_lcb"], cell["flip_ucb"],
                                             self.safety_ucb, cell["cost"], op["delta"], op["phi"], op["psi"])[0]
        return out

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
