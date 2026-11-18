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
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from . import pipeline
from .config import load_config
from .features import one_hot_category, surface_features
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
    def decide(self, prompt: str, tolerance: str = "strict") -> Decision:
        category, label, system, max_tokens = pipeline.prepare(prompt)
        emb = self._embed(prompt)
        surf = self._pred_scaler.transform(np.asarray([surface_features(prompt)], dtype=np.float64))
        x = np.hstack([emb[None, :], surf, np.asarray([one_hot_category(category, CATEGORIES)])])
        pred_class = int(self._pred_model.predict(x)[0])

        op = self.operating_point(tolerance)
        key = f"{category}|{pred_class}"
        cell = self.cells.get(key)
        if cell is None:
            raise RouterError(f"no policy cell {key!r}")
        cell_rung, admissible = choose(
            cell["acc_ref"], cell["acc_lcb"], cell["flip_ucb"], self.safety_ucb, cell["cost"],
            op["delta"], op["phi"], op["psi"],
        )

        score = float(self._clf.predict_proba(emb[None, :])[0, 1])
        flagged = score >= self.threshold
        rung = resolve_rung(cell_rung, self.floor, flagged)

        why = f"{key} -> {cell_rung} (cheapest of {len(admissible)} admissible)"
        if flagged:
            why += f"; flagged (score {score:.2f}) -> floor {self.floor}" + ("" if rung == cell_rung else f" -> {rung}")
        else:
            why += f"; not flagged (score {score:.2f})"

        return Decision(
            category=category, label=label, system=system, max_tokens=max_tokens,
            pred_class=pred_class, predicted_required_precision=LADDER[pred_class],
            cell=key, cell_rung=cell_rung, admissible=admissible,
            safety_score=round(score, 4), flagged=flagged, floor=self.floor, rung=rung,
            why=why, operating_point=op,
        )

    def route(self, prompt: str, tolerance: str = "strict", force_quant: Optional[str] = None,
              decision: Optional[Decision] = None) -> dict[str, Any]:
        d = decision or self.decide(prompt, tolerance)
        rung = resolve_rung(d.rung, d.floor, d.flagged, force_quant)
        with self._lock:
            client, swapped, swap_s = self.pool().get(rung)
            t0 = time.time()
            result = client.generate(prompt, system_prompt=d.system, max_tokens=d.max_tokens)
            latency = time.time() - t0
        answer = pipeline.extract(d.label, result.text)
        out = d.public()
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
