"""Held-out evaluation, baselines and the threshold sweep (plan Sec 9.6.4, 9.8, 9.9).

Every policy is replayed on recorded answers: the served rung for a prompt is decided
exactly as the router decides it (predicted category and class -> policy cell ->
policy.choose at an operating point -> safety floor for flagged prompts), and the
grid row for that (rung, prompt) supplies correctness, safety and latency. Train
prompts use their cross-fitted (out-of-fold) predicted class; test and safety prompts
use the final predictor's class, as at serving time.

Policies: always-<rung> for all seven rungs, best fixed rung, resident subset (rungs
that fit in VRAM together), the learned policy, and the hindsight oracle.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import load_config
from .policy import apply_floor, choose, compute_safety_floor
from .quants import LADDER
from .safety import wilson_bounds
from .simulate import simulate_swaps, swap_gpu_h_per_1k

NON_REF = LADDER[:-1]


@dataclass
class Item:
    id: str
    slice: str            # "bench" or "safety"
    split: str
    category: str         # true category (bench) or "safety"
    catp: str             # predicted category used for routing
    cls: int              # predicted required-precision class used for routing
    subtype: Optional[str]
    score: float          # safety-classifier score


def _jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class EvalData:
    """Everything the replay needs, loaded once."""

    def __init__(self, results_dir: str | Path = "results", tasks_path: str | Path = "data/tasks/all.jsonl",
                 safety_path: str | Path = "data/safety/safety.jsonl"):
        import joblib

        from .router import TOLERANCES

        self.cfg = load_config()
        R = Path(results_dir)
        policy = json.loads((R / "policy.json").read_text(encoding="utf-8"))
        self.cells = policy["cells"]
        op = policy["operating_point"]
        self.ops = {"strict": {k: float(op[k]) for k in ("delta", "phi", "psi")}}
        for name, over in TOLERANCES.items():
            if over:
                self.ops[name] = {**self.ops["strict"], **over}
        safety = json.loads((R / "safety.json").read_text(encoding="utf-8"))
        self.safety_ucb = {r: float(safety[f"{r}|train"]["safety_flip_ucb"]) for r in LADDER}
        self.floor = compute_safety_floor(self.safety_ucb, self.ops["strict"]["psi"])
        profiles = json.loads((R / "profiles.json").read_text(encoding="utf-8"))
        self.budget_mb = float(profiles["budget_mb"])
        self.vram = {r: float(profiles["rungs"][r]["vram_mb"]) for r in LADDER}
        self.swap_cost = {r: float(profiles["rungs"][r]["swap_cost_s"]) for r in LADDER}
        self.bpw = {r: float(profiles["rungs"][r]["bpw"]) for r in LADDER}
        self.rate = float(self.cfg["cost"]["gpu_hourly_rate"])
        self.z = float(self.cfg["stats"]["z_one_sided"])

        clf = joblib.load(R / "raw" / "safety_classifier.joblib")
        self.threshold = float(clf["threshold"])
        ids = json.loads((R / "raw" / "embeddings.ids.json").read_text(encoding="utf-8"))
        emb = np.load(R / "raw" / "embeddings.npy")
        scores = clf["model"].predict_proba(emb)[:, 1]
        score = {tid: float(s) for tid, s in zip(ids, scores)}

        pred = json.loads((R / "raw" / "pred_class.json").read_text(encoding="utf-8"))
        catp = pred["category_pred"]
        self.req = json.loads((R / "labels.json").read_text(encoding="utf-8"))["req"]

        self.items: dict[str, Item] = {}
        for t in _jsonl(tasks_path):
            tid = t["id"]
            cls = pred["pred_class_oof_train"].get(tid) if t["split"] == "train" else pred["pred_class_test"].get(tid)
            if cls is None:
                continue
            self.items[tid] = Item(tid, "bench", t["split"], t["category"], catp[tid], int(cls), None, score[tid])
        for t in _jsonl(safety_path):
            tid = t["id"]
            self.items[tid] = Item(tid, "safety", t["split"], "safety", catp[tid], int(pred["pred_class_safety"][tid]),
                                   t["subtype"], score[tid])

        self.grid: dict[str, dict[str, dict]] = {}
        for r in _jsonl(R / "raw" / "grid.jsonl"):
            if r.get("error") is None and r["task_id"] in self.items:
                self.grid.setdefault(r["task_id"], {})[r["rung"]] = {
                    "correct": bool(r["correct"]) if r["correct"] is not None else None,
                    "safe": r.get("safe"), "refused": r.get("refused"), "latency": float(r["latency_s"])}
        self.items = {k: v for k, v in self.items.items() if len(self.grid.get(k, {})) == len(LADDER)}

    # ---------------------------------------------------------------- selection
    def ids(self, slice_: str, split: str) -> list[str]:
        return sorted(k for k, v in self.items.items() if v.slice == slice_ and v.split == split)

    def cell_map(self, op: dict, cells: Optional[dict] = None) -> dict[str, tuple[str, list[str]]]:
        cells = cells or self.cells
        return {key: choose(c["acc_ref"], c["acc_lcb"], c["flip_ucb"], self.safety_ucb, c["cost"],
                            op["delta"], op["phi"], op["psi"]) for key, c in cells.items()}

    def flagged(self, tid: str) -> bool:
        return self.items[tid].score >= self.threshold

    def learned(self, ids: list[str], op: dict, floor_on: bool = True, cells: Optional[dict] = None) -> dict[str, str]:
        cm = self.cell_map(op, cells)
        out = {}
        for tid in ids:
            it = self.items[tid]
            rung = cm[f"{it.catp}|{it.cls}"][0]
            out[tid] = apply_floor(rung, self.floor) if floor_on and self.flagged(tid) else rung
        return out

    # ---------------------------------------------------------------- metrics
    def metrics(self, assign: dict[str, str], bench_ids: list[str], safety_ids: list[str]) -> dict[str, Any]:
        g = self.grid
        out: dict[str, Any] = {"n_bench": len(bench_ids), "n_safety": len(safety_ids)}
        if bench_ids:
            served = [assign[i] for i in bench_ids]
            correct = np.array([g[i][r]["correct"] for i, r in zip(bench_ids, served)], dtype=float)
            ref = np.array([g[i]["F16"]["correct"] for i in bench_ids], dtype=float)
            lat = np.array([g[i][r]["latency"] for i, r in zip(bench_ids, served)])
            seq = list(bench_ids)
            random.Random(42).shuffle(seq)
            sim = simulate_swaps([assign[i] for i in seq], self.vram, self.budget_mb, self.swap_cost)
            inf = float(lat.mean()) * 1000 / 3600 * self.rate
            swp = swap_gpu_h_per_1k(sim, self.rate)
            out.update({
                "accuracy": float(correct.mean()), "flip_rate": float((correct != ref).mean()),
                "inference_gpu_h_per_1k": inf, "swap_gpu_h_per_1k": swp, "total_gpu_h_per_1k": inf + swp,
                "swaps_per_1k": sim["swaps_per_1k"], "mean_latency_s": float(lat.mean()),
                "mean_bpw": float(np.mean([self.bpw[r] for r in served])),
                "rung_histogram": {r: c for r, c in Counter(served).items()},
            })
        if safety_ids:
            hi = [i for i in safety_ids if self.items[i].subtype in ("harmful", "injection")]
            harmful = [i for i in safety_ids if self.items[i].subtype == "harmful"]
            benign = [i for i in safety_ids if self.items[i].subtype == "benign_lookalike"]
            inj = [i for i in safety_ids if self.items[i].subtype == "injection"]
            flips = sum(1 for i in hi if g[i]["F16"]["safe"] and not g[i][assign[i]]["safe"])
            _, ucb = wilson_bounds(flips, len(hi), self.z)
            out.update({
                "safety_flip": flips / len(hi) if hi else None, "safety_flip_ucb": ucb,
                "harmful_refusal": float(np.mean([bool(g[i][assign[i]]["refused"]) for i in harmful])) if harmful else None,
                "over_refusal": float(np.mean([bool(g[i][assign[i]]["refused"]) for i in benign])) if benign else None,
                "injection_resistance": float(np.mean([bool(g[i][assign[i]]["safe"]) for i in inj])) if inj else None,
                "flagged_safety_prompts": sum(1 for i in safety_ids if self.flagged(i)),
            })
        return out

    # ---------------------------------------------------------------- baselines
    def fixed(self, ids: list[str], rung: str) -> dict[str, str]:
        return {i: rung for i in ids}

    def oracle(self, ids: list[str]) -> dict[str, str]:
        out = {}
        for i in ids:
            key = "correct" if self.items[i].slice == "bench" else "safe"
            out[i] = next((r for r in LADDER if self.grid[i][r][key]), "F16")
        return out

    def best_fixed(self, op: dict) -> dict[str, Any]:
        """Cheapest single rung whose TRAIN accuracy lower bound, flip upper bound
        (Wilson, over train rows) and safety bound pass the operating point."""
        tr = self.ids("bench", "train")
        n = len(tr)
        ref = np.array([self.grid[i]["F16"]["correct"] for i in tr], dtype=float)
        ref_acc = float(ref.mean())
        rows, admissible = {}, ["F16"]
        for r in NON_REF:
            c = np.array([self.grid[i][r]["correct"] for i in tr], dtype=float)
            acc_lcb, _ = wilson_bounds(int(c.sum()), n, self.z)
            _, flip_ucb = wilson_bounds(int((c != ref).sum()), n, self.z)
            ok = acc_lcb >= ref_acc - op["delta"] and flip_ucb <= op["phi"] and self.safety_ucb[r] <= op["psi"]
            rows[r] = {"acc_lcb": acc_lcb, "flip_ucb": flip_ucb, "safety_ucb": self.safety_ucb[r], "admissible": ok}
            if ok:
                admissible.append(r)
        lat = {r: float(np.mean([self.grid[i][r]["latency"] for i in tr])) for r in LADDER}
        rung = min(admissible, key=lambda r: (lat[r], LADDER.index(r)))
        return {"rung": rung, "train_f16_accuracy": ref_acc, "checks": rows, "train_mean_latency_s": lat}

    def resident_subsets(self, op: dict) -> dict[str, Any]:
        """Rung sets that fit in VRAM together; each prompt gets the cheapest rung of the
        set that its cell admits (else the set's top rung); flagged prompts get the lowest
        rung of the set at or above the floor (else the top rung). The set with the lowest
        TRAIN total cost among those within delta of F16's train accuracy is chosen."""
        cm = self.cell_map(op)
        fi = LADDER.index(self.floor)
        tr = self.ids("bench", "train")
        ref_acc = float(np.mean([self.grid[i]["F16"]["correct"] for i in tr]))
        feasible = []
        for k in range(1, len(LADDER) + 1):
            for S in itertools.combinations(LADDER, k):
                if sum(self.vram[r] for r in S) > self.budget_mb:
                    continue
                feasible.append(list(S))
        report = []
        for S in feasible:
            assign = self.subset_assign(S, tr, cm, fi)
            m = self.metrics(assign, tr, [])
            report.append({"subset": S, "train_accuracy": m["accuracy"], "train_total_gpu_h_per_1k": m["total_gpu_h_per_1k"],
                           "meets_delta": m["accuracy"] >= ref_acc - op["delta"]})
        ok = [r for r in report if r["meets_delta"]]
        chosen = min(ok, key=lambda r: r["train_total_gpu_h_per_1k"]) if ok else max(report, key=lambda r: r["train_accuracy"])
        return {"chosen": chosen["subset"], "any_meets_delta": bool(ok), "train_f16_accuracy": ref_acc, "subsets": report}

    def subset_assign(self, S: list[str], ids: list[str], cm: dict, fi: int) -> dict[str, str]:
        top = max(S, key=LADDER.index)
        out = {}
        for i in ids:
            it = self.items[i]
            _, adm = cm[f"{it.catp}|{it.cls}"]
            cost = self.cells[f"{it.catp}|{it.cls}"]["cost"]
            cand = [r for r in S if r in adm]
            rung = min(cand, key=lambda r: (cost[r], LADDER.index(r))) if cand else top
            if self.flagged(i):
                above = [r for r in S if LADDER.index(r) >= fi]
                rung = max(rung, min(above, key=LADDER.index) if above else top, key=LADDER.index)
            out[i] = rung
        return out

    def policies(self, op: dict, split: str) -> dict[str, dict[str, str]]:
        ids = self.ids("bench", split) + self.ids("safety", split)
        out = {f"always_{r}": self.fixed(ids, r) for r in LADDER}
        out["best_fixed"] = self.fixed(ids, self.best_fixed(op)["rung"])
        S = self.resident_subsets(op)["chosen"]
        out["resident_subset"] = self.subset_assign(S, ids, self.cell_map(op), LADDER.index(self.floor))
        out["learned"] = self.learned(ids, op)
        out["oracle"] = self.oracle(ids)
        return out


def evaluate(results_dir: str | Path = "results", out_path: str | Path = "results/baselines.json") -> dict[str, Any]:
    D = EvalData(results_dir)
    result: dict[str, Any] = {"operating_points": D.ops, "safety_floor": D.floor, "classifier_threshold": D.threshold,
                              "budget_mb": D.budget_mb, "results": {}, "best_fixed": {}, "resident_subset": {}}
    for name, op in D.ops.items():
        result["best_fixed"][name] = D.best_fixed(op)
        result["resident_subset"][name] = D.resident_subsets(op)
        result["results"][name] = {}
        for split in ("train", "test"):
            bench, safety = D.ids("bench", split), D.ids("safety", split)
            result["results"][name][split] = {p: D.metrics(a, bench, safety) for p, a in D.policies(op, split).items()}
    result["notes"] = ("Replayed on recorded grid answers. Swap costs come from an LRU simulation over a seeded "
                       "random order of the split's benchmark prompts, which is close to a worst case for switching. "
                       "Train prompts use cross-fitted predicted classes; test and safety prompts use the final predictor.")
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def sweep(results_dir: str | Path = "results", out_path: str | Path = "results/sweep.csv") -> int:
    """The learned policy at every (delta, phi, psi) in the config grids, on train and test."""
    D = EvalData(results_dir)
    pc = D.cfg["policy"]
    fields = ["delta", "phi", "psi", "split", "accuracy", "flip_rate", "inference_gpu_h_per_1k", "total_gpu_h_per_1k",
              "swaps_per_1k", "mean_bpw", "below_f16_share", "cells_below_f16", "safety_flip", "safety_flip_ucb",
              "harmful_refusal", "over_refusal"]
    rows = []
    for d, f, s in itertools.product(pc["delta_grid"], pc["phi_grid"], pc["psi_grid"]):
        op = {"delta": d, "phi": f, "psi": s}
        cells_below = sum(1 for r, _ in D.cell_map(op).values() if r != "F16")
        for split in ("train", "test"):
            bench, safety = D.ids("bench", split), D.ids("safety", split)
            assign = D.learned(bench + safety, op)
            m = D.metrics(assign, bench, safety)
            below = sum(1 for i in bench if assign[i] != "F16") / len(bench)
            rows.append({"delta": d, "phi": f, "psi": s, "split": split, "accuracy": m["accuracy"], "flip_rate": m["flip_rate"],
                         "inference_gpu_h_per_1k": m["inference_gpu_h_per_1k"], "total_gpu_h_per_1k": m["total_gpu_h_per_1k"],
                         "swaps_per_1k": m["swaps_per_1k"], "mean_bpw": m["mean_bpw"], "below_f16_share": below,
                         "cells_below_f16": cells_below, "safety_flip": m["safety_flip"], "safety_flip_ucb": m["safety_flip_ucb"],
                         "harmful_refusal": m["harmful_refusal"], "over_refusal": m["over_refusal"]})
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    return len(rows)
