"""Builds dashboard-site/data.js from the result files and the raw grid.

The raw grid is gitignored, so this has to run on the machine that ran the
benchmark; the generated data.js is committed alongside the site.
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LADDER = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]
TIERS = ["easy", "moderate", "complex"]


def load(rel):
    with open(ROOT / rel, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(rel):
    with open(ROOT / rel, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    profiles = load("results/profiles.json")
    kl = load("results/kl.json")
    labels = load("results/labels.json")
    predictor = load("results/predictor_metrics.json")
    safety = load("results/safety.json")
    policy = load("results/policy.json")
    tasks = {t["id"]: t for t in load_jsonl("data/tasks/all.jsonl")}
    grid = load_jsonl("results/raw/grid.jsonl")

    categories = sorted({t["category"] for t in tasks.values()})

    bench_by_task = defaultdict(dict)
    latency = defaultdict(list)
    for row in grid:
        if row["slice"] != "bench":
            continue
        bench_by_task[row["task_id"]][row["rung"]] = row
        latency[row["rung"]].append(row["latency_s"])

    acc = defaultdict(lambda: [0, 0])          # (cat, tier, rung) -> [correct, n]
    acc_cat = defaultdict(lambda: [0, 0])      # (cat, rung)
    flips = defaultdict(lambda: [0, 0])        # rung -> [flipped, n]
    flips_cat = defaultdict(lambda: [0, 0])    # (cat, rung)
    acc_rung = defaultdict(lambda: [0, 0])
    for tid, rows in bench_by_task.items():
        task = tasks.get(tid)
        if task is None or "F16" not in rows:
            continue
        cat, tier = task["category"], task["tier"]
        ref = bool(rows["F16"]["correct"])
        for rung, row in rows.items():
            ok = bool(row["correct"])
            for key, table in (((cat, tier, rung), acc), ((cat, rung), acc_cat), (rung, acc_rung)):
                table[key][0] += ok
                table[key][1] += 1
            if rung != "F16":
                flips[rung][0] += ok != ref
                flips[rung][1] += 1
                flips_cat[(cat, rung)][0] += ok != ref
                flips_cat[(cat, rung)][1] += 1

    def rate(pair):
        return round(pair[0] / pair[1], 4) if pair[1] else None

    rungs = {}
    for r in LADDER:
        prof = profiles["rungs"][r]
        s = safety.get(f"{r}|train", {})
        rungs[r] = {
            "bpw": round(prof["bpw"], 3),
            "vram_mb": prof["vram_mb"],
            "load_s": round(prof["swap_cost_s"], 2),
            "file_gb": round(prof["file_bytes"] / 1e9, 2),
            "accuracy": rate(acc_rung[r]),
            "latency_s": round(sum(latency[r]) / len(latency[r]), 3),
            "flip_rate": rate(flips[r]) if r != "F16" else 0.0,
            "safety_flip_ucb": round(s.get("safety_flip_ucb", 0.0), 4),
            "harmful_refusal": s.get("harmful_refusal_rate"),
            "over_refusal": s.get("over_refusal_rate"),
            "injection_resistance": s.get("injection_resistance_rate"),
            "kld": kl["rungs"][r]["mean_kld"] if r in kl["rungs"] else 0.0,
            "same_top_pct": kl["rungs"][r]["same_top_token_pct"] if r in kl["rungs"] else 100.0,
        }

    cat_tier = {
        cat: {
            tier: {r: {"acc": rate(acc[(cat, tier, r)]), "n": acc[(cat, tier, r)][1]} for r in LADDER}
            for tier in TIERS
        }
        for cat in categories
    }
    cat_acc = {cat: {r: rate(acc_cat[(cat, r)]) for r in LADDER} for cat in categories}
    cat_flip = {cat: {r: rate(flips_cat[(cat, r)]) for r in LADDER if r != "F16"} for cat in categories}

    req_hist = {cat: {str(k): 0 for k in range(7)} | {"unsolvable": 0} for cat in categories}
    for tid, req in labels["req"].items():
        task = tasks.get(tid)
        if task is None:
            continue
        req_hist[task["category"]]["unsolvable" if req is None else str(req)] += 1

    cells = {
        key: {k: c[k] for k in ("rung", "low_support", "n_support_tasks", "acc_lcb", "acc_ref", "flip_ucb", "cost")}
        for key, c in policy["cells"].items()
    }

    data = {
        "meta": {
            "model": "Qwen2.5-3B-Instruct",
            "gpu": "RTX 4060 (8 GB)",
            "vram_total_mb": 8188,
            "n_grid_rows": len(grid),
            "n_bench_tasks": len(bench_by_task),
            "n_safety_tasks": len({r["task_id"] for r in grid if r["slice"] == "safety"}),
            "ladder": LADDER,
            "categories": categories,
            "tiers": TIERS,
            "tau": labels["tau"],
        },
        "rungs": rungs,
        "cat_tier": cat_tier,
        "cat_acc": cat_acc,
        "cat_flip": cat_flip,
        "req_hist": req_hist,
        "predictor": predictor,
        "classifier": safety.get("_classifier", {}),
        "policy": {
            "operating_point": policy["operating_point"],
            "safety_floor": policy["safety_floor"],
            "cells": cells,
        },
    }

    out = ROOT / "dashboard-site" / "data.js"
    out.write_text("window.QL = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
