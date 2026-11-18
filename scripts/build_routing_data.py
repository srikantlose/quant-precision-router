"""Builds docs/routing/routing-data.js for the routing flowchart PDF.

Every routing outcome is computed with the live router's own pieces: the
policy cells in results/policy.json, policy.choose(), the per-rung safety
bounds in results/safety.json, the floor rule, the TOLERANCES presets in
quantlab.router and the API limits in config.yaml. The flowchart therefore
shows exactly the decisions `quantlab serve` can make.
"""

import json
from pathlib import Path

from quantlab.config import load_config
from quantlab.policy import apply_floor, choose, compute_safety_floor
from quantlab.quants import LADDER
from quantlab.router import TOLERANCES
from quantlab.tasktype import CATEGORIES

ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg = load_config()
    results = ROOT / cfg["paths"]["results_dir"]
    policy = json.loads((results / "policy.json").read_text(encoding="utf-8"))
    safety = json.loads((results / "safety.json").read_text(encoding="utf-8"))
    profiles = json.loads((results / "profiles.json").read_text(encoding="utf-8"))
    threshold = float(json.loads((results / "safety.json").read_text(encoding="utf-8"))["_classifier"]["threshold"])

    ucb = {r: float(safety[f"{r}|train"]["safety_flip_ucb"]) for r in LADDER}
    base = {k: float(v) for k, v in policy["operating_point"].items()}
    floor = compute_safety_floor(ucb, base["psi"])
    ops = {}
    for name, over in TOLERANCES.items():
        op = dict(base)
        if over:
            op.update(over)
        ops[name] = op

    cells, maps, flagged_maps = {}, {}, {}
    for key, c in policy["cells"].items():
        cells[key] = {"n": c.get("n_support_tasks"), "low": bool(c.get("low_support"))}
    for name, op in ops.items():
        maps[name], flagged_maps[name] = {}, {}
        for key, c in policy["cells"].items():
            rung, _ = choose(c["acc_ref"], c["acc_lcb"], c["flip_ucb"], ucb, c["cost"],
                             op["delta"], op["phi"], op["psi"])
            maps[name][key] = rung
            flagged_maps[name][key] = apply_floor(rung, floor)

    api = cfg["api"]
    data = {
        "ladder": LADDER,
        "categories": sorted(CATEGORIES),
        "operating_points": ops,
        "safety_ucb": ucb,
        "floor": floor,
        "floor_psi": base["psi"],
        "threshold": threshold,
        "cells": cells,
        "map": maps,
        "flagged_map": flagged_maps,
        "api": {k: api[k] for k in ("rate_per_min", "f16_rate_per_min", "max_prompt_chars", "allow_force_quant", "host", "port")},
        "budget_mb": profiles["budget_mb"],
        "vram_mb": {r: profiles["rungs"][r]["vram_mb"] for r in LADDER},
        "load_s": {r: round(profiles["rungs"][r]["swap_cost_s"], 2) for r in LADDER},
        "request": {"temperature": cfg["request"]["temperature"], "seed": cfg["request"]["seed"]},
    }
    out = ROOT / "docs" / "routing" / "routing-data.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.RD = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    from collections import Counter
    for name in ops:
        print(name, dict(Counter(maps[name].values())), "| flagged:", dict(Counter(flagged_maps[name].values())))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
