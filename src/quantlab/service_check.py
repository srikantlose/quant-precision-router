"""Live check of the routing API on the real GPU pool (P12).

1. Six test-split prompts (2 math, 2 code_gen, 1 XSTest safe, 1 XSTest unsafe) go through
   POST /route at both tolerances. Each response must be 200, its chosen_quant must equal
   router.decide for the same prompt, and the unsafe prompt must be flagged and served at
   or above the safety floor.
2. With force_quant enabled in a temporary config, five test tasks are requested at forced
   rungs and the extracted answer is compared with the grid's answer for the same
   (rung, task): the exact-match rate measures serving determinism.

The check uses a temporary config with the per-key limits raised so that it measures the
routing path rather than the rate limiter (which has its own unit tests and is exercised
in the adversarial evaluation). Results go to results/service_check.json.
"""

from __future__ import annotations

import copy
import json
import secrets
import time
from pathlib import Path
from typing import Any, Callable

from .quants import LADDER

FORCED_RUNGS = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q6_K", "Q8_0"]


def _jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_prompts(tasks_path: str = "data/tasks/all.jsonl", safety_path: str = "data/safety/safety.jsonl") -> list[dict]:
    bench = sorted((t for t in _jsonl(tasks_path) if t["split"] == "test"), key=lambda t: t["id"])
    safety = sorted((t for t in _jsonl(safety_path) if t["split"] == "test"), key=lambda t: t["id"])
    out = []
    for cat in ("math", "code_gen"):
        out += [{"kind": cat, "id": t["id"], "prompt": t["task"]} for t in bench if t["category"] == cat][:2]
    out += [{"kind": "xstest_safe", "id": t["id"], "prompt": t["task"]} for t in safety if t["subtype"] == "benign_lookalike"][:1]
    out += [{"kind": "xstest_unsafe", "id": t["id"], "prompt": t["task"]} for t in safety if t["subtype"] == "harmful"][:1]
    return out


def run_service_check(out_path: str | Path = "results/service_check.json",
                      log: Callable[[str], None] = print) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from .api import create_app
    from .config import load_config
    from .router import Router

    cfg = copy.deepcopy(load_config())
    cfg["api"]["rate_per_min"] = 1000
    cfg["api"]["f16_rate_per_min"] = 1000
    cfg["api"]["allow_force_quant"] = True
    key = secrets.token_urlsafe(48)
    router = Router(cfg=cfg)
    app = create_app(router, cfg, api_key=key)
    hdr = {"X-API-Key": key}
    t0 = time.time()
    checks: list[dict[str, Any]] = []
    try:
        with TestClient(app) as client:
            assert client.get("/health").json() == {"status": "ok"}
            assert client.post("/route", json={"prompt": "hi"}).status_code == 401
            for item in pick_prompts():
                for tol in ("strict", "balanced"):
                    expected = router.decide(item["prompt"], tol)
                    r = client.post("/route", json={"prompt": item["prompt"], "tolerance": tol}, headers=hdr)
                    body = r.json() if r.status_code == 200 else {}
                    row = {"kind": item["kind"], "id": item["id"], "tolerance": tol, "status": r.status_code,
                           "chosen_quant": body.get("chosen_quant"), "decided": expected.rung,
                           "flagged": body.get("flagged"), "category": body.get("category"),
                           "latency_s": body.get("latency_s"), "swap_occurred": body.get("swap_occurred")}
                    row["ok"] = r.status_code == 200 and body.get("chosen_quant") == expected.rung
                    if item["kind"] == "xstest_unsafe":
                        row["ok"] = row["ok"] and bool(body.get("flagged")) and \
                            LADDER.index(body["chosen_quant"]) >= LADDER.index(router.floor)
                    checks.append(row)
                    log(f"  {item['kind']:<14} {tol:<8} -> {row['chosen_quant']} (decided {row['decided']}) "
                        f"{'ok' if row['ok'] else 'FAIL'}")

            grid = {(g["task_id"], g["rung"]): g for g in _jsonl(cfg["paths"]["grid"]) if g["slice"] == "bench"}
            bench = sorted((t for t in _jsonl(cfg["paths"]["tasks_all"]) if t["split"] == "test"), key=lambda t: t["id"])
            seen, chosen = set(), []
            for t in bench:                                   # five tasks from five different categories
                if t["category"] not in seen and all((t["id"], r) in grid for r in FORCED_RUNGS):
                    seen.add(t["category"])
                    chosen.append(t)
                if len(chosen) == len(FORCED_RUNGS):
                    break
            determinism = []
            for t, rung in zip(chosen, FORCED_RUNGS):
                r = client.post("/route", json={"prompt": t["task"], "force_quant": rung}, headers=hdr)
                body = r.json() if r.status_code == 200 else {}
                g = grid[(t["id"], rung)]
                served = body.get("chosen_quant")
                determinism.append({"id": t["id"], "category": t["category"], "forced": rung, "served": served,
                                    "status": r.status_code, "answer": body.get("answer"), "grid_answer": g["answer"],
                                    "exact_match": served == rung and body.get("answer") == g["answer"]})
                log(f"  forced {rung:<7} {t['id']:<12} exact match: {determinism[-1]['exact_match']}")
    finally:
        router.shutdown()

    rate = sum(d["exact_match"] for d in determinism) / max(1, len(determinism))
    result = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
        "safety_floor": router.floor,
        "limits_note": "per-key limits raised in a temporary config for this check; force_quant enabled",
        "routing_checks": checks,
        "passed": all(c["ok"] for c in checks),
        "determinism": determinism,
        "determinism_exact_match_rate": rate,
    }
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
