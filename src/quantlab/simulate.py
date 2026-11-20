"""Swap simulation (plan Sec 9.8).

Replays a sequence of served rungs through an LRU pool whose loaded rungs must fit
a VRAM budget, as the live ServerPool does. Every miss (including the first load)
evicts least-recently-used rungs until the new one fits, then pays that rung's
measured load time. A rung larger than the budget (F16 on an 8 GB card) evicts
everything and runs alone. The request order used by the evaluation is a random
shuffle, which is close to a worst case for switching; that is stated in the report.
"""

from __future__ import annotations


def simulate_swaps(sequence: list[str], vram_mb: dict[str, float], budget_mb: float,
                   swap_cost_s: dict[str, float]) -> dict:
    resident: list[str] = []                       # least recently used first
    swaps, swap_s = 0, 0.0
    for rung in sequence:
        if rung in resident:
            resident.remove(rung)
            resident.append(rung)
            continue
        while resident and sum(vram_mb[r] for r in resident) + vram_mb[rung] > budget_mb:
            resident.pop(0)
        resident.append(rung)
        swaps += 1
        swap_s += swap_cost_s[rung]
    n = len(sequence)
    return {"n_requests": n, "swaps": swaps, "swap_s": swap_s,
            "swaps_per_1k": swaps * 1000 / n if n else 0.0}


def swap_gpu_h_per_1k(sim: dict, gpu_hourly_rate: float = 1.0) -> float:
    n = sim["n_requests"]
    return sim["swap_s"] / 3600 * 1000 / n * gpu_hourly_rate if n else 0.0
