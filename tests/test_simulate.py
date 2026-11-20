from quantlab.simulate import simulate_swaps, swap_gpu_h_per_1k

VRAM = {"A": 2000, "B": 1500, "C": 3000}
COST = {"A": 2.0, "B": 1.0, "C": 4.0}


def test_hand_worked_lru_sequence():
    # budget 4000: A (load), B (load, A+B=3500), A (hit), C (evict B then A, load),
    # A (evict C, load), B (A+B fits, load) -> 5 loads
    sim = simulate_swaps(["A", "B", "A", "C", "A", "B"], VRAM, 4000, COST)
    assert sim["swaps"] == 5
    assert sim["swap_s"] == 2.0 + 1.0 + 4.0 + 2.0 + 1.0


def test_all_hits_after_first_load():
    sim = simulate_swaps(["A"] * 10, VRAM, 4000, COST)
    assert sim["swaps"] == 1 and sim["swaps_per_1k"] == 100.0


def test_oversized_rung_runs_alone_and_evicts_everything():
    sim = simulate_swaps(["A", "B", "C", "A"], VRAM, 2500, COST)   # C alone exceeds the budget
    assert sim["swaps"] == 4


def test_swap_cost_conversion():
    sim = {"n_requests": 1000, "swap_s": 3600.0}
    assert swap_gpu_h_per_1k(sim) == 1.0
