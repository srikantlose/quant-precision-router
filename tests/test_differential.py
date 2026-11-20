from quantlab.differential import find_anomalies, two_proportion_p

RUNGS = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]


def test_two_proportion_p_is_small_for_a_large_gap_and_one_for_equal_rates():
    assert two_proportion_p(60, 100, 10, 100) < 1e-6
    assert two_proportion_p(10, 100, 10, 100) == 1.0


def test_a_planted_spike_at_one_rung_is_flagged():
    base = {r: (10, 120) for r in RUNGS}
    base["Q5_K_M"] = (60, 120)                       # 50% vs neighbours at 8%
    found = find_anomalies({"code_gen": base})
    assert [(a["category"], a["rung"]) for a in found] == [("code_gen", "Q5_K_M")]


def test_a_smooth_ladder_has_no_spikes_and_its_2bit_cliff_is_an_edge_collapse():
    smooth = dict(zip(RUNGS, [(45, 120), (25, 120), (15, 120), (12, 120), (9, 120), (5, 120)]))
    found = find_anomalies({"math": smooth})
    assert all(a["kind"] == "edge_collapse" for a in found)
    assert not [a for a in found if a["kind"] == "spike"]


def test_planted_spike_is_labelled_spike():
    base = {r: (10, 120) for r in RUNGS}
    base["Q5_K_M"] = (60, 120)
    assert find_anomalies({"code_gen": base})[0]["kind"] == "spike"
