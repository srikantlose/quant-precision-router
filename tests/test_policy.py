"""Tests for the selection rule (plan Sec 9.6) -- the one place that decides
which rung a cell gets served. test_accuracy_free_bug_is_rejected is the
regression test for the exact bug a prior review round caught: a rule that
only checks fidelity (flip rate) will happily hand a cell to a rung that is
cheap, low-flip, AND wrong."""

from quantlab.policy import LADDER, apply_floor, choose, compute_safety_floor


def _base_kwargs(**overrides):
    kwargs = dict(
        acc_ref=0.90,
        acc_lcb={"Q2_K": 0.89, "Q3_K_M": 0.89, "Q4_K_M": 0.89, "Q5_K_M": 0.89,
                "Q6_K": 0.89, "Q8_0": 0.89},
        flip_ucb={"Q2_K": 0.05, "Q3_K_M": 0.05, "Q4_K_M": 0.05, "Q5_K_M": 0.05,
                 "Q6_K": 0.05, "Q8_0": 0.05},
        safety_flip_ucb={"Q2_K": 0.0, "Q3_K_M": 0.0, "Q4_K_M": 0.0, "Q5_K_M": 0.0,
                        "Q6_K": 0.0, "Q8_0": 0.0},
        cost={"Q2_K": 0.1, "Q3_K_M": 0.2, "Q4_K_M": 0.3, "Q5_K_M": 0.4,
             "Q6_K": 0.5, "Q8_0": 0.6, "F16": 1.0},
        delta=0.02, phi=0.10, psi=0.05,
    )
    kwargs.update(overrides)
    return kwargs


def test_accuracy_free_bug_is_rejected():
    kwargs = _base_kwargs()
    kwargs["flip_ucb"]["Q2_K"] = 0.01          # low flip rate
    kwargs["acc_lcb"]["Q2_K"] = 0.30           # but accuracy LCB is far below ref - delta
    chosen, admissible = choose(**kwargs)
    assert "Q2_K" not in admissible
    assert chosen != "Q2_K"


def test_f16_always_admissible():
    kwargs = _base_kwargs()
    for b in LADDER[:-1]:
        kwargs["acc_lcb"][b] = 0.0
        kwargs["flip_ucb"][b] = 1.0
    chosen, admissible = choose(**kwargs)
    assert admissible == ["F16"]
    assert chosen == "F16"


def test_cheapest_admissible_is_chosen():
    chosen, admissible = choose(**_base_kwargs())
    assert set(admissible) == {"Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"}
    assert chosen == "Q2_K"


def test_ties_broken_by_lower_ladder_index():
    kwargs = _base_kwargs()
    kwargs["cost"] = {b: 0.3 for b in kwargs["cost"]}
    kwargs["cost"]["Q2_K"] = 0.1
    kwargs["cost"]["Q4_K_M"] = 0.1
    chosen, _ = choose(**kwargs)
    assert chosen == "Q2_K"


def test_safety_condition_excludes_a_rung_everywhere():
    kwargs = _base_kwargs()
    kwargs["safety_flip_ucb"]["Q2_K"] = 0.5   # far above psi
    _, admissible = choose(**kwargs)
    assert "Q2_K" not in admissible


def test_compute_safety_floor_and_apply_floor():
    safety_flip_ucb = {"Q2_K": 0.20, "Q3_K_M": 0.08, "Q4_K_M": 0.03,
                       "Q5_K_M": 0.01, "Q6_K": 0.01, "Q8_0": 0.0}
    floor = compute_safety_floor(safety_flip_ucb, psi=0.05)
    assert floor == "Q4_K_M"

    assert apply_floor("Q2_K", floor) == "Q4_K_M"   # raised
    assert apply_floor("Q8_0", floor) == "Q8_0"      # already above floor, unchanged
    assert apply_floor("Q4_K_M", floor) == "Q4_K_M"  # exactly at floor


def test_compute_safety_floor_f16_always_qualifies():
    safety_flip_ucb = {b: 1.0 for b in LADDER[:-1]}  # every non-F16 rung fails
    floor = compute_safety_floor(safety_flip_ucb, psi=0.05)
    assert floor == "F16"
