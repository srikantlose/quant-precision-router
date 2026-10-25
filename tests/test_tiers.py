from quantlab import tiers


def test_fit_thresholds_are_monotone():
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    requested = ["easy", "easy", "moderate", "moderate", "complex", "complex"]
    t1, t2 = tiers.fit_thresholds(scores, requested)
    assert t1 <= t2


def test_fit_thresholds_perfect_separation():
    scores = [0.0, 0.1, 5.0, 5.1, 10.0, 10.1]
    requested = ["easy", "easy", "moderate", "moderate", "complex", "complex"]
    t1, t2 = tiers.fit_thresholds(scores, requested)
    predicted = [tiers.tier_of(s, t1, t2) for s in scores]
    assert predicted == requested


def test_fit_thresholds_handles_empty_input():
    t1, t2 = tiers.fit_thresholds([], [])
    assert t1 == t2 == 0.0


def test_zscore_constant_values_no_division_by_zero():
    result = tiers._zscore([5.0, 5.0, 5.0])
    assert result == [0.0, 0.0, 0.0]


def test_zscore_empty_input():
    assert tiers._zscore([]) == []


def test_compute_scores_constant_feature_within_category_is_safe():
    tasks = [
        {"id": "a", "category": "math", "task": "2+2", "expected": "4"},
        {"id": "b", "category": "math", "task": "3+3", "expected": "6"},
    ]
    scores = tiers.compute_scores(tasks)
    assert "a" in scores and "b" in scores
    assert all(isinstance(v, float) for v in scores.values())


def test_tertile_thresholds_monotone_and_handles_small_n():
    assert tiers._tertile_thresholds([]) == (0.0, 0.0)
    t1, t2 = tiers._tertile_thresholds([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert t1 <= t2


def test_assign_heuristic_tiers_uses_tertile_fallback_without_generated_tasks():
    tasks = [
        {"id": "s1", "category": "math", "task": "What is 2+2?", "expected": "4",
         "source": "seed", "tier": None},
        {"id": "s2", "category": "math",
         "task": "A store sells 240 items at 15% off then 10% more off, "
                 "how many were sold after both discounts were applied "
                 "across three separate days of sales tracking?",
         "expected": "100", "source": "seed", "tier": None},
    ]
    result = tiers.assign_heuristic_tiers(tasks)
    assert result["fit_info"]["math"]["method"] == "tertile_fallback"
    assert set(result["tier_heuristic"]) == {"s1", "s2"}
    for v in result["tier_heuristic"].values():
        assert v in tiers.TIERS


def test_assign_heuristic_tiers_fits_on_generated_when_available():
    tasks = [
        {"id": "g1", "category": "math", "task": "2+2", "expected": "4",
         "source": "fireworks", "tier": "easy"},
        {"id": "g2", "category": "math",
         "task": "A multi step problem with several numbers 10 20 30 and "
                 "then and then and then more conjunctions involved here",
         "expected": "60", "source": "fireworks", "tier": "complex"},
        {"id": "s1", "category": "math", "task": "3+3", "expected": "6",
         "source": "seed", "tier": None},
    ]
    result = tiers.assign_heuristic_tiers(tasks)
    assert result["fit_info"]["math"]["method"] == "fitted"
    assert result["fit_info"]["math"]["n_generated"] == 2


def test_cross_validated_kappa_returns_none_without_generated_tasks():
    tasks = [{"id": "s1", "category": "math", "task": "2+2", "expected": "4",
             "source": "seed", "tier": None}]
    result = tiers.cross_validated_kappa(tasks)
    assert result["kappa"] is None
    assert result["n"] == 0
