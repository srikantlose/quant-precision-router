from quantlab.audit import cohen_kappa, parse_labels, stratified_sample


def test_parse_labels_accepts_a_clean_list_and_rejects_bad_ones():
    assert parse_labels('["FULL_REFUSAL", "full_compliance"]', 2) == ["FULL_REFUSAL", "FULL_COMPLIANCE"]
    assert parse_labels('Here: ["PARTIAL_REFUSAL"]', 1) == ["PARTIAL_REFUSAL"]
    assert parse_labels('["FULL_REFUSAL"]', 2) is None            # wrong length
    assert parse_labels('["MAYBE"]', 1) is None                  # unknown label
    assert parse_labels("no list", 1) is None


def test_cohen_kappa_extremes():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert abs(cohen_kappa([True, False, True, False], [False, True, False, True]) + 1.0) < 1e-9


def test_stratified_sample_is_proportional_and_deterministic():
    rows = [{"task_id": f"h{i}", "rung": r} for i in range(20) for r in ("Q2_K", "F16")] + \
           [{"task_id": f"b{i}", "rung": r} for i in range(60) for r in ("Q2_K", "F16")]
    sub = {**{f"h{i}": "harmful" for i in range(20)}, **{f"b{i}": "benign_lookalike" for i in range(60)}}
    s1, s2 = stratified_sample(rows, sub, 40), stratified_sample(rows, sub, 40)
    assert s1 == s2 and len(s1) == 40
    assert sum(1 for r in s1 if sub[r["task_id"]] == "harmful") == 10
