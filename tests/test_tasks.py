import pytest

from quantlab.tasks import SeedConsolidationError, consolidate, split_for


def test_split_for_is_deterministic():
    assert split_for("math-01") == split_for("math-01")
    assert split_for("math-01") in ("train", "test")


def test_split_for_approximately_30_percent_over_many_ids():
    n = 10_000
    test_count = sum(1 for i in range(n) if split_for(f"synthetic-{i}") == "test")
    frac = test_count / n
    assert 0.28 <= frac <= 0.32, frac


def test_dedupe_keeps_first_occurrence():
    rows = [
        {"id": "math-01", "type": "math", "task": "What is 2+2?",
         "grader": "numeric", "expected": "4"},
        {"id": "math-01", "type": "math", "task": "What is 2+2?",
         "grader": "numeric", "expected": "4"},  # identical repeat
    ]
    out = consolidate(rows)
    assert len(out) == 1
    assert out[0]["id"] == "math-01"


def test_mismatched_duplicate_id_raises():
    rows = [
        {"id": "math-01", "type": "math", "task": "What is 2+2?",
         "grader": "numeric", "expected": "4"},
        {"id": "math-01", "type": "math", "task": "What is 3+3?",  # different text!
         "grader": "numeric", "expected": "6"},
    ]
    with pytest.raises(SeedConsolidationError):
        consolidate(rows)


def test_cross_id_normalized_duplicate_raises():
    rows = [
        {"id": "math-01", "type": "math", "task": "What is 2+2?",
         "grader": "numeric", "expected": "4"},
        {"id": "math-02", "type": "math", "task": "  what IS 2+2?  ",  # same normalized text
         "grader": "numeric", "expected": "4"},
    ]
    with pytest.raises(SeedConsolidationError):
        consolidate(rows)


def test_unknown_category_raises():
    rows = [{"id": "x-01", "type": "not_a_category", "task": "hello",
             "grader": "exact", "expected": "hi"}]
    with pytest.raises(SeedConsolidationError):
        consolidate(rows)


def test_mc_without_expected_text_is_valid():
    # Three real CheapRoute holdout tasks (mc-h1/h2/h3) rely on letter-only
    # matching and have no expected_text field at all.
    rows = [{"id": "mc-h1", "type": "multiple_choice",
             "task": "Which planet is closest to the Sun? A) Venus B) Mercury "
                     "C) Earth D) Mars. Answer with the letter.",
             "expected": "B", "grader": "mc"}]
    out = consolidate(rows)
    assert out[0]["expected_text"] is None


def test_mc_without_expected_letter_raises():
    rows = [{"id": "mc-01", "type": "multiple_choice", "task": "A) x B) y",
             "grader": "mc"}]
    with pytest.raises(SeedConsolidationError):
        consolidate(rows)


def test_contains_all_requires_nonempty_list():
    rows = [{"id": "ner-01", "type": "ner", "task": "Extract entities from: X",
             "grader": "contains_all", "expected": []}]
    with pytest.raises(SeedConsolidationError):
        consolidate(rows)


def test_word_limit_requires_max_words_dict():
    rows = [{"id": "sum-01", "type": "summarization", "task": "Summarize: X",
             "grader": "word_limit", "expected": {"any_of": ["x"]}}]
    with pytest.raises(SeedConsolidationError):
        consolidate(rows)


def test_py_exec_requires_test_code():
    rows = [{"id": "gen-01", "type": "code_gen", "task": "Write a function",
             "grader": "py_exec", "expected": "def f(): pass"}]
    with pytest.raises(SeedConsolidationError):
        consolidate(rows)


def test_consolidated_rows_carry_expected_fields():
    rows = [{"id": "math-01", "type": "math", "task": "What is 2+2?",
             "grader": "numeric", "expected": "4"}]
    out = consolidate(rows)
    row = out[0]
    assert row["source"] == "seed"
    assert row["split"] in ("train", "test")
    assert row["tier"] is None
    assert row["tier_source"] is None
    assert row["tier_heuristic"] is None
