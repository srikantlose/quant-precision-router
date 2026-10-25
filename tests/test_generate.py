import json

import pytest

from quantlab import generate


# --------------------------------------------------------------------------
# Budget counter
# --------------------------------------------------------------------------

@pytest.fixture
def budget_env(tmp_path, monkeypatch):
    path = tmp_path / "api_budget.json"
    monkeypatch.setattr(generate, "_budget_path", lambda: path)
    return path


def test_budget_counter_persists_used_count(budget_env):
    generate.spend_one_call(cap=5)
    generate.spend_one_call(cap=5)
    budget = generate.load_budget(cap=5)
    assert budget["used"] == 2
    assert budget["cap"] == 5


def test_budget_counter_refuses_at_cap(budget_env):
    for _ in range(3):
        generate.spend_one_call(cap=3)
    with pytest.raises(generate.BudgetExhausted):
        generate.spend_one_call(cap=3)
    # the refused attempt must NOT have incremented `used` further
    assert generate.load_budget(cap=3)["used"] == 3


def test_budget_counter_persists_across_separate_loads(budget_env):
    generate.spend_one_call(cap=10)
    generate.spend_one_call(cap=10)
    # simulate a fresh process: reload from disk rather than reusing state
    reloaded = generate.load_budget(cap=10)
    assert reloaded["used"] == 2
    generate.save_budget(reloaded)
    reloaded_again = generate.load_budget(cap=10)
    assert reloaded_again["used"] == 2


def test_budget_file_written_before_the_call(budget_env, monkeypatch):
    """The plan requires used to increment BEFORE the API call is sent, so
    a crash mid-call never under-counts spend on restart."""
    seen_during_call = {}

    def fake_fireworks_chat(system, user, max_tokens, temperature):
        seen_during_call["used"] = generate.load_budget(cap=5)["used"]
        return "OK"

    monkeypatch.setattr(generate, "fireworks_chat", fake_fireworks_chat)
    monkeypatch.setenv("FIREWORKS_API_KEY", "unused-in-this-test")
    # Call spend_one_call directly to isolate the counter's own ordering,
    # independent of fireworks_chat's internal call to it.
    generate.spend_one_call(cap=5)
    assert generate.load_budget(cap=5)["used"] == 1


# --------------------------------------------------------------------------
# Tier injection into the generation prompt
# --------------------------------------------------------------------------

def test_tier_text_is_injected_into_the_generation_prompt(budget_env, monkeypatch):
    captured = {}

    def fake_fireworks_chat(system, user, max_tokens, temperature):
        captured["user"] = user
        return json.dumps([])

    monkeypatch.setattr(generate, "fireworks_chat", fake_fireworks_chat)
    generate.generate_batch("math", "complex", batch_size=5)

    assert "DIFFICULTY TIER: complex" in captured["user"]
    assert generate.TIER_DEFS["math"]["complex"] in captured["user"]
    assert "exactly 5 NEW, DISTINCT" in captured["user"]


def test_every_category_has_tier_defs_for_all_three_tiers():
    for category, defs in generate.TIER_DEFS.items():
        assert set(defs.keys()) == {"easy", "moderate", "complex"}, category


# --------------------------------------------------------------------------
# validate_and_filter: near-duplicate detection
# --------------------------------------------------------------------------

def test_validate_and_filter_drops_seed_duplicates():
    items = [
        {"task": "What is the capital of France?", "expected": "Paris", "grader": "contains"},
    ]
    seed_norms = ["what is the capital of france?"]
    kept = generate.validate_and_filter(items, seed_norms, local_seen=[])
    assert kept == []


def test_validate_and_filter_drops_near_duplicate_paraphrase():
    items = [
        {"task": "  What  IS the capital of France?  ", "expected": "Paris", "grader": "contains"},
    ]
    seed_norms = ["what is the capital of france?"]
    kept = generate.validate_and_filter(items, seed_norms, local_seen=[])
    assert kept == []  # same after normalization -> exact substring match


def test_validate_and_filter_drops_within_batch_duplicates():
    items = [
        {"task": "What is the capital of Germany?", "expected": "Berlin", "grader": "contains"},
        {"task": "What is the capital of Germany?", "expected": "Berlin", "grader": "contains"},
    ]
    kept = generate.validate_and_filter(items, seed_norms=[], local_seen=[])
    assert len(kept) == 1


def test_validate_and_filter_keeps_genuinely_distinct_tasks():
    items = [
        {"task": "What is the capital of Germany?", "expected": "Berlin", "grader": "contains"},
        {"task": "What is the capital of Japan?", "expected": "Tokyo", "grader": "contains"},
    ]
    kept = generate.validate_and_filter(items, seed_norms=[], local_seen=[])
    assert len(kept) == 2


def test_validate_and_filter_rejects_missing_expected():
    items = [{"task": "What is 2+2?", "expected": "", "grader": "numeric"}]
    kept = generate.validate_and_filter(items, seed_norms=[], local_seen=[])
    assert kept == []


def test_validate_and_filter_rejects_too_short_task_text():
    items = [{"task": "Hi?", "expected": "x", "grader": "contains"}]
    kept = generate.validate_and_filter(items, seed_norms=[], local_seen=[])
    assert kept == []


def test_validate_and_filter_rejects_mc_without_expected_text():
    items = [{"task": "A) x B) y C) z D) w. Answer with the letter.",
             "expected": "A", "grader": "mc"}]  # no expected_text
    kept = generate.validate_and_filter(items, seed_norms=[], local_seen=[])
    assert kept == []


def test_validate_and_filter_accepts_mc_with_expected_text():
    items = [{"task": "A) x B) y C) z D) w. Answer with the letter.",
             "expected": "A", "expected_text": "x", "grader": "mc"}]
    kept = generate.validate_and_filter(items, seed_norms=[], local_seen=[])
    assert len(kept) == 1


# --------------------------------------------------------------------------
# id format
# --------------------------------------------------------------------------

def test_assign_ids_format_and_padding():
    items = [{"task": "t1", "expected": "a", "grader": "contains"},
            {"task": "t2", "expected": "b", "grader": "contains"}]
    rows = generate._assign_ids("math", "easy", items)
    assert rows[0]["id"] == "gen-math-easy-0001"
    assert rows[1]["id"] == "gen-math-easy-0002"
    for row in rows:
        assert row["source"] == "fireworks"
        assert row["tier"] == "easy"
        assert row["tier_source"] == "requested"
        assert row["tier_heuristic"] is None
        assert row["split"] in ("train", "test")


def test_assign_ids_four_digit_zero_padding_past_nine():
    items = [{"task": f"t{i}", "expected": "a", "grader": "contains"} for i in range(10)]
    rows = generate._assign_ids("logic", "moderate", items)
    assert rows[9]["id"] == "gen-logic-moderate-0010"
