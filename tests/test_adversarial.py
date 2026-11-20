from quantlab.adversarial import (
    MARKERS,
    NEUTRAL_FILLER,
    attack_casual,
    attack_pad,
    attack_stuff,
    build_vocab,
    greedy_suffix,
    with_suffix,
)
from quantlab.router import normalize_routing_input

VOCAB = ["the", "a", "compute", "prove", "step", "nice", "ok", "hello"] + [f"w{i}" for i in range(200)]
WEIGHTS = {"compute": 3.0, "prove": 2.0, "step": 1.0, "nice": -2.0, "ok": -1.0}


def toy_objective(suffixes):
    # a linear "predictor": the score is the summed weight of the suffix tokens
    return [sum(WEIGHTS.get(t, 0.0) for t in s) for s in suffixes]


def test_greedy_search_improves_toy_linear_objective_upwards():
    res = greedy_suffix(toy_objective, VOCAB, length=4, passes=3, n_candidates=len(VOCAB), maximize=True)
    assert res["objective"] > res["initial_objective"]
    assert res["objective"] == 12.0                    # every position becomes "compute"
    assert res["trajectory"] == sorted(res["trajectory"])


def test_greedy_search_improves_toy_linear_objective_downwards():
    res = greedy_suffix(toy_objective, VOCAB, length=3, passes=3, n_candidates=len(VOCAB), maximize=False)
    assert res["objective"] == -6.0
    assert res["trajectory"] == sorted(res["trajectory"], reverse=True)


def test_greedy_search_is_deterministic_and_stops_early():
    a = greedy_suffix(toy_objective, VOCAB, length=4, n_candidates=20)
    b = greedy_suffix(toy_objective, VOCAB, length=4, n_candidates=20)
    assert a == b
    flat = greedy_suffix(lambda s: [0.0] * len(s), VOCAB, length=4, n_candidates=20)
    assert flat["passes_run"] == 1 and flat["gadget"] == "the the the the"


def test_attacks_are_deterministic():
    p = "What is 2 + 2?"
    assert attack_pad(p) == attack_pad(p) and attack_pad(p).startswith(p + "\n\n")
    assert len(NEUTRAL_FILLER.split()) == 200
    assert attack_stuff(p).endswith("compare all three results.")
    assert attack_casual(p) == "quick one, no big deal: What is 2 + 2?"
    assert with_suffix(p, ["a", "b"]) == p + "\n\na b"


def test_vocab_has_500_distinct_tokens_including_markers():
    texts = ["Compute the total of the numbers", "Explain the result step by step"] * 3 + \
            [" ".join(f"w{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(600))]
    v = build_vocab(texts)
    assert len(v) == len(set(v)) == 500
    assert set(MARKERS) <= set(v) and {str(d) for d in range(10)} <= set(v)


def test_normalisation_removes_padding_tricks():
    text = "Hello​   world.  Hello​ world.\x07 Next  sentence!"
    assert normalize_routing_input(text) == "Hello world. Next sentence!"
    assert normalize_routing_input(" ".join(["x"] * 300), max_words=256) == " ".join(["x"] * 256)
    # repeated filler collapses to one copy of each sentence
    padded = attack_pad("Is this review positive?")
    assert len(normalize_routing_input(padded).split()) < len(padded.split())
