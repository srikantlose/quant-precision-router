from quantlab.features import SURFACE_FEATURE_NAMES, surface_features


def test_surface_features_length_matches_names():
    feats = surface_features("What is 17 + 25? Then add 3.")
    assert len(feats) == len(SURFACE_FEATURE_NAMES)


def test_surface_features_detects_code():
    feats = surface_features("```python\ndef f(x):\n    return x + 1\n```")
    idx = SURFACE_FEATURE_NAMES.index("has_code")
    assert feats[idx] == 1.0


def test_surface_features_empty_text_no_crash():
    feats = surface_features("")
    assert len(feats) == len(SURFACE_FEATURE_NAMES)
    assert all(f == 0.0 for f in feats)


def test_surface_features_counts_numbers_and_math_ops():
    feats = surface_features("12 + 7 = 19, then 100 - 3")
    names = SURFACE_FEATURE_NAMES
    assert feats[names.index("n_numbers")] == 5
    assert feats[names.index("n_math_ops")] >= 2
