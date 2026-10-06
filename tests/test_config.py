from quantlab.config import load_config


def test_int_env_override_tolerates_float_string(monkeypatch):
    monkeypatch.setenv("QUANTLAB_LLAMA_CTX_SIZE", "2048.0")
    cfg = load_config(path="__no_such_file__.yaml")
    assert cfg["llama"]["ctx_size"] == 2048


def test_int_env_override_plain_int_still_works(monkeypatch):
    monkeypatch.setenv("QUANTLAB_API_PORT", "9001")
    cfg = load_config(path="__no_such_file__.yaml")
    assert cfg["api"]["port"] == 9001


def test_float_env_override(monkeypatch):
    monkeypatch.setenv("QUANTLAB_POLICY_DELTA", "0.07")
    cfg = load_config(path="__no_such_file__.yaml")
    assert cfg["policy"]["delta"] == 0.07


def test_bool_env_override(monkeypatch):
    monkeypatch.setenv("QUANTLAB_ROUTING_NORMALIZE", "true")
    cfg = load_config(path="__no_such_file__.yaml")
    assert cfg["routing"]["normalize"] is True


def test_flat_top_level_scalar_env_override(monkeypatch):
    monkeypatch.setenv("QUANTLAB_SEED", "7")
    cfg = load_config(path="__no_such_file__.yaml")
    assert cfg["seed"] == 7


def test_list_valued_defaults_not_overridable_by_env(monkeypatch):
    # Structured values (lists/dicts) are config-file only -- there is no
    # single-string env encoding for them, so the env var is ignored.
    monkeypatch.setenv("QUANTLAB_MODEL_LADDER", "Q2_K,Q3_K_M")
    cfg = load_config(path="__no_such_file__.yaml")
    assert cfg["model"]["ladder"] == ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M",
                                      "Q6_K", "Q8_0", "F16"]


def test_file_overrides_defaults(tmp_path):
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text("policy:\n  delta: 0.09\n  phi: 0.5\n", encoding="utf-8")
    cfg = load_config(path=str(cfg_file))
    assert cfg["policy"]["delta"] == 0.09
    assert cfg["policy"]["phi"] == 0.5
    # un-overridden keys in the same section keep their default
    assert cfg["policy"]["psi"] == 0.05


def test_env_override_wins_over_file(monkeypatch, tmp_path):
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text("policy:\n  delta: 0.09\n", encoding="utf-8")
    monkeypatch.setenv("QUANTLAB_POLICY_DELTA", "0.11")
    cfg = load_config(path=str(cfg_file))
    assert cfg["policy"]["delta"] == 0.11
