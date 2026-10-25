import hashlib
import json

import pytest

from quantlab import quants


def _fake_cfg(tmp_path, port_base=8101):
    return {
        "paths": {"lock_file": str(tmp_path / "models.lock.json"),
                  "gguf_dir": str(tmp_path / "gguf")},
        "model": {"repo": "Qwen/Qwen2.5-3B-Instruct-GGUF"},
        "llama": {"port_base": port_base},
        "sandbox": {"image_digest": "python@sha256:deadbeef"},
    }


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    cfg = _fake_cfg(tmp_path)
    monkeypatch.setattr(quants, "load_config", lambda *a, **k: cfg)
    return tmp_path, cfg


def test_port_for_mapping(fake_env):
    tmp_path, cfg = fake_env
    assert quants.port_for("Q2_K") == cfg["llama"]["port_base"] + 0
    assert quants.port_for("Q4_K_M") == cfg["llama"]["port_base"] + 2
    assert quants.port_for("F16") == cfg["llama"]["port_base"] + 6


def test_ladder_and_files_cover_every_rung():
    assert set(quants.FILES.keys()) == set(quants.LADDER)
    assert quants.FILES["F16"] == [
        "qwen2.5-3b-instruct-fp16-00001-of-00002.gguf",
        "qwen2.5-3b-instruct-fp16-00002-of-00002.gguf",
    ]
    for rung, files in quants.FILES.items():
        if rung != "F16":
            assert len(files) == 1


def test_lock_file_round_trip(fake_env):
    tmp_path, cfg = fake_env
    lock = {"repo": cfg["model"]["repo"], "revision": "a" * 40,
            "files": {"Q4_K_M": [{"name": "x.gguf", "size": 123, "sha256": "b" * 64}]}}
    quants._save_lock(lock)
    loaded = quants._load_lock()
    assert loaded == lock

    # resolve_revision reads the pinned revision back from the lock file
    # without touching the hub, once it's already there.
    assert quants.resolve_revision() == "a" * 40


def test_model_path_points_at_first_file(fake_env):
    tmp_path, cfg = fake_env
    gguf_dir = tmp_path / "gguf"
    assert quants.model_path("Q4_K_M") == gguf_dir / "qwen2.5-3b-instruct-q4_k_m.gguf"
    assert quants.model_path("F16") == gguf_dir / "qwen2.5-3b-instruct-fp16-00001-of-00002.gguf"


def _write_file_and_lock(tmp_path, cfg, rung, filename, content: bytes):
    gguf_dir = tmp_path / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    path = gguf_dir / filename
    path.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    lock = {"repo": cfg["model"]["repo"], "revision": "a" * 40,
            "files": {rung: [{"name": filename, "size": len(content), "sha256": sha}]}}
    quants._save_lock(lock)
    return path


def test_verify_passes_for_matching_file(fake_env):
    tmp_path, cfg = fake_env
    _write_file_and_lock(tmp_path, cfg, "Q4_K_M", "q4.gguf", b"hello world" * 100)
    assert quants.verify("Q4_K_M") is True


def test_verify_detects_size_change(fake_env):
    tmp_path, cfg = fake_env
    path = _write_file_and_lock(tmp_path, cfg, "Q4_K_M", "q4.gguf", b"hello world" * 100)

    # Overwrite with different content of a different size -- verify() must
    # notice (size changed -> re-hash -> hash no longer matches the lock).
    path.write_bytes(b"corrupted, much shorter")
    assert quants.verify("Q4_K_M") is False


def test_verify_without_full_trusts_cache_when_unchanged(fake_env):
    tmp_path, cfg = fake_env
    _write_file_and_lock(tmp_path, cfg, "Q4_K_M", "q4.gguf", b"hello world" * 100)
    assert quants.verify("Q4_K_M", full=False) is True
    # Second call should read from .verified.json without re-hashing, and
    # still report True since nothing changed.
    assert quants.verify("Q4_K_M", full=False) is True
    cache_path = tmp_path / "gguf" / ".verified.json"
    assert cache_path.is_file()


def test_verify_missing_rung_raises(fake_env):
    with pytest.raises(quants.QuantsError):
        quants.verify("Q2_K")


def test_verify_missing_file_fails(fake_env):
    tmp_path, cfg = fake_env
    lock = {"repo": cfg["model"]["repo"], "revision": "a" * 40,
            "files": {"Q4_K_M": [{"name": "does-not-exist.gguf", "size": 1, "sha256": "f" * 64}]}}
    quants._save_lock(lock)
    assert quants.verify("Q4_K_M") is False
