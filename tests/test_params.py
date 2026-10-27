"""Regression test for the tied-embedding detection in
quants.compute_n_params_and_layers -- a silent regression here would
corrupt n_params without any obvious symptom (the file still loads fine;
only the parameter count would be wrong)."""

import numpy as np
import pytest

from quantlab import quants


class _FakeTensor:
    def __init__(self, name, shape, data):
        self.name = name
        self.shape = np.array(shape)
        self.data = data
        self.n_elements = int(np.prod(shape))


class _FakeField:
    def __init__(self, value):
        self.parts = [np.array([value])]
        self.data = [0]


class _FakeReader:
    def __init__(self, tensors, n_layers=36):
        self.tensors = tensors
        self.fields = {"qwen2.block_count": _FakeField(n_layers)} if n_layers is not None else {}


@pytest.fixture
def fake_config(tmp_path, monkeypatch):
    cfg = {"paths": {"gguf_dir": str(tmp_path)}}
    monkeypatch.setattr(quants, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(quants, "FILES", {**quants.FILES, "F16": ["shard1.gguf", "shard2.gguf"]})
    return cfg


def test_tied_embeddings_counted_once(fake_config, monkeypatch):
    shared = np.random.rand(2048, 1000).astype(np.float32)
    shard1 = _FakeReader([
        _FakeTensor("token_embd.weight", (2048, 1000), shared),
        _FakeTensor("output.weight", (2048, 1000), shared.copy()),  # identical data, tied
        _FakeTensor("blk.0.attn.weight", (100, 100), np.zeros((100, 100))),
    ])
    shard2 = _FakeReader([
        _FakeTensor("blk.1.attn.weight", (50, 50), np.zeros((50, 50))),
    ], n_layers=None)

    readers = iter([shard1, shard2])
    monkeypatch.setattr(quants.gguf, "GGUFReader", lambda path: next(readers))

    n_params, n_layers = quants.compute_n_params_and_layers()
    expected = 2048 * 1000 + 100 * 100 + 50 * 50  # output.weight NOT double-counted
    assert n_params == expected
    assert n_layers == 36


def test_untied_embeddings_both_counted(fake_config, monkeypatch):
    shard1 = _FakeReader([
        _FakeTensor("token_embd.weight", (2048, 1000), np.ones((2048, 1000))),
        _FakeTensor("output.weight", (2048, 1000), np.zeros((2048, 1000))),  # different data
    ])
    shard2 = _FakeReader([], n_layers=None)

    readers = iter([shard1, shard2])
    monkeypatch.setattr(quants.gguf, "GGUFReader", lambda path: next(readers))

    n_params, n_layers = quants.compute_n_params_and_layers()
    expected = 2048 * 1000 * 2  # both counted: genuinely distinct parameters
    assert n_params == expected


def test_missing_block_count_raises(fake_config, monkeypatch):
    shard1 = _FakeReader([_FakeTensor("x", (10, 10), np.zeros((10, 10)))], n_layers=None)
    shard2 = _FakeReader([], n_layers=None)
    readers = iter([shard1, shard2])
    monkeypatch.setattr(quants.gguf, "GGUFReader", lambda path: next(readers))

    with pytest.raises(quants.QuantsError):
        quants.compute_n_params_and_layers()
