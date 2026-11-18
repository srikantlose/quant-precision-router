"""ServerPool unit tests against a fake LlamaServer -- no GPU, no Docker,
no real binaries needed."""

import pytest

import quantlab.server as server_mod
from quantlab.server import ServerPool


class FakeLlamaServer:
    instances: list["FakeLlamaServer"] = []

    def __init__(self, rung, ngl):
        self.rung = rung
        self.ngl = ngl
        self.process = None
        self.started = False
        self.stopped = False
        FakeLlamaServer.instances.append(self)

    def start(self, known_pids=frozenset()):
        self.started = True

    def stop(self):
        self.stopped = True

    def client(self):
        return f"client-for-{self.rung}"


@pytest.fixture(autouse=True)
def _reset_instances():
    FakeLlamaServer.instances = []
    yield


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr(server_mod, "LlamaServer", FakeLlamaServer)
    return ServerPool(budget_mb=100, vram_mb={"A": 60, "B": 60, "C": 60},
                      ngl={"A": 1, "B": 1, "C": 1})


def test_first_get_starts_a_server(pool):
    client, swapped, swap_s = pool.get("A")
    assert swapped is True
    assert client == "client-for-A"
    assert len(FakeLlamaServer.instances) == 1
    assert FakeLlamaServer.instances[0].started is True


def test_repeated_get_is_a_cache_hit(pool):
    pool.get("A")
    client, swapped, swap_s = pool.get("A")
    assert swapped is False
    assert swap_s == 0.0
    assert len(FakeLlamaServer.instances) == 1  # no second server started


def test_budget_forces_eviction_of_lru(pool):
    pool.get("A")  # resident: {A (60)}
    pool.get("B")  # A+B=120 > budget 100 -> evict A, resident: {B (60)}
    assert "A" not in pool._servers
    assert "B" in pool._servers
    assert FakeLlamaServer.instances[0].stopped is True  # A was stopped


def test_lru_order_respected_not_just_insertion_order(pool):
    pool.get("A")
    pool.get("B")  # evicts A
    pool.get("A")  # reload A -> evicts B (the only/LRU resident)
    assert "A" in pool._servers
    assert "B" not in pool._servers


def test_swap_occurred_flag_correct(pool):
    _, swapped_first, _ = pool.get("A")
    _, swapped_cached, _ = pool.get("A")
    _, swapped_new, _ = pool.get("B")
    assert swapped_first is True
    assert swapped_cached is False
    assert swapped_new is True


def test_three_rungs_each_fit_alone_but_not_together(pool):
    # budget=100, each costs 60 -> at most one resident at a time here.
    pool.get("A")
    pool.get("B")
    pool.get("C")
    assert len(pool._servers) == 1
    assert "C" in pool._servers


def test_shutdown_stops_all_residents(pool):
    pool.get("A")
    pool.shutdown()
    assert len(pool._servers) == 0
    assert all(inst.stopped for inst in FakeLlamaServer.instances)


def test_resident_lists_loaded_rungs_in_lru_order():
    from quantlab.server import ServerPool
    pool = ServerPool(budget_mb=4000, vram_mb={"Q4_K_M": 2000}, ngl={"Q4_K_M": 37})
    assert pool.resident() == []
    pool._lru = ["Q5_K_M", "Q4_K_M"]
    assert pool.resident() == ["Q5_K_M", "Q4_K_M"]
    assert pool.resident() is not pool._lru   # a copy, so callers cannot mutate the pool
