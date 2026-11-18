"""The routing API's guard rails (plan P12), with a fake router: no GPU, no models."""

import copy

import pytest
from fastapi.testclient import TestClient

from quantlab.api import TokenBucket, create_app
from quantlab.config import load_config
from quantlab.router import Decision, resolve_rung

KEY = "k" * 40


class FakeRouter:
    def __init__(self, rung="Q5_K_M", flagged=False, floor="Q5_K_M"):
        self.rung, self.flagged, self.floor, self.threshold = rung, flagged, floor, 0.42
        self.routed = []

    def decide(self, prompt, tolerance="strict"):
        return Decision(category="math", label="math", system="sys", max_tokens=96, pred_class=1,
                        predicted_required_precision="Q3_K_M", cell="math|1", cell_rung=self.rung,
                        admissible=["F16", self.rung], safety_score=0.9 if self.flagged else 0.1,
                        flagged=self.flagged, floor=self.floor, rung=resolve_rung(self.rung, self.floor, self.flagged),
                        why="test", operating_point={"delta": 0.02, "phi": 0.1, "psi": 0.05})

    def route(self, prompt, tolerance="strict", force_quant=None, decision=None):
        d = decision or self.decide(prompt, tolerance)
        served = resolve_rung(d.rung, d.floor, d.flagged, force_quant)
        self.routed.append(served)
        return {**d.public(), "answer": "42", "chosen_quant": served, "forced": bool(force_quant),
                "latency_s": 0.1, "tokens_out": 3, "est_cost_gpu_h": 0.0, "swap_occurred": False,
                "swap_cost_s": 0.0, "vram_mb": 2450, "classifier_threshold": 0.42}

    def summary(self):
        return {"cells": {}}

    def shutdown(self):
        pass


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    c["paths"]["results_dir"] = str(tmp_path)
    c["api"].update({"rate_per_min": 100, "f16_rate_per_min": 100, "max_prompt_chars": 50, "allow_force_quant": False})
    return c


def client(cfg, router=None, **api):
    cfg["api"].update(api)
    return TestClient(create_app(router=router or FakeRouter(), cfg=cfg, api_key=KEY))


def post(c, body, key=KEY):
    return c.post("/route", json=body, headers={"X-API-Key": key} if key else {})


def test_startup_refuses_without_a_long_enough_key(cfg):
    with pytest.raises(RuntimeError):
        create_app(router=FakeRouter(), cfg=cfg, api_key="")
    with pytest.raises(RuntimeError):
        create_app(router=FakeRouter(), cfg=cfg, api_key="short")


def test_health_and_page_need_no_key(cfg):
    c = client(cfg)
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/").status_code == 200
    assert c.get("/info").json()["allow_force_quant"] is False


def test_missing_or_wrong_key_is_401(cfg):
    c = client(cfg)
    assert post(c, {"prompt": "hi"}, key=None).status_code == 401
    assert post(c, {"prompt": "hi"}, key="x" * 40).status_code == 401
    assert c.get("/policy").status_code == 401
    assert c.get("/policy", headers={"X-API-Key": KEY}).status_code == 200


def test_oversized_prompt_is_413(cfg):
    assert post(client(cfg), {"prompt": "x" * 51}).status_code == 413


def test_force_quant_is_403_when_disabled(cfg):
    assert post(client(cfg), {"prompt": "hi", "force_quant": "Q2_K"}).status_code == 403


def test_rate_limit_returns_429_with_retry_after(cfg):
    c = client(cfg, rate_per_min=2)
    assert post(c, {"prompt": "a"}).status_code == 200
    assert post(c, {"prompt": "b"}).status_code == 200
    r = post(c, {"prompt": "c"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_f16_quota_is_separate_and_never_downgrades(cfg):
    router = FakeRouter(rung="F16")
    c = client(cfg, router=router, f16_rate_per_min=1)
    assert post(c, {"prompt": "a"}).status_code == 200
    r = post(c, {"prompt": "b"})
    assert r.status_code == 429
    assert router.routed == ["F16"]          # the second request was refused, not served cheaper


def test_forced_rung_never_goes_below_the_floor_when_flagged(cfg):
    router = FakeRouter(rung="Q8_0", flagged=True, floor="Q5_K_M")
    c = client(cfg, router=router, allow_force_quant=True)
    r = post(c, {"prompt": "a", "force_quant": "Q2_K"})
    assert r.status_code == 200 and r.json()["chosen_quant"] == "Q5_K_M"
    unflagged = client(cfg, router=FakeRouter(rung="Q8_0"), allow_force_quant=True)
    assert post(unflagged, {"prompt": "a", "force_quant": "Q2_K"}).json()["chosen_quant"] == "Q2_K"


def test_response_carries_the_decision_fields(cfg):
    body = post(client(cfg), {"prompt": "What is 2+2?"}).json()
    for k in ("answer", "category", "predicted_required_precision", "chosen_quant", "why", "flagged",
              "latency_s", "est_cost_gpu_h", "swap_occurred", "swap_cost_s"):
        assert k in body
    assert "system" not in body


def test_log_has_a_prompt_hash_and_never_the_prompt(cfg, tmp_path):
    post(client(cfg), {"prompt": "my secret question"})
    log = (tmp_path / "logs" / "service.jsonl").read_text(encoding="utf-8")
    assert "prompt_sha256" in log and "my secret question" not in log


def test_resolve_rung_floor_rules():
    assert resolve_rung("Q2_K", "Q5_K_M", flagged=True) == "Q5_K_M"
    assert resolve_rung("Q8_0", "Q5_K_M", flagged=True) == "Q8_0"
    assert resolve_rung("Q2_K", "Q5_K_M", flagged=False) == "Q2_K"
    assert resolve_rung("Q8_0", "Q5_K_M", flagged=True, force_quant="Q3_K_M") == "Q5_K_M"


def test_token_bucket_refuses_then_reports_wait():
    b = TokenBucket(1)
    assert b.take() == 0
    assert b.take() > 0
