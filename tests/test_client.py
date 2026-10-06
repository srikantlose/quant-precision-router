import math

import requests

from quantlab.client import MockLocalClient, OpenAICompatClient


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def test_payload_contains_logprobs_top_logprobs_seed_cache_prompt_temperature(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["payload"] = json
        return FakeResponse(200, payload={
            "choices": [{
                "message": {"content": "Answer: 42"},
                "finish_reason": "stop",
                "logprobs": {"content": [
                    {"token": "4", "logprob": -0.1},
                    {"token": "2", "logprob": -0.2},
                ]},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            "timings": {"predicted_per_second": 42.0},
        })

    monkeypatch.setattr(requests, "post", fake_post)
    client = OpenAICompatClient(base_url="http://127.0.0.1:8101/v1", model="local",
                                top_logprobs=1, cache_prompt=False)
    result = client.generate("What is 2+2?", system_prompt="Answer plainly.", seed=42)

    payload = captured["payload"]
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 1
    assert payload["seed"] == 42
    assert payload["cache_prompt"] is False
    assert payload["temperature"] == 0.0
    assert payload["messages"][0] == {"role": "system", "content": "Answer plainly."}
    assert payload["messages"][1] == {"role": "user", "content": "What is 2+2?"}

    assert result.text == "Answer: 42"
    assert abs(result.mean_logprob - (-0.15)) < 1e-9
    assert result.min_logprob == -0.2
    assert result.timings == {"predicted_per_second": 42.0}
    assert result.tokens_in == 10
    assert result.tokens_out == 2
    assert result.finish_reason == "stop"


def test_logprob_variant_path(monkeypatch):
    """llama-server variants that report a raw probability instead of a
    logprob must still produce a usable mean_logprob."""
    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResponse(200, payload={
            "choices": [{
                "message": {"content": "Answer: yes"},
                "finish_reason": "stop",
                "logprobs": {"content": [
                    {"token": "yes", "prob": math.exp(-0.3)},
                ]},
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            # no "timings" key in this response
        })

    monkeypatch.setattr(requests, "post", fake_post)
    client = OpenAICompatClient(base_url="http://127.0.0.1:8101/v1", model="local")
    result = client.generate("Is this correct?", system_prompt="Answer yes or no.")
    assert result.mean_logprob is not None
    assert abs(result.mean_logprob - (-0.3)) < 1e-6
    assert result.timings is None


def test_missing_logprobs_gives_none_not_an_error(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResponse(200, payload={
            "choices": [{"message": {"content": "Answer: 1"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })

    monkeypatch.setattr(requests, "post", fake_post)
    client = OpenAICompatClient(base_url="http://127.0.0.1:8101/v1", model="local")
    result = client.generate("1", system_prompt="sys")
    assert result.mean_logprob is None
    assert result.min_logprob is None


def test_mock_client_accepts_system_prompt_kwarg_for_interface_parity():
    client = MockLocalClient()
    result = client.generate("2 + 2", system_prompt="ignored by the mock", seed=1)
    assert result.text == "Answer: 4"
    assert result.mean_logprob is not None
