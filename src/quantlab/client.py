"""Server client for one quantization rung.

Production: an OpenAI-compatible /v1/chat/completions endpoint served by
llama-server, with logprobs enabled -- logprobs are the primary confidence
signal used to label required precision (see labels.py). Development:
MockLocalClient, which needs no GPU and produces synthetic logprobs, so the
rest of the pipeline can be built and tested on any machine.

Every request in this project runs at temperature 0 with a fixed seed, for
determinism across the whole benchmark grid (see config.yaml: request.*).
`system_prompt` has no default here -- it always comes from
tasktype.profile_for(label).local_style via pipeline.prepare(), so a caller
can never accidentally fall back to a generic prompt that the rest of the
pipeline doesn't know about.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from typing import Optional

import requests

from .schema import GenResult


class LocalClientError(Exception):
    pass


class OpenAICompatClient:
    """Talks to one llama-server instance (one quantization rung)."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 60,
                 max_tokens: int = 512, api_key: str = "not-needed",
                 top_logprobs: int = 1, cache_prompt: bool = False):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.top_logprobs = top_logprobs
        self.cache_prompt = cache_prompt

    def generate(self, prompt: str, system_prompt: str, temperature: float = 0.0,
                 seed: Optional[int] = None, max_tokens: Optional[int] = None,
                 timeout_s: Optional[float] = None) -> GenResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "cache_prompt": self.cache_prompt,
        }
        if seed is not None:
            payload["seed"] = seed
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=timeout_s if timeout_s is not None else self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            raise LocalClientError(f"local model call failed: {exc}") from exc

        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LocalClientError(f"unexpected local response shape: {exc}") from exc

        mean_lp, min_lp = _logprob_stats(choice)
        usage = data.get("usage") or {}
        return GenResult(
            text=text,
            mean_logprob=mean_lp,
            min_logprob=min_lp,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            timings=data.get("timings"),
        )


def _logprob_stats(choice: dict) -> tuple[Optional[float], Optional[float]]:
    """Extract mean/min token logprob from an OpenAI-format choice. Tolerates
    servers that omit logprobs, and llama-server variants that report a raw
    probability field instead of a logprob."""
    content = (choice.get("logprobs") or {}).get("content") or []
    lps = []
    for t in content:
        if not isinstance(t, dict):
            continue
        if isinstance(t.get("logprob"), (int, float)):
            lps.append(float(t["logprob"]))
        elif isinstance(t.get("prob"), (int, float)):  # llama-server variant
            lps.append(math.log(max(float(t["prob"]), 1e-10)))
    if not lps:
        return None, None
    return sum(lps) / len(lps), min(lps)


def _det_rng(prompt: str, temperature: float, seed: Optional[int]) -> random.Random:
    """Deterministic RNG across processes (built-in hash() is randomized)."""
    key = f"{prompt}|{temperature:.3f}|{seed}".encode("utf-8", errors="replace")
    return random.Random(int.from_bytes(hashlib.md5(key).digest()[:8], "big"))


class MockLocalClient:
    """GPU-free stand-in with the same generate() interface as
    OpenAICompatClient. Answers simple arithmetic confidently (high
    logprobs); anything else gets a vague answer with low logprobs. Fully
    deterministic for a given (prompt, temperature, seed)."""

    _GUESSES = ["It might be A.", "Possibly B, hard to say.",
                "I believe the answer is C."]

    def __init__(self, **_ignored):
        self.model = "mock-local"

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.0,
                 seed: Optional[int] = None, **_kwargs) -> GenResult:
        rng = _det_rng(prompt, temperature, seed)
        m = re.search(r"(-?\d+)\s*([+\-*])\s*(-?\d+)", prompt)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            val = {"+": a + b, "-": a - b, "*": a * b}[op]
            return GenResult(
                text=f"Answer: {val}",
                mean_logprob=-0.12 + rng.uniform(-0.03, 0.03),
                min_logprob=-0.9,
                tokens_in=len(prompt) // 4,
                tokens_out=6,
            )
        # Unknown territory: low-confidence waffle; different seeds pick
        # different guesses.
        guess = self._GUESSES[(seed or 0) % len(self._GUESSES)]
        return GenResult(
            text=f"{guess}\nAnswer: {guess.split()[-1].rstrip('.,')}",
            mean_logprob=-1.9 + rng.uniform(-0.4, 0.4),
            min_logprob=-4.2,
            tokens_in=len(prompt) // 4,
            tokens_out=18,
        )
