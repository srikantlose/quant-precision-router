"""Shared data structures passed between the benchmark runner, the router,
and the server clients."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Task:
    """A single task to answer. `text` is the normalized prompt; `raw` keeps the
    original payload so callers can echo back ids/fields the grader expects."""

    id: str
    text: str
    raw: Any = None


@dataclass
class GenResult:
    """One generation from a model (one quantization rung)."""

    text: str
    mean_logprob: Optional[float] = None  # None when the backend gave no logprobs
    min_logprob: Optional[float] = None
    tokens_in: int = 0
    tokens_out: int = 0
    finish_reason: str = "stop"
    timings: Optional[dict] = None  # raw llama-server "timings" object, if returned


def task_id_for(text: str) -> str:
    """Stable fallback id when the incoming payload has none."""
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]
