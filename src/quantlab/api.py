"""Hardened HTTP API in front of the router (plan P12).

    quantlab serve                      # 127.0.0.1:8000, from config.yaml
    quantlab serve --host 0.0.0.0       # reachable from other machines on the network

Endpoints:
    GET  /          the "Ask QuantRoute" demo page (no secrets in it; it asks for the key)
    GET  /health    {"status": "ok"}, no auth
    GET  /info      public settings the page needs (limits, whether force_quant is on), no auth
    GET  /policy    which rung every cell serves at each tolerance, auth required
    POST /route     {prompt, tolerance?, force_quant?} -> answer + full decision, auth required
                    (tolerance "strict" = the fitted policy, "balanced" = looser per-cell bounds;
                    the floor for flagged prompts is the same in both)

Rules (plan P12): the server refuses to start without QUANTLAB_API_KEY (32+ characters);
the key travels in X-API-Key and is compared in constant time; oversized prompts get 413;
each key has a token bucket of api.rate_per_min requests and a separate bucket of
api.f16_rate_per_min requests whose served rung is F16 (an F16 request over quota is
rejected with 429, never silently downgraded, because a downgrade could bypass the
safety floor); force_quant is refused with 403 unless api.allow_force_quant is set,
and even then never goes below the floor for a flagged prompt. The log records a
SHA-256 of each prompt and never the prompt text.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import load_config

Rung = Literal["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]
PAGE = Path(__file__).resolve().parent / "static" / "ask.html"


class RouteRequest(BaseModel):
    prompt: str = Field(min_length=1)
    tolerance: Literal["strict", "balanced"] = "strict"
    force_quant: Optional[Rung] = None


class TokenBucket:
    """`rate_per_min` requests per minute, refilled continuously, burst = rate."""

    def __init__(self, rate_per_min: float):
        self.capacity = float(rate_per_min)
        self.tokens = float(rate_per_min)
        self.refill = float(rate_per_min) / 60.0
        self.t = time.monotonic()
        self._lock = threading.Lock()

    def take(self) -> float:
        """Take one token. Returns 0 when allowed, else the seconds until one is free."""
        with self._lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.refill)
            self.t = now
            if self.tokens >= 1:
                self.tokens -= 1
                return 0.0
            return (1 - self.tokens) / self.refill if self.refill > 0 else 60.0


def create_app(router: Any = None, cfg: Optional[dict] = None, api_key: Optional[str] = None) -> FastAPI:
    cfg = cfg or load_config()
    key = os.environ.get("QUANTLAB_API_KEY", "") if api_key is None else api_key
    if len(key) < 32:
        raise RuntimeError("QUANTLAB_API_KEY must be set to a key of at least 32 characters; refusing to start")
    api = cfg["api"]
    allow_force = bool(api["allow_force_quant"])
    max_chars = int(api["max_prompt_chars"])
    buckets: dict[str, TokenBucket] = {}
    f16_buckets: dict[str, TokenBucket] = {}
    log_path = Path(cfg["paths"]["results_dir"]) / "logs" / "service.jsonl"
    log_lock = threading.Lock()
    state = {"router": router}

    def get_router():
        if state["router"] is None:
            from .router import Router
            state["router"] = Router(cfg=cfg)
        return state["router"]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if state["router"] is not None:
            state["router"].shutdown()

    app = FastAPI(title="QuantRoute", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def require_key(x_api_key: Optional[str]) -> str:
        if not x_api_key or not hmac.compare_digest(x_api_key.encode("utf-8"), key.encode("utf-8")):
            raise HTTPException(status_code=401, detail="Missing or invalid API key (send it in the X-API-Key header).")
        return x_api_key

    def too_many(wait_s: float, what: str) -> JSONResponse:
        retry = max(1, math.ceil(wait_s))
        return JSONResponse(status_code=429, headers={"Retry-After": str(retry)},
                            content={"detail": f"{what} Try again in {retry} s."})

    def log(entry: dict) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_lock, open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/info")
    def info() -> dict:
        return {"allow_force_quant": allow_force, "max_prompt_chars": max_chars,
                "rate_per_min": api["rate_per_min"], "f16_rate_per_min": api["f16_rate_per_min"],
                "tolerances": ["strict", "balanced"]}

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return PAGE.read_text(encoding="utf-8")

    @app.get("/policy")
    def policy(x_api_key: Optional[str] = Header(default=None)) -> dict:
        require_key(x_api_key)
        return get_router().summary()

    @app.post("/route")
    def route(req: RouteRequest, x_api_key: Optional[str] = Header(default=None)):
        k = require_key(x_api_key)
        if len(req.prompt) > max_chars:
            raise HTTPException(status_code=413, detail=f"Prompt is longer than {max_chars} characters.")
        if req.force_quant and not allow_force:
            raise HTTPException(status_code=403, detail="force_quant is disabled on this server.")
        wait = buckets.setdefault(k, TokenBucket(api["rate_per_min"])).take()
        if wait:
            return too_many(wait, f"Rate limit of {api['rate_per_min']} requests per minute reached.")

        from .router import RouterError, resolve_rung
        r = get_router()
        try:
            d = r.decide(req.prompt, req.tolerance)
        except RouterError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        served = resolve_rung(d.rung, d.floor, d.flagged, req.force_quant)
        if served == "F16":
            wait = f16_buckets.setdefault(k, TokenBucket(api["f16_rate_per_min"])).take()
            if wait:
                return too_many(wait, f"F16 quota of {api['f16_rate_per_min']} requests per minute reached; "
                                      "the request is not downgraded to a cheaper version.")
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "prompt_sha256": hashlib.sha256(req.prompt.encode("utf-8")).hexdigest(),
                 "category": d.category, "pred_class": d.pred_class, "cell": d.cell,
                 "tolerance": req.tolerance, "flagged": d.flagged, "safety_score": d.safety_score,
                 "forced": req.force_quant, "rung": served}
        try:
            out = r.route(req.prompt, req.tolerance, req.force_quant, decision=d)
        except Exception as exc:   # model server failed to start or answer: report, never leak a trace
            log({**entry, "status": "error", "error": type(exc).__name__})
            raise HTTPException(status_code=503, detail=f"The model server could not answer ({type(exc).__name__}).")
        log({**entry, "status": "ok", "latency_s": out["latency_s"], "swap_occurred": out["swap_occurred"]})
        return out

    return app
