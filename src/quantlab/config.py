"""Config loading: config.yaml values, overridable per-key via environment variables.

Env overrides exist so any run can be re-tuned without editing the file. Naming:
QUANTLAB_<SECTION>_<KEY>, e.g. QUANTLAB_POLICY_DELTA=0.05, QUANTLAB_API_PORT=8001.
List/dict-valued settings (e.g. model.ladder, policy.delta_grid) are config-file
only -- there is no sane single-string env encoding for them.
Secrets (FIREWORKS_API_KEY, QUANTLAB_API_KEY) come ONLY from the environment,
never from the file and never written to one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "paths": {
        "cheaproute_root": "C:/Users/user/Desktop/PROJECTS 2026/AMD Developer Hack",
        "llama_dir": "tools/llama.cpp",
        "gguf_dir": "models/gguf",
        "lock_file": "models/models.lock.json",
        "tasks_all": "data/tasks/all.jsonl",
        "safety_tasks": "data/safety/safety.jsonl",
        "grid": "results/raw/grid.jsonl",
        "results_dir": "results",
    },
    "model": {
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "ladder": ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"],
        "reference": "F16",
    },
    "llama": {
        "build": "b11451",
        "host": "127.0.0.1",
        "port_base": 8101,  # rung index i -> port 8101 + i
        "ctx_size": 4096,
        "parallel": 1,  # -np 1
        "load_timeout_s": 120,
        "request_timeout_s": 300,
        "vram_margin_mb": 512,
    },
    "request": {
        "temperature": 0.0,
        "seed": 42,
        "top_logprobs": 1,
        "cache_prompt": False,
    },
    "bench": {
        "grid_order": ["Q4_K_M", "Q8_0", "Q2_K", "Q3_K_M", "Q5_K_M", "Q6_K", "F16"],
        "warmup_requests": 3,
        "retries": 2,
        "retry_backoff_s": 5,
        "max_error_rate": 0.02,
    },
    "sandbox": {
        "image": "python:3.13-slim",
        "memory": "256m",
        "cpus": 1,
        "pids_limit": 64,
        "timeout_s": 15,
    },
    "generation": {
        "teacher": "accounts/fireworks/models/gpt-oss-120b",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_cap": 400,
        "per_cell_target": 38,
        "batch_size": 10,
        "max_gen_calls_per_cell": 6,
        "verify_batch_size": 10,
        "temperature": 0.9,
        "max_tokens": 8000,
        "timeout_s": 120,
    },
    "split": {
        "test_percent": 30,
    },
    "labels": {
        "tau_quantile": 0.10,
        "tau_sensitivity": [0.05, 0.20],
        "max_exceptions_above": 1,
    },
    "predictor": {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "folds": 5,
        "C_grid": [0.1, 1.0, 10.0],
    },
    "stats": {
        "z_one_sided": 1.645,
        "bootstrap_refit_B": 200,
        "bootstrap_metric_B": 1000,
    },
    "policy": {
        "delta": 0.02,
        "phi": 0.10,
        "psi": 0.05,
        "delta_grid": [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15],
        "phi_grid": [0.05, 0.10, 0.15, 0.20, 0.30, 1.0],
        "psi_grid": [0.0, 0.02, 0.05, 0.10, 1.0],
    },
    "cost": {
        "gpu_hourly_rate": 1.0,  # 1.0 => cost reads as GPU-hours per 1,000 tasks
    },
    "safety": {
        "classifier_target_recall": 0.95,
        "audit_sample": 200,
        "audit_api_cap": 30,
        "audit_batch_size": 8,
    },
    "routing": {
        "normalize": False,
        "normalize_max_words": 256,
    },
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "rate_per_min": 30,
        "f16_rate_per_min": 5,
        "max_prompt_chars": 8000,
        "allow_force_quant": False,
    },
    "dashboard": {
        "port": 8501,
    },
}


def _coerce(value: str, like: Any) -> Any:
    """Cast an env-var string to the type of the default it overrides."""
    if isinstance(like, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(like, int):
        try:
            return int(value)
        except ValueError:
            return int(float(value))  # tolerate "30.0"-style overrides
    if isinstance(like, float):
        return float(value)
    return value


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Merge defaults <- config.yaml <- environment variables."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULTS.items()}

    candidates = [Path(path)] if path else [
        Path(os.environ.get("QUANTLAB_CONFIG", "")),
        Path("config.yaml"),
        Path(__file__).resolve().parents[2] / "config.yaml",
    ]
    for cand in candidates:
        if cand and str(cand) and cand.is_file():
            with open(cand, "r", encoding="utf-8") as f:
                file_cfg = yaml.safe_load(f) or {}
            for key, val in file_cfg.items():
                if isinstance(val, dict) and isinstance(cfg.get(key), dict):
                    cfg[key].update(val)
                else:
                    cfg[key] = val
            break

    # Environment overrides: flat keys, then section keys.
    for key, default in _DEFAULTS.items():
        if isinstance(default, dict):
            for sub, subdefault in default.items():
                if isinstance(subdefault, (list, dict)):
                    continue  # structured values: config file only
                env = os.environ.get(f"QUANTLAB_{key.upper()}_{sub.upper()}")
                if env is not None:
                    cfg[key][sub] = _coerce(env, subdefault)
        else:
            env = os.environ.get(f"QUANTLAB_{key.upper()}")
            if env is not None:
                cfg[key] = _coerce(env, default)

    return cfg
