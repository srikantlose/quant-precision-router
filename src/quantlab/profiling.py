"""Per-rung profiling: VRAM footprint, residency, and load time for every
rung on the quantization ladder, measured on THIS machine at THIS moment --
never assumed from file size or nominal bit-width labels. This is the
measurement that gives the VRAM-residency cliff (the point where a rung
stops fitting and must partially offload to CPU) its empirical teeth: it is
measured here, not asserted.

Must run with no llama-server already up (the VRAM baseline would otherwise
include another rung's footprint).
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

from . import quants
from .config import load_config
from .server import LlamaServer, ServerStartError, find_ngl

LADDER = quants.LADDER


class ProfilingError(Exception):
    pass


def _assert_no_llama_server_running() -> None:
    for proc in psutil.process_iter(["name"]):
        try:
            name = proc.info.get("name") or ""
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if name.lower() == "llama-server.exe":
            raise ProfilingError(
                "a llama-server.exe process is already running -- stop it before profiling"
            )


def _gpu_memory_mb() -> tuple[int, int]:
    """(used_mb, free_mb) from nvidia-smi."""
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        raise ProfilingError(f"nvidia-smi failed: {proc.stderr}")
    used_str, free_str = proc.stdout.strip().split(",")
    return int(used_str.strip()), int(free_str.strip())


def _wait_for_vram_settle(baseline_used_mb: int, timeout_s: float = 30.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used, _ = _gpu_memory_mb()
        if used <= baseline_used_mb + 100:
            return
        time.sleep(1)
    # Not fatal -- the next rung's measurement will simply include whatever
    # residue is left, which is itself informative; log rather than raise.


def _file_bytes(rung: str) -> int:
    lock = quants._load_lock()
    entries = lock.get("files", {}).get(rung, [])
    return sum(e["size"] for e in entries)


def profile_rung(rung: str, n_layers: int, baseline_used_mb: int, n_runs: int = 3) -> dict[str, Any]:
    """`baseline_used_mb` is used only to decide when VRAM has "settled"
    between runs -- it is NOT used as the subtrahend for the vram_mb delta.
    Other desktop GPU usage (browser tabs, compositor, etc.) drifts over
    the many minutes a full profiling pass takes, so a single baseline
    captured once at the start would contaminate every later rung's
    reading. Instead, each individual run takes its OWN fresh "before"
    measurement immediately before that run's server starts, and the delta
    is computed against THAT -- a few hundred milliseconds of drift risk
    instead of several minutes'."""
    ngl = find_ngl(rung, n_layers)
    fully_resident = ngl >= n_layers + 1

    load_s_runs = []
    vram_mb_runs = []
    cfg = load_config()
    warmup_requests = cfg["bench"]["warmup_requests"]

    for _ in range(n_runs):
        used_before, _ = _gpu_memory_mb()
        server = LlamaServer(rung, ngl)
        server.start()
        try:
            load_s_runs.append(server.load_s)
            client = server.client()
            for _ in range(warmup_requests):
                try:
                    client.generate("Warm-up request.", system_prompt="Reply OK.", max_tokens=4)
                except Exception:
                    pass
            used_after, _ = _gpu_memory_mb()
            vram_mb_runs.append(max(0, used_after - used_before))
        finally:
            server.stop()
        _wait_for_vram_settle(baseline_used_mb)

    file_bytes = _file_bytes(rung)
    n_params = quants._load_lock()["n_params"]
    bpp = file_bytes / n_params
    bpw = 8 * bpp

    return {
        "rung": rung,
        "files": quants.FILES[rung],
        "file_bytes": file_bytes,
        "bpp": bpp,
        "bpw": bpw,
        "ngl": ngl,
        "n_layers": n_layers,
        "fully_resident": fully_resident,
        "load_s_runs": load_s_runs,
        "swap_cost_s": statistics.median(load_s_runs),
        "vram_mb": max(vram_mb_runs),
    }


def profile_all(out_path: str | Path = "results/profiles.json") -> dict[str, Any]:
    _assert_no_llama_server_running()

    lock = quants._load_lock()
    n_layers = lock.get("n_layers")
    if n_layers is None:
        raise ProfilingError("n_layers not in the lock file -- run P9's n_params/n_layers step first")

    cfg = load_config()
    vram_margin_mb = cfg["llama"]["vram_margin_mb"]
    baseline_used_mb, baseline_free_mb = _gpu_memory_mb()
    budget_mb = baseline_free_mb - vram_margin_mb

    result: dict[str, Any] = {
        "budget_mb": budget_mb, "baseline_used_mb": baseline_used_mb,
        "baseline_free_mb": baseline_free_mb, "vram_margin_mb": vram_margin_mb,
        "rungs": {},
    }
    for rung in LADDER:
        print(f"profiling {rung}...")
        record = profile_rung(rung, n_layers, baseline_used_mb)
        result["rungs"][rung] = record
        print(f"  ngl={record['ngl']} fully_resident={record['fully_resident']} "
             f"vram_mb={record['vram_mb']} bpw={record['bpw']:.2f} "
             f"swap_cost_s={record['swap_cost_s']:.2f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
