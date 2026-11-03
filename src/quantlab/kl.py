"""Corpus-level KL-divergence of each rung against the F16 reference, measured
with llama.cpp's own --kl-divergence machinery on WikiText-2 -- never assumed
from bits-per-weight alone. The F16 pass writes per-token logits to disk; every
other rung is then compared against that saved file.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import quants
from .config import load_config

LADDER = quants.LADDER
NON_REFERENCE = [r for r in LADDER if r != "F16"]


class KLError(Exception):
    pass


def _free_gb(path: str = ".") -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def _llama_exe(name: str) -> Path:
    cfg = load_config()
    exe = Path(cfg["paths"]["llama_dir"]) / f"{name}.exe"
    if not exe.is_file():
        raise KLError(f"{exe} not found -- run setup-llama first")
    return exe


def _corpus_path() -> Path:
    return Path("data/kl/wikitext-2-raw/wiki.test.raw")


def _ngl_for(rung: str) -> int:
    profiles_path = Path("results/profiles.json")
    if profiles_path.is_file():
        with open(profiles_path, encoding="utf-8") as f:
            profiles = json.load(f)
        rec = profiles.get("rungs", {}).get(rung)
        if rec is not None:
            return int(rec["ngl"])
    lock = quants._load_lock()
    n_layers = lock.get("n_layers")
    if n_layers is None:
        raise KLError("n_layers not in lock file and no profiles.json -- run P9 first")
    return n_layers + 1


_KLD_RE = re.compile(
    r"Mean\s+KLD\s*:\s*([-\d.]+)\s*[^\d.\-]+\s*([-\d.]+)", re.IGNORECASE
)
_SAME_TOP_RE = re.compile(
    r"Same top p\s*:\s*([\d.]+)", re.IGNORECASE
)


def _parse_kld(stdout: str) -> dict[str, Any]:
    """Pull mean KL-divergence, its uncertainty, and the same-top-token
    percentage out of llama-perplexity's --kl-divergence stdout. The exact
    section header varies slightly across builds, so this scans the whole
    "KL divergence" block rather than anchoring to one fixed line format."""
    kld_matches = _KLD_RE.findall(stdout)
    same_top_matches = _SAME_TOP_RE.findall(stdout)
    if not kld_matches:
        raise KLError(
            "could not parse mean KL-divergence from llama-perplexity output -- "
            "output format may have changed; see the saved log"
        )
    mean_kld, kld_uncertainty = kld_matches[0]
    result: dict[str, Any] = {
        "mean_kld": float(mean_kld),
        "kld_uncertainty": float(kld_uncertainty),
    }
    if same_top_matches:
        result["same_top_token_pct"] = float(same_top_matches[0])
    else:
        result["same_top_token_pct"] = None
    return result


def run_kl(out_path: str | Path = "results/kl.json") -> dict[str, Any]:
    corpus = _corpus_path()
    if not corpus.is_file():
        raise KLError(f"{corpus} not found -- download WikiText-2 first")

    cfg = load_config()
    ctx_size = cfg["llama"]["ctx_size"]
    chunks = 40

    kl_dir = Path("results/kl")
    kl_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("results/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    base_logits = kl_dir / "f16_logits.bin"

    perplexity_exe = _llama_exe("llama-perplexity")
    f16_path = quants.model_path("F16")
    f16_ngl = _ngl_for("F16")

    if base_logits.is_file() and base_logits.stat().st_size > 0:
        base_size = base_logits.stat().st_size
        print(f"reusing existing base logits file: {base_size} bytes")
    else:
        if _free_gb() < 20:
            raise KLError(f"free disk is below 20 GB ({_free_gb():.1f} GB) -- "
                          f"aborting before writing the base logits file")
        base_cmd = [
            str(perplexity_exe), "-m", str(f16_path), "-f", str(corpus),
            "-c", str(ctx_size), "--chunks", str(chunks), "-ngl", str(f16_ngl),
            "--kl-divergence-base", str(base_logits),
        ]
        print("running F16 base pass:", " ".join(base_cmd))
        base_proc = subprocess.run(base_cmd, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=1800)
        (logs_dir / "kl-F16-base.txt").write_text(
            base_proc.stdout + "\n" + base_proc.stderr, encoding="utf-8"
        )
        if base_proc.returncode != 0 or not base_logits.is_file():
            raise KLError(
                f"F16 base pass failed (exit {base_proc.returncode}); "
                f"see results/logs/kl-F16-base.txt"
            )
        base_size = base_logits.stat().st_size
        print(f"base logits file: {base_size} bytes")

    result: dict[str, Any] = {"corpus": str(corpus), "ctx_size": ctx_size,
                               "chunks": chunks, "base_logits_bytes": base_size,
                               "rungs": {}}

    for rung in NON_REFERENCE:
        rung_path = quants.model_path(rung)
        ngl = _ngl_for(rung)
        cmd = [
            str(perplexity_exe), "-m", str(rung_path), "-f", str(corpus),
            "-c", str(ctx_size), "--chunks", str(chunks), "-ngl", str(ngl),
            "--kl-divergence-base", str(base_logits), "--kl-divergence",
        ]
        print(f"running KL pass for {rung}:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=1800)
        log_text = proc.stdout + "\n" + proc.stderr
        (logs_dir / f"kl-{rung}.txt").write_text(log_text, encoding="utf-8")
        if proc.returncode != 0:
            raise KLError(f"KL pass for {rung} failed (exit {proc.returncode}); "
                          f"see results/logs/kl-{rung}.txt")
        parsed = _parse_kld(log_text)
        result["rungs"][rung] = parsed
        print(f"  {rung}: mean_kld={parsed['mean_kld']} "
             f"same_top_token_pct={parsed['same_top_token_pct']}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    base_logits.unlink(missing_ok=True)
    return result
