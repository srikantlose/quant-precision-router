"""Model download, verification and path resolution for the quantization
ladder.

Every GGUF file comes from one pinned commit of the official
Qwen/Qwen2.5-3B-Instruct-GGUF repo -- never mixed with another repo's
quantization recipe (bartowski's Q4_K_M, for example, is a different build
at a different size; see docs/literature-survey.md). The commit SHA is
resolved against the hub once and then pinned into models/models.lock.json;
every later call reads it back from the lock file rather than re-resolving
"whatever HEAD happens to be" against the hub. Likewise, each file's
SHA-256 is recorded in the lock file at download time, and verify() checks
local files against those pinned hashes -- never against a freshly-queried
remote hash, so a compromised or edited upstream file after this project's
own download would be caught, not silently trusted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import gguf
from huggingface_hub import HfApi, hf_hub_download

from .config import load_config

LADDER = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]

FILES: dict[str, list[str]] = {
    "Q2_K": ["qwen2.5-3b-instruct-q2_k.gguf"],
    "Q3_K_M": ["qwen2.5-3b-instruct-q3_k_m.gguf"],
    "Q4_K_M": ["qwen2.5-3b-instruct-q4_k_m.gguf"],
    "Q5_K_M": ["qwen2.5-3b-instruct-q5_k_m.gguf"],
    "Q6_K": ["qwen2.5-3b-instruct-q6_k.gguf"],
    "Q8_0": ["qwen2.5-3b-instruct-q8_0.gguf"],
    "F16": ["qwen2.5-3b-instruct-fp16-00001-of-00002.gguf",
            "qwen2.5-3b-instruct-fp16-00002-of-00002.gguf"],
}


class QuantsError(Exception):
    pass


def port_for(rung: str) -> int:
    cfg = load_config()
    return cfg["llama"]["port_base"] + LADDER.index(rung)


def _lock_path() -> Path:
    return Path(load_config()["paths"]["lock_file"])


def _gguf_dir() -> Path:
    return Path(load_config()["paths"]["gguf_dir"])


def _load_lock() -> dict[str, Any]:
    path = _lock_path()
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_lock(lock: dict[str, Any]) -> None:
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lock, f, ensure_ascii=False, indent=2)


def resolve_revision() -> str:
    """Return the pinned commit SHA. Resolved against the hub and written
    into the lock file only the first time; every later call reads it back
    from the lock file instead of re-resolving against the hub."""
    lock = _load_lock()
    if "revision" in lock:
        return lock["revision"]

    cfg = load_config()
    repo = cfg["model"]["repo"]
    info = HfApi().model_info(repo)
    rev = info.sha
    if not rev or len(rev) != 40:
        raise QuantsError(f"unexpected revision sha from hub: {rev!r}")

    lock["repo"] = repo
    lock["revision"] = rev
    lock.setdefault("files", {})
    sandbox_digest = cfg["sandbox"].get("image_digest")
    if sandbox_digest:
        lock["sandbox_image_digest"] = sandbox_digest
    _save_lock(lock)
    return rev


def remote_sha256(rung: str) -> dict[str, dict[str, Any]]:
    """{filename: {"size": int, "sha256": str}} for every file of `rung`,
    read from the hub's LFS metadata at the pinned revision."""
    cfg = load_config()
    repo = cfg["model"]["repo"]
    rev = resolve_revision()
    info = HfApi().model_info(repo, revision=rev, files_metadata=True)
    by_name = {s.rfilename: s for s in info.siblings}

    out: dict[str, dict[str, Any]] = {}
    for fname in FILES[rung]:
        sib = by_name.get(fname)
        if sib is None:
            raise QuantsError(f"file {fname!r} not found in {repo}@{rev}")
        if sib.lfs is None or not getattr(sib.lfs, "sha256", None):
            raise QuantsError(f"file {fname!r} has no LFS sha256 metadata")
        out[fname] = {"size": sib.size, "sha256": sib.lfs.sha256}
    return out


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download(rungs: list[str]) -> None:
    """Download every file of each rung, verify its SHA-256 against the
    hub's LFS metadata, and record name/size/sha256 in the lock file. A
    hash mismatch deletes the bad file and raises -- it is never left on
    disk half-trusted."""
    cfg = load_config()
    repo = cfg["model"]["repo"]
    rev = resolve_revision()
    gguf_dir = _gguf_dir()
    gguf_dir.mkdir(parents=True, exist_ok=True)
    lock = _load_lock()
    lock.setdefault("files", {})

    for rung in rungs:
        remote = remote_sha256(rung)
        entries = []
        for fname, meta in remote.items():
            local_path = Path(hf_hub_download(
                repo_id=repo, filename=fname, revision=rev,
                local_dir=str(gguf_dir),
            ))
            local_sha = _sha256_file(local_path)
            if local_sha != meta["sha256"]:
                local_path.unlink(missing_ok=True)
                raise QuantsError(
                    f"hash mismatch for {fname}: expected {meta['sha256']}, "
                    f"got {local_sha} -- file deleted"
                )
            entries.append({"name": fname, "size": meta["size"], "sha256": local_sha})
        lock["files"][rung] = entries
        _save_lock(lock)


def model_path(rung: str) -> Path:
    """Path to the first file of `rung` -- the F16 shard 00001 for F16,
    the single file for every other rung. This is what llama-server's -m
    is pointed at."""
    return _gguf_dir() / FILES[rung][0]


def read_n_layers(rung: str) -> int:
    """Read the architecture's layer count (qwen2.block_count) straight out
    of a downloaded GGUF file's metadata. This is identical across every
    quantization level of the same base model, so any already-downloaded
    rung can answer it -- used by find_ngl() as an ngl search starting
    point before P9's profiling pass has written a trusted n_layers into
    the lock file."""
    reader = gguf.GGUFReader(str(model_path(rung)))
    field = reader.fields.get("qwen2.block_count")
    if field is None:
        raise QuantsError(
            f"qwen2.block_count not found in {rung} metadata -- unexpected architecture"
        )
    return int(field.parts[field.data[0]][0])


def _verified_cache_path() -> Path:
    return _gguf_dir() / ".verified.json"


def verify(rung: str, full: bool = False) -> bool:
    """True iff every file of `rung` matches its pinned SHA-256. Without
    `full`, trusts a cached verification keyed by (size, mtime) and only
    re-hashes a file when either has changed since it was last verified;
    `full=True` always re-hashes."""
    lock = _load_lock()
    entries = lock.get("files", {}).get(rung)
    if not entries:
        raise QuantsError(f"rung {rung!r} is not in the lock file -- download it first")

    cache_path = _verified_cache_path()
    cache: dict[str, Any] = {}
    if cache_path.is_file():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    gguf_dir = _gguf_dir()
    all_ok = True
    for entry in entries:
        path = gguf_dir / entry["name"]
        if not path.is_file():
            all_ok = False
            continue
        stat = path.stat()
        cached = cache.get(entry["name"])
        needs_hash = (full or cached is None
                     or cached.get("size") != stat.st_size
                     or cached.get("mtime") != stat.st_mtime)
        if needs_hash:
            actual = _sha256_file(path)
            ok = actual == entry["sha256"]
            cache[entry["name"]] = {"size": stat.st_size, "mtime": stat.st_mtime,
                                    "sha256_ok": ok}
        else:
            ok = bool(cached.get("sha256_ok", False))
        all_ok = all_ok and ok

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    return all_ok
