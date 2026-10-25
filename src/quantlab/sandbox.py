"""Sandboxed execution of model-generated code, used by graders.grade() for
py_exec tasks. Every py_exec answer runs here, inside a locked-down,
disposable Docker container -- never on the host, never with network access,
and always under hard memory/CPU/process-count/time limits. If the Docker
daemon is unreachable this raises SandboxUnavailable rather than silently
running the code anywhere else.
"""

from __future__ import annotations

import subprocess
from typing import Optional
from uuid import uuid4

from .config import load_config

_cfg: Optional[dict] = None


def _sandbox_cfg() -> dict:
    global _cfg
    if _cfg is None:
        _cfg = load_config()["sandbox"]
    return _cfg


class SandboxUnavailable(Exception):
    """Raised when the Docker daemon cannot be reached. Never caught by
    run_in_sandbox itself -- callers must not interpret it as "code failed"."""


def _check_daemon() -> None:
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        raise SandboxUnavailable(f"docker daemon unreachable: {exc}") from exc
    if proc.returncode != 0:
        raise SandboxUnavailable(
            "docker daemon unreachable: "
            + proc.stderr.decode("utf-8", errors="replace")[:300]
        )


def run_in_sandbox(program: str) -> bool:
    """Run `program` as `python -I -` inside a disposable container and
    return True iff it exits 0. Never raises for a failing/hanging/
    resource-violating program -- those all come back as False. Only raises
    SandboxUnavailable when the sandbox itself (the Docker daemon) cannot be
    reached at all."""
    _check_daemon()
    cfg = _sandbox_cfg()
    image = cfg["image_digest"] or cfg["image"]
    name = f"qlab-exec-{uuid4().hex[:12]}"
    cmd = [
        "docker", "run", "--rm", "-i", "--name", name,
        "--network", "none",
        "--memory", cfg["memory"], "--memory-swap", cfg["memory"],
        "--cpus", str(cfg["cpus"]), "--pids-limit", str(cfg["pids_limit"]),
        "--read-only", "--tmpfs", "/tmp:rw,size=16m",
        "--user", "65534:65534", "--security-opt", "no-new-privileges",
        image, "python", "-I", "-",
    ]
    try:
        proc = subprocess.run(
            cmd, input=program.encode("utf-8"),
            capture_output=True, timeout=cfg["timeout_s"],
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return False
