"""llama-server process lifecycle and a VRAM-budgeted pool across rungs.

A llama-server instance is launched per rung, never shared; only one rung's
server runs at a time unless the pool's VRAM budget allows more. Every
launch re-verifies the model file's hash before starting (quants.verify) --
a server is never started against a file that might have been corrupted or
tampered with since it was last downloaded.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import psutil
import requests

from . import quants
from .client import OpenAICompatClient
from .config import load_config


class ServerStartError(Exception):
    pass


def _foreign_llama_server_pids(known_pids: frozenset[int]) -> list[int]:
    """PIDs of running llama-server.exe processes NOT in `known_pids` (i.e.
    not ones this process itself started and is tracking)."""
    foreign = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = proc.info.get("name") or ""
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if name.lower() == "llama-server.exe" and proc.info["pid"] not in known_pids:
            foreign.append(proc.info["pid"])
    return foreign


class LlamaServer:
    """One llama-server process serving one quantization rung."""

    def __init__(self, rung: str, ngl: int):
        self.rung = rung
        self.ngl = ngl
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.load_s: Optional[float] = None
        self._log_path: Optional[Path] = None

    def _last_log_lines(self, n: int = 40) -> str:
        if self._log_path is None or not self._log_path.is_file():
            return "(no log available)"
        lines = self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])

    def start(self, known_pids: frozenset[int] = frozenset()) -> None:
        foreign = _foreign_llama_server_pids(known_pids)
        if foreign:
            raise ServerStartError(
                f"found unexpected llama-server.exe process(es) already running "
                f"(PIDs {foreign}) -- stop them manually before continuing"
            )

        if not quants.verify(self.rung):
            raise ServerStartError(
                f"model verification failed for rung {self.rung!r} -- refusing to start"
            )

        cfg = load_config()
        self.port = quants.port_for(self.rung)
        llama_dir = Path(cfg["paths"]["llama_dir"])
        exe = llama_dir / "llama-server.exe"
        cmd = [
            str(exe),
            "-m", str(quants.model_path(self.rung)),
            "--host", cfg["llama"]["host"],
            "--port", str(self.port),
            "-c", str(cfg["llama"]["ctx_size"]),
            "-np", str(cfg["llama"]["parallel"]),
            "-ngl", str(self.ngl),
            "--seed", str(cfg["request"]["seed"]),
            "--jinja",
        ]

        log_dir = Path(cfg["paths"]["results_dir"]) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self._log_path = log_dir / f"llama-{self.rung}-{timestamp}.log"
        log_file = open(self._log_path, "w", encoding="utf-8", errors="replace")

        t0 = time.time()
        self.process = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT,
        )

        health_url = f"http://{cfg['llama']['host']}:{self.port}/health"
        load_timeout_s = cfg["llama"]["load_timeout_s"]
        while time.time() - t0 < load_timeout_s:
            if self.process.poll() is not None:
                log_file.close()
                raise ServerStartError(
                    f"llama-server for rung {self.rung!r} exited early "
                    f"(code {self.process.returncode}) -- last log lines:\n"
                    f"{self._last_log_lines()}"
                )
            try:
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200:
                    self.load_s = time.time() - t0
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)

        # Timed out without a healthy response.
        self.stop()
        log_file.close()
        raise ServerStartError(
            f"llama-server for rung {self.rung!r} did not become healthy within "
            f"{load_timeout_s}s -- last log lines:\n{self._last_log_lines()}"
        )

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None

    def client(self) -> OpenAICompatClient:
        cfg = load_config()
        return OpenAICompatClient(
            base_url=f"http://{cfg['llama']['host']}:{self.port}/v1",
            model=self.rung,
            timeout_s=cfg["llama"]["request_timeout_s"],
            max_tokens=cfg["request"].get("max_tokens", 512),
            top_logprobs=cfg["request"]["top_logprobs"],
            cache_prompt=cfg["request"]["cache_prompt"],
        )


def find_ngl(rung: str, n_layers: int) -> int:
    """Largest -ngl that successfully starts, searching downward from
    n_layers+1 (an intentional over-estimate meaning "offload everything")
    in steps of 2 to 0. The first ngl that reaches /health wins; a caller
    can tell full residency apart from partial offload via
    `ngl >= n_layers + 1`."""
    ngl = n_layers + 1
    last_error: Optional[Exception] = None
    while ngl >= 0:
        server = LlamaServer(rung, ngl)
        try:
            server.start()
            server.stop()
            return ngl
        except ServerStartError as exc:
            last_error = exc
            ngl -= 2
    raise ServerStartError(
        f"no working -ngl found for rung {rung!r} down to ngl=0; last error: {last_error}"
    )


class ServerPool:
    """VRAM-budgeted LRU pool of LlamaServer instances across rungs.
    Single-threaded; get() is the only entry point and is lock-guarded."""

    def __init__(self, budget_mb: float, vram_mb: dict[str, float], ngl: dict[str, int]):
        self.budget_mb = budget_mb
        self.vram_mb = vram_mb
        self.ngl = ngl
        self._servers: dict[str, LlamaServer] = {}
        self._lru: list[str] = []  # most-recently-used at the end
        self._lock = threading.Lock()

    def _known_pids(self) -> frozenset[int]:
        return frozenset(
            s.process.pid for s in self._servers.values()
            if s.process is not None and s.process.poll() is None
        )

    def _resident_vram(self, exclude: Optional[str] = None) -> float:
        return sum(self.vram_mb.get(r, 0.0) for r in self._servers if r != exclude)

    def get(self, rung: str) -> tuple[OpenAICompatClient, bool, float]:
        with self._lock:
            if rung in self._servers:
                self._lru.remove(rung)
                self._lru.append(rung)
                return self._servers[rung].client(), False, 0.0

            needed = self.vram_mb.get(rung, 0.0)
            while self._lru and self._resident_vram(exclude=rung) + needed > self.budget_mb:
                lru_rung = self._lru.pop(0)
                self._servers.pop(lru_rung).stop()

            t0 = time.time()
            server = LlamaServer(rung, self.ngl[rung])
            server.start(known_pids=self._known_pids())
            swap_s = time.time() - t0

            self._servers[rung] = server
            self._lru.append(rung)
            return server.client(), True, swap_s

    def resident(self) -> list[str]:
        """Rungs currently loaded, least recently used first."""
        with self._lock:
            return list(self._lru)

    def shutdown(self) -> None:
        with self._lock:
            for server in self._servers.values():
                server.stop()
            self._servers.clear()
            self._lru.clear()
