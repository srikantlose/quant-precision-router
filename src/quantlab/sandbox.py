"""Sandboxed execution of model-generated code, used by graders.grade() for
py_exec tasks.

Stub for now: the real implementation runs the program inside a locked-down
Docker container (no network, memory/CPU/pids limits, read-only filesystem,
a timeout) and never on the host. Until that is built, this refuses to run
anything rather than silently falling back to the host.
"""

from __future__ import annotations


def run_in_sandbox(program: str) -> bool:
    raise NotImplementedError("sandbox not built yet")
