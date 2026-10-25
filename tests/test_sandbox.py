"""Tests for the Docker sandbox. All require a running Docker daemon, so
they're marked @pytest.mark.docker and excluded from the default `pytest -q`
run (see pyproject.toml addopts). Run with: pytest -q -m docker
"""

import subprocess
import time

import pytest

from quantlab.sandbox import run_in_sandbox

pytestmark = pytest.mark.docker


def test_correct_code_passes():
    program = "def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\n"
    assert run_in_sandbox(program) is True


def test_wrong_code_fails():
    program = "def add(a, b):\n    return a - b\n\nassert add(2, 3) == 5\n"
    assert run_in_sandbox(program) is False


def test_no_network_access():
    program = ("import urllib.request\n"
               "urllib.request.urlopen('http://example.com', timeout=3)\n")
    assert run_in_sandbox(program) is False


def test_memory_limit_enforced():
    program = "b = bytearray(1024**3)\n"  # 1 GB, well over the 256m cap
    assert run_in_sandbox(program) is False


def test_readonly_filesystem():
    program = "open('/etc/x', 'w').write('x')\n"
    assert run_in_sandbox(program) is False


def test_hang_is_killed_within_timeout_and_container_is_removed():
    program = "while True:\n    pass\n"
    t0 = time.time()
    result = run_in_sandbox(program)
    elapsed = time.time() - t0
    assert result is False
    # cfg.sandbox.timeout_s is 15s; this must not take materially longer.
    assert elapsed < 15 + 10, elapsed

    ps = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=qlab-exec", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=10,
    )
    assert ps.stdout.strip() == "", f"leftover container(s): {ps.stdout!r}"
