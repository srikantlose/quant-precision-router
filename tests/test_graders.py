import subprocess
import sys

import pytest

from quantlab.graders import grade


def _subprocess_executor(program: str) -> bool:
    """Fake executor for tests: runs the program on the host in a fresh
    subprocess, same behaviour the production sandbox (built in a later
    phase) must reproduce inside Docker. Used only in tests, never in the
    real grading path, which defaults to quantlab.sandbox.run_in_sandbox."""
    try:
        proc = subprocess.run([sys.executable, "-c", program],
                              capture_output=True, timeout=5)
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def test_numeric_tolerates_currency_units_and_unicode_minus():
    assert grade({"grader": "numeric", "expected": "42"}, "$42")
    assert grade({"grader": "numeric", "expected": "42"}, "42 kg")
    assert grade({"grader": "numeric", "expected": "-5"}, "−5")  # unicode minus
    assert not grade({"grader": "numeric", "expected": "42"}, "41")  # still wrong
    assert not grade({"grader": "numeric", "expected": "42"}, "$41")


def test_contains_all():
    task = {"grader": "contains_all", "expected": ["merkel", "paris"]}
    assert grade(task, "PERSON: Angela Merkel, LOCATION: Paris")
    assert not grade(task, "Only Merkel is mentioned here")


def test_word_limit():
    task = {"grader": "word_limit",
            "expected": {"max_words": 10, "any_of": ["wright", "flight"]}}
    assert grade(task, "The Wright brothers made the first powered flight.")
    assert not grade(task, "The Wright brothers made the very first powered "
                           "airplane flight near Kitty Hawk in December 1903.")
    assert not grade(task, "A short unrelated sentence about nothing much.")
    assert not grade(task, "")


def test_py_exec_passes_correct_code():
    task = {"grader": "py_exec",
            "test_code": "assert add(2, 3) == 5\nassert add(-1, 1) == 0"}
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    assert grade(task, good, executor=_subprocess_executor)
    assert grade(task, "def add(a, b):\n    return a + b",
                executor=_subprocess_executor)  # unfenced too


def test_py_exec_fails_buggy_and_hanging_code():
    task = {"grader": "py_exec", "test_code": "assert add(2, 3) == 5"}
    assert not grade(task, "def add(a, b):\n    return a - b",
                     executor=_subprocess_executor)
    hang = ("def add(a, b):\n    while True:\n        pass\n")
    assert not grade({"grader": "py_exec", "test_code": "assert add(1, 1) == 2"},
                     hang, executor=_subprocess_executor)  # timeout kills it


def test_py_exec_rejects_empty():
    # Empty code/test_code is rejected before the executor is ever
    # consulted, so no executor needs to be supplied here.
    assert not grade({"grader": "py_exec", "test_code": "assert True"}, "")


@pytest.mark.docker
def test_py_exec_default_executor_runs_in_docker_sandbox():
    # With no executor given, grade() must go through the real Docker
    # sandbox (quantlab.sandbox.run_in_sandbox), never the host.
    task = {"grader": "py_exec", "test_code": "assert add(2, 3) == 5"}
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "def add(a, b):\n    return a - b"
    assert grade(task, good) is True
    assert grade(task, bad) is False
