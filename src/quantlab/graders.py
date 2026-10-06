"""Graders for QuantLab benchmark tasks. These string/execution graders are
cheap, deterministic proxies that let the benchmark grid and the analysis
pipeline measure accuracy per (rung, task) without an LLM judge.

Grader types (per task, in data/tasks/*.jsonl):
  exact        - normalized string equality with `expected`
  numeric      - parse both as numbers, small tolerance
  contains     - answer contains ANY of `expected` (string or list), normalized
  contains_all - answer contains ALL of `expected` (list) — NER/multi-entity
  mc           - multiple choice: right letter OR the option text
  word_limit   - `expected` = {"max_words": N, "any_of": [...]}: within length
                 AND mentions at least one keyword — summarisation proxy
  py_exec      - run the answer's Python code + the task's `test_code` asserts
                 via an injectable executor; pass = exit 0 — code gen/debug
                 tasks. The default executor runs the program inside the
                 Docker sandbox (quantlab.sandbox.run_in_sandbox, built in a
                 later phase) — model-generated code is never run on the
                 host. Tests pass an explicit `executor=` to avoid depending
                 on Docker.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from quantlab.confidence import answers_agree, normalize_answer
from quantlab.tasktype import _FENCE_RE

_LETTER_RE = re.compile(r"\b([a-d])\b")
_LOOSE_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_CURRENCY_RE = re.compile(r"[$€£]")

Executor = Callable[[str], bool]


def _loose_number(s: str) -> float | None:
    """Best-effort number extraction for grading only: strips currency
    symbols and unit words, normalizes a unicode minus, and pulls the first
    number out of noisy answer text like "$42" or "42 kg". This is
    deliberately more permissive than confidence.answers_agree (which also
    drives production routing agreement scores and must stay strict)."""
    s = normalize_answer(s).replace("−", "-")
    s = _CURRENCY_RE.sub("", s)
    m = _LOOSE_NUM_RE.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


def grade(task: dict, answer: str, executor: Optional[Executor] = None) -> bool:
    kind = task.get("grader", "exact")
    expected = task.get("expected", "")
    ans_norm = normalize_answer(answer or "")

    if kind == "numeric":
        if answers_agree(str(expected), answer or ""):
            return True
        # answers_agree requires the WHOLE normalized string to be a bare
        # number; fall back to a loose extraction for grading purposes only
        # (a correct "$42" or "42 kg" shouldn't read as a wrong answer).
        exp_n, got_n = _loose_number(str(expected)), _loose_number(answer or "")
        if exp_n is None or got_n is None:
            return False
        return abs(exp_n - got_n) <= 1e-6 * max(1.0, abs(exp_n), abs(got_n))

    if kind == "contains":
        options = expected if isinstance(expected, list) else [expected]
        return any(normalize_answer(str(o)) in ans_norm for o in options)

    if kind == "contains_all":
        items = expected if isinstance(expected, list) else [expected]
        return all(normalize_answer(str(o)) in ans_norm for o in items)

    if kind == "word_limit":
        spec = expected if isinstance(expected, dict) else {}
        words = len((answer or "").split())
        if words == 0 or words > int(spec.get("max_words", 10**6)):
            return False
        any_of = spec.get("any_of") or []
        return (not any_of) or any(normalize_answer(str(k)) in ans_norm
                                   for k in any_of)

    if kind == "py_exec":
        return _run_python(answer or "", task.get("test_code", ""), executor)

    if kind == "mc":
        letter = normalize_answer(str(expected))
        text = normalize_answer(str(task.get("expected_text", "")))
        m = _LETTER_RE.search(ans_norm)
        if m and m.group(1) == letter:
            return True
        return bool(text) and text in ans_norm

    # exact
    return answers_agree(str(expected), answer or "")


def _run_python(answer: str, test_code: str, executor: Optional[Executor] = None) -> bool:
    """Assemble the answer's code with the task's test asserts and run the
    result via `executor` (a callable taking the program text and returning
    True iff it exited 0 — timeouts and other failures are the executor's
    responsibility to turn into False, never an uncaught exception).

    When `executor` is omitted, defaults to quantlab.sandbox.run_in_sandbox,
    so that grading code never runs in-process unless a caller explicitly
    opts into a different executor (as tests do)."""
    blocks = _FENCE_RE.findall(answer)
    code = (blocks[-1] if blocks else answer).strip()
    if not code or not test_code:
        return False
    program = code + "\n\n" + test_code + "\n"
    if executor is None:
        from quantlab.sandbox import run_in_sandbox
        executor = run_in_sandbox
    return executor(program)
