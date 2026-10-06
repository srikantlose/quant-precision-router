"""The single place that turns a prompt into a request and a response into an
answer. Used by BOTH bench.py (the offline benchmark grid) and router.py (the
live routing service), so that a task is graded under the exact same
category/prompting/token-budget/answer-extraction logic it would receive in
production. Two call sites doing this independently is how train/serve skew
creeps in -- this module exists so that can't happen.
"""

from __future__ import annotations

from .confidence import extract_final
from .tasktype import classify, extract_answer, profile_for, to_category


def prepare(prompt: str) -> tuple[str, str, str, int]:
    label = classify(prompt)                 # classifier label
    category = to_category(label)            # 12-category name
    prof = profile_for(label)
    return category, label, prof.local_style, prof.local_max_tokens


def extract(label: str, raw_text: str) -> str:
    return extract_answer(label, raw_text, extract_final) or (raw_text or "").strip()
