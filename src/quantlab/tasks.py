"""Consolidate the CheapRoute seed task set into QuantLab's benchmark schema.

Reads CheapRoute's four `eval/tasks/*.jsonl` files (practice.jsonl, holdout.jsonl,
practice19.jsonl, mycustom.jsonl), which overlap: 19 ids are duplicated verbatim
across files. Deduplicating keeps the first occurrence (in the file order above),
and the result is 135 unique tasks -- the number this module's CLI asserts on.

CheapRoute's `type` field (already spelled as one of QuantLab's 12 categories,
e.g. "factual_qa", "multiple_choice" -- not a classify()-style label like "mc"
or "short_qa") becomes `category`; `source="seed"`; `split` is assigned
deterministically by split_for(), which is also used by later phases' generated
tasks and safety-slice tasks so every task lands in one stable train/test split
regardless of when it was added to the set. `tier`/`tier_source`/`tier_heuristic`
are left null here and filled in once the generated task set exists.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .tasktype import CATEGORIES

SEED_FILES = ["practice.jsonl", "holdout.jsonl", "practice19.jsonl", "mycustom.jsonl"]


class SeedConsolidationError(Exception):
    pass


def split_for(task_id: str, test_percent: int = 30) -> str:
    """Deterministic train/test assignment by hash of the task id -- not a
    random draw, so existing assignments never move as the task set grows and
    new tasks (generated or safety-slice) slot in on their own id."""
    h = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16)
    return "test" if h % 100 < test_percent else "train"


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _grader_ok(row: dict) -> bool:
    grader = row.get("grader", "exact")
    expected = row.get("expected")
    if grader in ("numeric", "exact", "contains"):
        return bool(expected)
    if grader == "contains_all":
        return isinstance(expected, list) and len(expected) > 0
    if grader == "mc":
        # expected_text is an optional fallback in graders.grade() (it treats
        # a missing one as ""); three holdout tasks (mc-h1/h2/h3) rely on
        # letter-only matching and have no expected_text at all, so it must
        # not be required here.
        return bool(expected)
    if grader == "word_limit":
        return isinstance(expected, dict) and "max_words" in expected
    if grader == "py_exec":
        return bool(row.get("test_code"))
    raise SeedConsolidationError(f"task {row.get('id')!r} has unknown grader {grader!r}")


def load_seed_files(cheaproute_root: str | Path) -> list[dict]:
    root = Path(cheaproute_root) / "eval" / "tasks"
    rows: list[dict] = []
    for fname in SEED_FILES:
        path = root / fname
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def consolidate(raw_rows: list[dict]) -> list[dict]:
    """Dedupe by id (keep first occurrence; a duplicate id with different task
    text raises), reject any two different ids sharing normalized text, rename
    type->category with validation, validate the grader's required fields, and
    assign source/split. Returns rows sorted by id."""
    by_id: dict[str, dict] = {}
    seen_norm: dict[str, str] = {}

    for row in raw_rows:
        task_id = row["id"]
        text = row["task"]
        if task_id in by_id:
            if by_id[task_id]["task"] != text:
                raise SeedConsolidationError(
                    f"duplicate id {task_id!r} has different task text:\n"
                    f"  kept: {by_id[task_id]['task']!r}\n"
                    f"  dupe: {text!r}"
                )
            continue  # keep first occurrence, ignore the identical repeat

        norm = _normalize_text(text)
        if norm in seen_norm and seen_norm[norm] != task_id:
            raise SeedConsolidationError(
                f"task {task_id!r} has the same normalized text as "
                f"{seen_norm[norm]!r} -- unexpected cross-id duplicate"
            )
        seen_norm[norm] = task_id
        by_id[task_id] = row

    out: list[dict] = []
    for task_id, row in by_id.items():
        category = row["type"]
        if category not in CATEGORIES:
            raise SeedConsolidationError(f"task {task_id!r} has unknown category {category!r}")
        if not _grader_ok(row):
            raise SeedConsolidationError(
                f"task {task_id!r} (grader={row.get('grader', 'exact')!r}) "
                f"fails its schema requirement"
            )

        out.append({
            "id": task_id,
            "category": category,
            "task": row["task"],
            "grader": row.get("grader", "exact"),
            "expected": row.get("expected"),
            "expected_text": row.get("expected_text"),
            "test_code": row.get("test_code"),
            "tier": None,
            "tier_source": None,
            "tier_heuristic": None,
            "source": "seed",
            "split": split_for(task_id),
        })

    out.sort(key=lambda r: r["id"])
    return out


def write_jsonl(tasks: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def write_seed_summary(tasks: list[dict], out_path: str | Path) -> dict[str, Any]:
    by_category = Counter(t["category"] for t in tasks)
    by_grader = Counter(t["grader"] for t in tasks)
    summary = {
        "total": len(tasks),
        "by_category": dict(sorted(by_category.items())),
        "by_grader": dict(sorted(by_grader.items())),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


EXPECTED_SEED_COUNTS = {
    "classification": 11, "code_debug": 10, "code_gen": 12, "extraction": 10,
    "factual_qa": 18, "language": 5, "logic": 13, "math": 19,
    "multiple_choice": 18, "ner": 5, "sentiment": 4, "summarization": 10,
}


def build_seeds(cheaproute_root: str | Path, seed_out: str | Path,
                summary_out: str | Path) -> list[dict]:
    raw_rows = load_seed_files(cheaproute_root)
    tasks = consolidate(raw_rows)

    if len(tasks) != 135:
        raise SeedConsolidationError(
            f"expected exactly 135 unique seed tasks, got {len(tasks)}"
        )
    actual_counts = Counter(t["category"] for t in tasks)
    if dict(actual_counts) != EXPECTED_SEED_COUNTS:
        raise SeedConsolidationError(
            f"seed category counts do not match the verified baseline:\n"
            f"  expected: {EXPECTED_SEED_COUNTS}\n"
            f"  actual:   {dict(sorted(actual_counts.items()))}"
        )

    write_jsonl(tasks, seed_out)
    write_seed_summary(tasks, summary_out)
    return tasks
