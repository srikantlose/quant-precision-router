"""Generate benchmark tasks with the Fireworks teacher model, verify them,
and merge with the seed set into data/tasks/all.jsonl.

Two passes per (category, tier) cell, both metered through a persistent API
call budget (data/generation/api_budget.json) that survives restarts --
every call is counted BEFORE it is sent, so a crash mid-call never
under-counts spend on the next run.

  Pass 1 (generation): ask the teacher for new, schema-valid, gradeable
    tasks at a stated difficulty tier, batch_size at a time, until the cell
    has enough validated candidates or max_gen_calls_per_cell is used.
    Validate: schema-complete; py_exec candidates must have their
    reference `expected` implementation pass its own `test_code` (run in
    the Docker sandbox -- a model-generated "reference" is never trusted
    without being executed); decontaminated against the 135 seed tasks and
    everything accepted so far in this run, by near-duplicate text match,
    not just exact match.

  Pass 2 (verification, non-py_exec only -- code tasks were already proven
    correct by executing them in pass 1): batches of up to
    verify_batch_size same-cell tasks per call. The teacher answers all of
    them at once (one numbered JSON array of answers) using the production
    local system prompt for that cell's category, and each answer is
    graded with its own task's grader. Only tasks whose teacher answer
    grades correct survive. This is a known, deliberate bias -- it drops
    tasks the teacher itself gets wrong, not just malformed ones -- and the
    per-cell drop rate is recorded rather than hidden.

If the budget runs out mid-run, generation stops with whatever exists kept
(not a failure); the shortfall is recorded in generation_log.json and the
report must state it.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

from .config import load_config
from .confidence import extract_final
from .graders import grade
from .tasks import split_for
from .tasktype import CATEGORIES, extract_answer, profile_for

_FENCE_STRIP = re.compile(r"^```[a-zA-Z0-9_+-]*\n|\n```$", re.MULTILINE)

# category name -> the classify()-output label that drives its prompting
# profile. Identity for every category except the two whose classify()
# label differs from the data category name (see tasktype.CATEGORY_MAP).
_CATEGORY_TO_LABEL = {"multiple_choice": "mc", "factual_qa": "short_qa"}


def _label_for_category(category: str) -> str:
    return _CATEGORY_TO_LABEL.get(category, category)


class GenerationError(Exception):
    pass


class BudgetExhausted(GenerationError):
    pass


# --------------------------------------------------------------------------
# Generation specs (8 ported from CheapRoute's eval/gen_synthetic.py, 4 new
# per the plan's Appendix B). Only `grader`, `instructions`, `example_schema`
# and optional `max_tokens`/`temperature`/`grader_override` matter here --
# the per-category/per-total `count` fields from the original are dropped,
# since QuantRoute targets a fixed count PER (category, tier) CELL instead.
# --------------------------------------------------------------------------

GEN_SPECS: dict[str, dict] = {
    "factual_qa": {
        "grader": "contains_all",
        "instructions": (
            "Generate factual-knowledge questions (explaining concepts, "
            "definitions, how things work). Some should be multi-part "
            "(e.g. 'What is the capital of Australia, and what body of "
            "water is it near?'). `expected` is a list of short strings "
            "that must ALL appear in a correct answer (one per part)."
        ),
        "example_schema": '{"task": "...", '
                          '"expected": ["canberra", "lake burley griffin"], '
                          '"grader": "contains_all"}',
    },
    "math": {
        "grader": "numeric",
        "instructions": (
            "Generate mathematical-reasoning word problems: multi-step "
            "arithmetic, percentages, projections (e.g. 'A store has 240 "
            "items. It sells 15% on Monday and 60 more on Tuesday. How "
            "many items remain?'). If the task counts discrete objects "
            "(widgets, books, people, boxes, etc.), choose numbers so "
            "every intermediate and final calculation is a whole number -- "
            "never design a problem whose arithmetic produces a fractional "
            "count of a discrete item. `expected` is the final numeric "
            "answer as a string."
        ),
        "example_schema": '{"task": "...", "expected": "144", "grader": "numeric"}',
    },
    "sentiment": {
        "grader": "contains",
        "instructions": (
            "Generate sentiment-classification tasks. Each `task` MUST be "
            "phrased exactly like: \"Classify the sentiment of this review "
            "as positive or negative: '<review text>'\" -- invent a new "
            "review/statement each time (a review can mention one minor "
            "downside/upside as long as the OVERALL sentiment is "
            "unambiguous -- do not invent genuinely 50/50 mixed cases, "
            "since the classifier only supports a binary label). "
            "`expected` is the single correct label word, either "
            "\"positive\" or \"negative\"."
        ),
        "example_schema": "{\"task\": \"Classify the sentiment of this "
                          "review as positive or negative: '...'\", "
                          "\"expected\": \"positive\", \"grader\": \"contains\"}",
    },
    "summarization": {
        "grader": "word_limit",
        "instructions": (
            "Generate summarization tasks: a short passage (3-6 sentences, "
            "invent realistic content) plus an instruction to summarize it "
            "in exactly one sentence or under N words. Give the model "
            "realistic room to comply: if the instruction is a plain 'in "
            "one sentence' with no explicit word count, set `max_words` to "
            "30-40 (a real one-sentence summary of a few facts routinely "
            "runs 25-35 words); only use a tight `max_words` like 15-22 "
            "when the task text ITSELF states an explicit count (e.g. 'in "
            "20 words or fewer'), matching that count plus a couple words "
            "of slack. `expected` is {\"max_words\": N, \"any_of\": [2-4 "
            "keywords a correct summary would very likely include]}."
        ),
        "example_schema": '{"task": "Summarize the following in exactly '
                          'one sentence: <passage>", '
                          '"expected": {"max_words": 25, '
                          '"any_of": ["keyword1", "keyword2"]}, '
                          '"grader": "word_limit"}',
    },
    "ner": {
        "grader": "contains_all",
        "instructions": (
            "Generate named-entity-recognition tasks: a sentence with "
            "person/organization/location/date entities, asking to extract "
            "and label them (e.g. 'Extract all named entities and their "
            "types from: Maria Sanchez joined Fireworks AI in Berlin last "
            "March.'). `expected` is a list of the entity strings that "
            "must all appear in a correct answer."
        ),
        "example_schema": '{"task": "...", '
                          '"expected": ["maria sanchez", "fireworks ai", '
                          '"berlin"], "grader": "contains_all"}',
    },
    "code_debug": {
        "grader": "py_exec",
        "max_tokens": 14000,
        "instructions": (
            "Generate Python code-debugging tasks: a short function with a "
            "clear, single bug and a description of the intended behavior "
            "(e.g. 'This function should return the max of a list but has "
            "a bug: def get_max(nums): return nums[0]. Find and fix it.'). "
            "`expected` is the CORRECTED function as a fenced python code "
            "block (this is the reference fix, used to verify test_code, "
            "not shown to the model being trained). `test_code` is 2-3 "
            "assert statements that the corrected function must pass."
        ),
        "example_schema": '{"task": "...", '
                          '"expected": "```python\\ndef get_max(nums):\\n'
                          '    return max(nums)\\n```", '
                          '"test_code": "assert get_max([1,5,3]) == 5\\n'
                          'assert get_max([2,2]) == 2", "grader": "py_exec"}',
    },
    "logic": {
        "grader": "contains",
        "max_tokens": 14000,
        "temperature": 0.55,
        "instructions": (
            "Generate logical/deductive-reasoning puzzles where all given "
            "constraints must be satisfied to reach a unique answer (e.g. "
            "'Three friends, Sam, Jo, and Lee, each own a different pet: "
            "cat, dog, bird. Sam does not own the bird. Jo owns the dog. "
            "Who owns the cat?'). Before finalizing each puzzle, verify by "
            "systematically checking the possibilities that the stated "
            "constraints rule out every assignment except one -- if more "
            "than one assignment satisfies all the constraints, add "
            "another constraint (or change the numbers) until the solution "
            "is unique. A puzzle with two valid answers is a broken "
            "puzzle; `expected` is the short final answer."
        ),
        "example_schema": '{"task": "...", "expected": "sam", "grader": "contains"}',
    },
    "code_gen": {
        "grader": "py_exec",
        "max_tokens": 14000,
        "instructions": (
            "Generate code-generation specs: a clear description of a "
            "Python function to write, including edge cases to handle "
            "(e.g. 'Write a Python function that returns the second-"
            "largest number in a list, handling duplicates correctly.'). "
            "`expected` is a REFERENCE correct implementation as a fenced "
            "python code block. `test_code` is 3-4 assert statements "
            "covering the spec including edge cases."
        ),
        "example_schema": '{"task": "...", '
                          '"expected": "```python\\ndef f(...):\\n    ...\\n'
                          '```", "test_code": "assert f(...) == ...", '
                          '"grader": "py_exec"}',
    },
    "multiple_choice": {
        "grader": "mc",
        "instructions": (
            "Generate multiple-choice questions with exactly four options "
            "labelled A) B) C) D) on one line inside the task text, ending "
            "with 'Answer with the letter.'. Exactly one option is "
            "correct. `expected` is the correct letter (A, B, C or D) and "
            "`expected_text` is the exact text of the correct option. Vary "
            "which letter is correct."
        ),
        "example_schema": '{"task": "Which is the largest planet in our '
                          'solar system? A) Earth B) Jupiter C) Mars D) '
                          'Venus. Answer with the letter.", "expected": '
                          '"B", "expected_text": "Jupiter", "grader": "mc"}',
    },
    "classification": {
        "grader": "contains",
        "instructions": (
            "Generate text-classification tasks that are NOT about "
            "sentiment. Use topic (sports, politics, technology, health, "
            "finance), spam vs not spam, formal vs informal, fact vs "
            "opinion, or question vs statement. The task text must name "
            "the allowed labels and quote the text to classify. `expected` "
            "is the single correct label, lowercase, exactly as named in "
            "the task."
        ),
        "example_schema": '{"task": "Classify the topic of this headline '
                          'as sports, politics, technology, health or '
                          "finance: 'Central bank holds interest rates "
                          "steady for third month'\", \"expected\": "
                          '"finance", "grader": "contains"}',
    },
    "extraction": {
        "grader": "contains",
        "instructions": (
            "Generate information-extraction tasks: quote a short "
            "realistic text (one to three sentences) and ask for one "
            "specific value from it (an order number, a date, an amount, "
            "an email, a name, a code). Say 'Reply with just the value.' "
            "`expected` is the exact value. If the value is purely "
            "numeric, set `grader` to `numeric`; otherwise `contains`."
        ),
        "example_schema": '{"task": "Extract the order number from this '
                          "text: 'Order #4521 was shipped to 42 Elm Street "
                          "on March 3rd.' Reply with just the value.\", "
                          '"expected": "4521", "grader": "numeric"}',
    },
    "language": {
        "grader": "contains",
        "instructions": (
            "Generate short language tasks: antonyms, synonyms, plurals, "
            "past tenses, or translations of a word or short phrase "
            "between English and French, Spanish or German. There must be "
            "one clearly correct short answer. `expected` is that answer "
            "in lowercase without punctuation."
        ),
        "example_schema": "{\"task\": \"What is the past tense of 'swim'?\", "
                          '"expected": "swam", "grader": "contains"}',
    },
}

assert set(GEN_SPECS) == set(CATEGORIES), set(GEN_SPECS) ^ set(CATEGORIES)

TIER_DEFS: dict[str, dict[str, str]] = {
    "math": {
        "easy": "One arithmetic operation on small numbers.",
        "moderate": "Two or three dependent steps (e.g. a percentage, then a subtraction).",
        "complex": "Four or more dependent steps, several quantities, rates or unit conversions.",
    },
    "multiple_choice": {
        "easy": "Direct recall; distractors are clearly wrong.",
        "moderate": "Needs one inference; distractors are plausible.",
        "complex": "Needs multi-step reasoning; two options are close competitors.",
    },
    "factual_qa": {
        "easy": "One well-known fact.",
        "moderate": "A two-part question or a less common fact.",
        "complex": "Three parts, a comparison, or connecting two facts.",
    },
    "classification": {
        "easy": "Explicit cue words make the label obvious (not sentiment).",
        "moderate": "The cue is implicit (not sentiment).",
        "complex": "Mixed signals with one clear overall label, or four or more "
                  "candidate labels (not sentiment).",
    },
    "sentiment": {
        "easy": "Explicit polarity words.",
        "moderate": "Mild or hedged polarity.",
        "complex": "Sarcasm, or a minor opposite-polarity aside with a clear overall polarity.",
    },
    "extraction": {
        "easy": "One value, clearly stated.",
        "moderate": "The target value appears among distractor values of the same kind.",
        "complex": "Requires disambiguation (e.g. the corrected value, or the "
                  "second of several dates).",
    },
    "language": {
        "easy": "A common word transformation (antonym, plural, simple translation).",
        "moderate": "An irregular form or less common word.",
        "complex": "An idiom, a rare word, or translating a short phrase with "
                  "word-order changes.",
    },
    "logic": {
        "easy": "One deduction step.",
        "moderate": "Two or three constraints.",
        "complex": "Four or more constraints or an ordering puzzle.",
    },
    "summarization": {
        "easy": "60-100-word passage; one-sentence summary; max_words 35-40.",
        "moderate": "120-180-word passage; explicit word limit 20-25.",
        "complex": "200-300-word passage with several points; explicit limit "
                  "<= 25 words; two required keywords.",
    },
    "ner": {
        "easy": "Two or three entities of one type.",
        "moderate": "Four or five entities across two or three types.",
        "complex": "Six or more entities including an ambiguous one (e.g. an "
                  "organisation named after a person).",
    },
    "code_gen": {
        "easy": "One function with a single loop or conditional; 3 asserts.",
        "moderate": "Handles edge cases (empty input, negatives); 4 asserts.",
        "complex": "An algorithmic task (intervals, parsing, recursion); 5+ "
                  "asserts including edge cases.",
    },
    "code_debug": {
        "easy": "One obvious bug (wrong operator or return).",
        "moderate": "An off-by-one or edge-case bug.",
        "complex": "Two interacting bugs, or a logic bug that requires "
                  "understanding the stated intent.",
    },
}

assert set(TIER_DEFS) == set(CATEGORIES), set(TIER_DEFS) ^ set(CATEGORIES)

_GEN_SYSTEM = (
    "You generate training data for an AI evaluation benchmark. Output "
    "ONLY a JSON array of task objects, no commentary, no markdown fences. "
    "Every task must be self-contained, unambiguous, and have a single "
    "objectively correct answer. Do not reuse the style examples verbatim "
    "-- invent new content (different numbers, names, topics, code)."
)

_SHORT_ANSWER_RIDER = (" At most 1-2 brief reasoning sentences, then the final "
                      "line exactly as instructed.")
_CODE_ANSWER_RIDER = " Return only the fenced code block, no commentary."
_FREEFORM_ANSWER_RIDER = " Output only the requested text, nothing else."


# --------------------------------------------------------------------------
# Budget counter -- persisted, checked before every call, never reset by a
# restart.
# --------------------------------------------------------------------------

def _budget_path() -> Path:
    return Path("data/generation/api_budget.json")


def load_budget(cap: int) -> dict[str, int]:
    path = _budget_path()
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            budget = json.load(f)
        budget["cap"] = cap  # config may have changed the cap since last run
        return budget
    return {"cap": cap, "used": 0}


def save_budget(budget: dict[str, int]) -> None:
    path = _budget_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(budget, f, ensure_ascii=False, indent=2)


def spend_one_call(cap: int) -> int:
    """Read -> check -> increment -> write, BEFORE the API call is sent, so
    a crash mid-call never under-counts spend on the next run. Returns the
    new `used` count."""
    budget = load_budget(cap)
    if budget["used"] >= budget["cap"]:
        raise BudgetExhausted(f"API call budget exhausted ({budget['used']}/{budget['cap']})")
    budget["used"] += 1
    save_budget(budget)
    return budget["used"]


# --------------------------------------------------------------------------
# Fireworks client
# --------------------------------------------------------------------------

def fireworks_chat(system: str, user: str, max_tokens: int, temperature: float) -> str:
    """One teacher call, charged to the generation budget."""
    spend_one_call(load_config()["generation"]["api_cap"])
    return fireworks_request(system, user, max_tokens, temperature)


def fireworks_request(system: str, user: str, max_tokens: int, temperature: float) -> str:
    """One teacher call with no budget accounting; callers charge their own budget first."""
    cfg = load_config()["generation"]
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        raise GenerationError("FIREWORKS_API_KEY is not set")
    resp = requests.post(
        f"{cfg['base_url']}/chat/completions",
        json={
            "model": cfg["teacher"],
            "messages": [{"role": "system", "content": system},
                        {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=cfg["timeout_s"],
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"].get("content") or ""


def probe_teacher() -> None:
    """One cheap call to confirm the teacher is reachable before spending
    any real generation budget. Raises GenerationError on any failure.
    max_tokens is generous enough to survive a reasoning model's hidden
    reasoning_content tokens, which count against the same budget as the
    visible answer and can otherwise consume it entirely."""
    try:
        reply = fireworks_chat("Reply with exactly OK.", "OK?", max_tokens=64, temperature=0.0)
    except Exception as exc:
        raise GenerationError(f"probe call failed: {exc}") from exc
    if "OK" not in reply:
        raise GenerationError(f"probe call succeeded but reply lacked 'OK': {reply!r}")


# --------------------------------------------------------------------------
# Pass 1: generation
# --------------------------------------------------------------------------

def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_STRIP.sub("", text).strip()
    return text


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _similar(a: str, b: str, threshold: float = 0.93) -> bool:
    """Near-duplicate check. difflib's ratio conflates "shares a template"
    with "shares content" -- two distinct-fact questions built on the same
    template can score ~0.88, so the threshold sits above that band and
    relies on the substring check to still catch true near-verbatim
    duplicates/paraphrases."""
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() > threshold


def generate_batch(category: str, tier: str, batch_size: int) -> list[dict]:
    spec = GEN_SPECS[category]
    tier_text = TIER_DEFS[category][tier]
    prompt = (
        f"{spec['instructions']}\n\nEach object must match this schema "
        f"exactly (extra/missing keys not allowed): {spec['example_schema']}"
        f"\n\nDIFFICULTY TIER: {tier}. {tier_text} Every task in this batch "
        f"must be at this tier."
        f"\n\nGenerate a JSON array of exactly {batch_size} NEW, DISTINCT "
        f"tasks of this kind. Return ONLY the JSON array."
    )
    # 12000 (up from an earlier 4000 default): gpt-oss-120b is a reasoning
    # model whose hidden reasoning_content tokens count against the same
    # max_tokens budget as the visible answer. On harder prompts, reasoning
    # alone can consume the entire budget (observed: 3997/4000 tokens, all
    # reasoning, finish_reason="length", zero visible content) -- this is
    # not a reasoning_effort problem to paper over by reducing reasoning
    # (that trades away exactly the careful self-checking multi-step
    # problems need, and produces wrong ground-truth answers on harder
    # tiers); it is a token-budget problem, fixed by giving the model
    # enough room to finish reasoning AND still emit the full JSON array.
    raw = fireworks_chat(_GEN_SYSTEM, prompt, max_tokens=spec.get("max_tokens", 12000),
                         temperature=spec.get("temperature", 0.9))
    try:
        items = json.loads(_strip_json_fences(raw))
        if not isinstance(items, list):
            return []
    except json.JSONDecodeError:
        print(f"  [{category}/{tier}] generation returned non-JSON", file=sys.stderr)
        return []

    out = []
    for item in items:
        if not isinstance(item, dict) or "task" not in item or "expected" not in item:
            continue
        item = dict(item)
        item["category"] = category
        item.setdefault("grader", spec.get("grader_override", spec["grader"]))
        out.append(item)
    return out


def validate_and_filter(items: list[dict], seed_norms: list[str],
                        local_seen: list[str]) -> list[dict]:
    """Schema/execution validation + decontamination + dedup. `local_seen`
    is mutated in place (appended to) so the caller's running set grows
    across calls within a cell."""
    kept = []
    for item in items:
        task_text = (item.get("task") or "").strip()
        if not task_text or len(task_text) < 8:
            continue
        expected = item.get("expected")
        if expected is None or expected == "":
            continue
        grader_kind = item.get("grader", "exact")

        if grader_kind == "mc" and not item.get("expected_text"):
            continue  # under-specified; ask again next batch rather than accept it

        if grader_kind == "py_exec":
            test_code = item.get("test_code", "")
            if not test_code:
                continue
            # the "reference" implementation is model-generated too -- it is
            # never trusted without being executed (in the Docker sandbox,
            # grade()'s default executor).
            if not grade({"grader": "py_exec", "test_code": test_code}, str(expected)):
                continue

        norm = _norm(task_text)
        if any(norm == s or _similar(norm, s) for s in seed_norms):
            continue  # near-duplicate of a seed task
        if any(_similar(norm, s) for s in local_seen):
            continue  # near-duplicate of something already accepted
        local_seen.append(norm)
        kept.append(item)
    return kept


# --------------------------------------------------------------------------
# Pass 2: verification (non-py_exec only)
# --------------------------------------------------------------------------

def _answer_style_rider(label: str) -> str:
    prof = profile_for(label)
    if label in ("code_debug", "code_gen"):
        return _CODE_ANSWER_RIDER
    return _FREEFORM_ANSWER_RIDER if prof.freeform else _SHORT_ANSWER_RIDER


def verify_batch(category: str, batch: list[dict]) -> list[dict]:
    """Ask the teacher to answer all of `batch` (<= verify_batch_size items,
    same category) in one call; keep only the ones whose teacher answer
    grades correct against the task's own grader."""
    label = _label_for_category(category)
    prof = profile_for(label)
    system = f"You are a careful assistant. {prof.local_style}" + _answer_style_rider(label)

    numbered = "\n".join(f"{i+1}. {t['task']}" for i, t in enumerate(batch))
    user = (
        f"Answer each of the following {len(batch)} tasks. Return ONLY a "
        f"JSON array of exactly {len(batch)} strings, in order -- each "
        f"string is the complete answer to the correspondingly numbered "
        f"task, following the system instructions for format.\n\n{numbered}"
    )
    # Headroom for the same hidden-reasoning-token reason as generate_batch,
    # plus the visible per-item answers; "+2000" covers the model's
    # reasoning about a small batch of short answers, which in practice
    # needs much less than a full generation call but is not zero.
    max_tokens = max(prof.local_max_tokens * len(batch) + 2000, 2500)
    raw = fireworks_chat(system, user, max_tokens=max_tokens, temperature=0.3)
    try:
        answers = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError:
        print(f"  [{category}] verification returned non-JSON", file=sys.stderr)
        return []
    if not isinstance(answers, list) or len(answers) != len(batch):
        print(f"  [{category}] verification returned {len(answers) if isinstance(answers, list) else type(answers)} "
             f"answers for a batch of {len(batch)}", file=sys.stderr)
        return []

    kept = []
    for task, raw_answer in zip(batch, answers):
        if not isinstance(raw_answer, str) or not raw_answer.strip():
            continue
        extracted = extract_answer(label, raw_answer, extract_final)
        if grade(task, extracted):
            kept.append(task)
    return kept


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def generate_cell(category: str, tier: str, target: int, max_calls: int,
                  batch_size: int, seed_norms: list[str], global_norms: list[str],
                  raw_dir: Path) -> tuple[list[dict], dict]:
    """Generate (pass 1) >= target validated candidates for one
    (category, tier) cell, or stop after max_calls batches -- whichever
    comes first. BudgetExhausted propagates to the caller."""
    local_seen: list[str] = list(global_norms)  # start from everything accepted so far
    kept: list[dict] = []
    raw_count = 0
    calls_used = 0
    raw_path = raw_dir / f"{category}-{tier}.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_path, "a", encoding="utf-8") as raw_f:
        while len(kept) < target and calls_used < max_calls:
            batch = generate_batch(category, tier, batch_size)
            calls_used += 1
            raw_count += len(batch)
            for item in batch:
                raw_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            raw_f.flush()
            validated = validate_and_filter(batch, seed_norms, local_seen)
            kept.extend(validated)

    stats = {"calls_used": calls_used, "raw_count": raw_count,
             "validated_count": len(kept)}
    return kept, stats


def verify_cell(category: str, candidates: list[dict], batch_size: int) -> tuple[list[dict], dict]:
    """Pass 2 for one cell's candidates (skipped entirely for py_exec
    categories, whose candidates were already proven correct by execution
    in pass 1)."""
    spec = GEN_SPECS[category]
    if spec["grader"] == "py_exec":
        return candidates, {"calls_used": 0, "verified_count": len(candidates),
                            "dropped_count": 0}

    kept: list[dict] = []
    calls_used = 0
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        kept.extend(verify_batch(category, batch))
        calls_used += 1
    return kept, {"calls_used": calls_used, "verified_count": len(kept),
                 "dropped_count": len(candidates) - len(kept)}


def _assign_ids(category: str, tier: str, items: list[dict]) -> list[dict]:
    out = []
    for i, item in enumerate(items, start=1):
        row = {
            "id": f"gen-{category}-{tier}-{i:04d}",
            "category": category,
            "task": item["task"],
            "grader": item.get("grader", "exact"),
            "expected": item.get("expected"),
            "expected_text": item.get("expected_text"),
            "test_code": item.get("test_code"),
            "tier": tier,
            "tier_source": "requested",
            "tier_heuristic": None,
            "source": "fireworks",
            "split": split_for(f"gen-{category}-{tier}-{i:04d}"),
        }
        out.append(row)
    return out


def run_generation(seed_tasks: list[dict], out_dir: Path = Path("data")) -> dict:
    cfg = load_config()["generation"]
    target = cfg["per_cell_target"]
    batch_size = cfg["batch_size"]
    max_gen_calls = cfg["max_gen_calls_per_cell"]
    verify_batch_size = cfg["verify_batch_size"]

    raw_dir = out_dir / "generation" / "raw"
    seed_norms = [_norm(t["task"]) for t in seed_tasks]
    global_norms: list[str] = []

    generated_rows: list[dict] = []
    cell_log: dict[str, Any] = {}
    budget_exhausted_at: Optional[str] = None

    for category in CATEGORIES:
        for tier in ("easy", "moderate", "complex"):
            cell_key = f"{category}|{tier}"
            try:
                candidates, gen_stats = generate_cell(
                    category, tier, target, max_gen_calls, batch_size,
                    seed_norms, global_norms, raw_dir,
                )
                verified, verify_stats = verify_cell(category, candidates, verify_batch_size)
            except BudgetExhausted as exc:
                budget_exhausted_at = cell_key
                cell_log[cell_key] = {"error": str(exc), "shortfall": True}
                break

            trimmed = verified[:target]
            rows = _assign_ids(category, tier, trimmed)
            generated_rows.extend(rows)
            global_norms.extend(_norm(r["task"]) for r in rows)

            cell_log[cell_key] = {
                **gen_stats, **verify_stats,
                "final_count": len(rows), "target": target,
                "shortfall": len(rows) < target,
            }
            print(f"[{cell_key}] {len(rows)}/{target} "
                 f"(gen_calls={gen_stats['calls_used']}, "
                 f"verify_calls={verify_stats['calls_used']})", file=sys.stderr)
        if budget_exhausted_at:
            break

    return {"rows": generated_rows, "cell_log": cell_log,
           "budget_exhausted_at": budget_exhausted_at}


def resume_generation(seed_tasks: list[dict], existing_generated: list[dict],
                      out_dir: Path = Path("data")) -> dict:
    """Top up only the (category, tier) cells that fell short of target in
    a previous run -- cells already at target are left completely
    untouched (not re-generated, not re-verified). Existing rows keep
    their exact ids; new rows continue the per-cell sequence from where it
    left off. This exists because a from-scratch re-run of every cell
    would spend the API budget a second time on cells that already
    succeeded."""
    cfg = load_config()["generation"]
    target = cfg["per_cell_target"]
    batch_size = cfg["batch_size"]
    max_gen_calls = cfg["max_gen_calls_per_cell"]
    verify_batch_size = cfg["verify_batch_size"]

    raw_dir = out_dir / "generation" / "raw"
    seed_norms = [_norm(t["task"]) for t in seed_tasks]

    by_cell: dict[tuple[str, str], list[dict]] = {}
    for r in existing_generated:
        by_cell.setdefault((r["category"], r["tier"]), []).append(r)
    for rows in by_cell.values():
        rows.sort(key=lambda r: r["id"])

    global_norms = [_norm(t["task"]) for t in seed_tasks] + \
        [_norm(r["task"]) for r in existing_generated]

    final_rows: list[dict] = list(existing_generated)
    cell_log: dict[str, Any] = {}
    budget_exhausted_at: Optional[str] = None

    for category in CATEGORIES:
        for tier in ("easy", "moderate", "complex"):
            cell_key = f"{category}|{tier}"
            existing_cell_rows = by_cell.get((category, tier), [])
            needed = target - len(existing_cell_rows)
            if needed <= 0:
                cell_log[cell_key] = {"skipped": True, "existing_count": len(existing_cell_rows)}
                continue

            try:
                candidates, gen_stats = generate_cell(
                    category, tier, needed, max_gen_calls, batch_size,
                    seed_norms, global_norms, raw_dir,
                )
                verified, verify_stats = verify_cell(category, candidates, verify_batch_size)
            except BudgetExhausted as exc:
                budget_exhausted_at = cell_key
                cell_log[cell_key] = {"error": str(exc), "shortfall": True,
                                      "existing_count": len(existing_cell_rows)}
                break

            trimmed = verified[:needed]
            start_at = len(existing_cell_rows) + 1
            new_rows = []
            for i, item in enumerate(trimmed, start=start_at):
                row_id = f"gen-{category}-{tier}-{i:04d}"
                new_rows.append({
                    "id": row_id, "category": category, "task": item["task"],
                    "grader": item.get("grader", "exact"),
                    "expected": item.get("expected"),
                    "expected_text": item.get("expected_text"),
                    "test_code": item.get("test_code"),
                    "tier": tier, "tier_source": "requested", "tier_heuristic": None,
                    "source": "fireworks", "split": split_for(row_id),
                })
            final_rows.extend(new_rows)
            global_norms.extend(_norm(r["task"]) for r in new_rows)

            cell_log[cell_key] = {
                **gen_stats, **verify_stats,
                "existing_count": len(existing_cell_rows),
                "added_count": len(new_rows),
                "final_count": len(existing_cell_rows) + len(new_rows),
                "target": target,
                "shortfall": (len(existing_cell_rows) + len(new_rows)) < target,
            }
            print(f"[{cell_key}] {len(existing_cell_rows)} existing + {len(new_rows)} new "
                 f"= {len(existing_cell_rows) + len(new_rows)}/{target} "
                 f"(gen_calls={gen_stats['calls_used']}, verify_calls={verify_stats['calls_used']})",
                 file=sys.stderr)
        if budget_exhausted_at:
            break

    return {"rows": final_rows, "cell_log": cell_log,
           "budget_exhausted_at": budget_exhausted_at}


def write_outputs(seed_tasks: list[dict], generation_result: dict,
                  out_dir: Path = Path("data"), results_dir: Path = Path("results")) -> dict:
    """Steps 10-12: fit heuristic tiers over seed+generated, fill them in
    for the seed tasks (tier_source="heuristic"), write generated.jsonl and
    all.jsonl (sorted by id), and write task_summary.json /
    tier_agreement.json."""
    from . import tiers as tiers_mod

    generated_rows = generation_result["rows"]
    all_tasks = [dict(t) for t in seed_tasks] + [dict(r) for r in generated_rows]

    tier_result = tiers_mod.assign_heuristic_tiers(all_tasks)
    tier_heuristic = tier_result["tier_heuristic"]
    for t in all_tasks:
        t["tier_heuristic"] = tier_heuristic.get(t["id"])
        if t.get("source") == "seed":
            t["tier"] = t["tier_heuristic"]
            t["tier_source"] = "heuristic"

    all_tasks.sort(key=lambda t: t["id"])
    generated_sorted = sorted(
        (t for t in all_tasks if t.get("source") == "fireworks"),
        key=lambda t: t["id"],
    )

    tasks_dir = out_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    with open(tasks_dir / "generated.jsonl", "w", encoding="utf-8") as f:
        for t in generated_sorted:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(tasks_dir / "all.jsonl", "w", encoding="utf-8") as f:
        for t in all_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    kappa = tiers_mod.cross_validated_kappa(all_tasks)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "tier_agreement.json", "w", encoding="utf-8") as f:
        json.dump({**tier_result, "cross_validated_kappa": kappa}, f,
                 ensure_ascii=False, indent=2)

    summary: dict[str, Any] = {"total": len(all_tasks), "by_category_tier_split_source": {},
                               "cell_log": generation_result["cell_log"],
                               "budget_exhausted_at": generation_result["budget_exhausted_at"]}
    counts: dict[str, int] = {}
    for t in all_tasks:
        key = f"{t['category']}|{t.get('tier')}|{t['split']}|{t['source']}"
        counts[key] = counts.get(key, 0) + 1
    summary["by_category_tier_split_source"] = dict(sorted(counts.items()))

    budget = load_budget(load_config()["generation"]["api_cap"])
    summary["api_calls_used"] = budget["used"]
    summary["api_cap"] = budget["cap"]

    with open(results_dir / "task_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    gen_dir = out_dir / "generation"
    gen_dir.mkdir(parents=True, exist_ok=True)
    with open(gen_dir / "generation_log.json", "w", encoding="utf-8") as f:
        json.dump(generation_result["cell_log"], f, ensure_ascii=False, indent=2)

    return summary
