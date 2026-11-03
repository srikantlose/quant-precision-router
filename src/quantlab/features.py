"""Feature extraction for the required-precision predictor (plan Sec 9.3).

Two independent feature families, concatenated by predictor.py:
  - a 384-dim MiniLM sentence embedding of the task text (semantic)
  - 13 cheap surface features (lexical/structural)

Scaling the surface features happens inside predictor.py's cross-validation
loop, one StandardScaler per fold fit on that fold's training rows only --
never here, since fitting a scaler over all rows first would leak test-set
statistics into train.

Embeddings are the expensive part (a forward pass per text), so they are
computed once over every id that needs one and cached to
results/raw/embeddings.npy, with the id order saved alongside it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .tasktype import _CODE_MARKERS

SURFACE_FEATURE_NAMES = [
    "n_chars", "n_words", "n_sentences", "n_numbers", "log_max_number",
    "n_math_ops", "n_conjunctions", "n_step_markers", "n_question_marks",
    "has_code", "n_code_lines", "n_quotes", "n_list_items",
]

_NUMBER_RE = re.compile(r"-?\d[\d,]*(\.\d+)?")
_MATH_OP_RE = re.compile(r"[+\-*/^=%]")
_CONJUNCTION_RE = re.compile(r"\b(and|then|after|before|if|but|while|unless)\b", re.IGNORECASE)
_STEP_MARKER_RE = re.compile(r"\b(first|second|then|next|finally|step)\b", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_FENCED_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s", re.MULTILINE)


def surface_features(text: str) -> list[float]:
    text = text or ""
    numbers = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0).replace(",", "")
        try:
            numbers.append(abs(float(raw)))
        except ValueError:
            continue
    n_sentences = len([s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()])
    has_code = bool(_CODE_MARKERS.search(text))
    code_lines = 0
    for block in _FENCED_BLOCK_RE.findall(text):
        code_lines += sum(1 for ln in block.splitlines() if ln.strip())
    for line in text.splitlines():
        if line.startswith("    ") and line.strip():
            code_lines += 1
    n_quotes = (text.count("'") + text.count('"')) / 2

    return [
        float(len(text)),
        float(len(text.split())),
        float(n_sentences),
        float(len(numbers)),
        float(np.log10(1 + max(numbers))) if numbers else 0.0,
        float(len(_MATH_OP_RE.findall(text))),
        float(len(_CONJUNCTION_RE.findall(text))),
        float(len(_STEP_MARKER_RE.findall(text))),
        float(text.count("?")),
        float(has_code),
        float(code_lines),
        float(n_quotes),
        float(len(_LIST_ITEM_RE.findall(text))),
    ]


def _load_embedder():
    from sentence_transformers import SentenceTransformer
    from .config import load_config
    cfg = load_config()
    model_name = cfg["predictor"]["embedding_model"]
    return SentenceTransformer(model_name, device="cpu")


def compute_embeddings(
    id_text_pairs: list[tuple[str, str]],
    out_path: str | Path = "results/raw/embeddings.npy",
) -> dict[str, np.ndarray]:
    """Embed every (id, text) pair once; cache to out_path plus an id-index
    JSON next to it. Returns {id: embedding}."""
    ids = [i for i, _ in id_text_pairs]
    texts = [t for _, t in id_text_pairs]

    model = _load_embedder()
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vectors)
    index_path = out_path.with_suffix(".ids.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)

    return {i: vectors[k] for k, i in enumerate(ids)}


def load_embeddings(path: str | Path = "results/raw/embeddings.npy") -> dict[str, np.ndarray]:
    path = Path(path)
    vectors = np.load(path)
    index_path = path.with_suffix(".ids.json")
    with open(index_path, encoding="utf-8") as f:
        ids = json.load(f)
    return {i: vectors[k] for k, i in enumerate(ids)}


def one_hot_category(category: str, categories: list[str]) -> list[float]:
    return [1.0 if category == c else 0.0 for c in categories]
