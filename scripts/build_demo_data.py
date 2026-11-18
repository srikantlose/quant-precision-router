"""Builds dashboard-site/demo-data.js for the replay demo (dashboard-site/demo.html).

The demo replays the routing decision for every held-out (test-split) benchmark
question using only recorded artefacts, so it runs in any browser with no model,
GPU or Python:

  - category_pred and the predicted required-precision class come from
    results/raw/pred_class.json (the real predictor output, not recomputed);
  - the safety-classifier score is computed here from the saved classifier
    (results/raw/safety_classifier.joblib) and the saved MiniLM embeddings;
  - every rung's answer, correctness and latency come from results/raw/grid.jsonl.

Policy cells and rung-level safety bounds already ship in data.js, so the page
recomputes the decision in the browser exactly as the dashboard's policy lab does.

Safety-slice prompts are deliberately left out: XSTest declares no licence, and
the project never publishes raw completions from the safety slice.

Like data.js, the output is generated on the machine that ran the benchmark
(the raw grid is gitignored) and committed alongside the site.
"""

import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LADDER = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]
MAX_ANSWER_CHARS = 420


def load_jsonl(rel):
    with open(ROOT / rel, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def expected_text(task):
    g, e = task["grader"], task["expected"]
    if g == "py_exec":
        return "code that passes the task's hidden tests"
    if g == "word_limit":
        return f"at most {e['max_words']} words, mentioning {' or '.join(e['any_of'])}"
    if g == "contains_all":
        return " + ".join(e)
    if g == "contains" and isinstance(e, list):
        return " or ".join(e)
    if g == "mc" and task.get("expected_text"):
        return f"{e} ({task['expected_text']})"
    return str(e)


def clip(text):
    text = (text or "").strip()
    return text if len(text) <= MAX_ANSWER_CHARS else text[:MAX_ANSWER_CHARS].rstrip() + " …"


def main():
    tasks = [t for t in load_jsonl("data/tasks/all.jsonl") if t["split"] == "test"]
    labels = json.loads((ROOT / "results/labels.json").read_text(encoding="utf-8"))
    preds = json.loads((ROOT / "results/raw/pred_class.json").read_text(encoding="utf-8"))
    clf = joblib.load(ROOT / "results/raw/safety_classifier.joblib")
    ids = json.loads((ROOT / "results/raw/embeddings.ids.json").read_text(encoding="utf-8"))
    emb = np.load(ROOT / "results/raw/embeddings.npy")
    row_of = {tid: i for i, tid in enumerate(ids)}

    wanted = {t["id"] for t in tasks}
    grid = {}
    for row in load_jsonl("results/raw/grid.jsonl"):
        if row["slice"] == "bench" and row["task_id"] in wanted and row.get("error") is None:
            grid.setdefault(row["task_id"], {})[row["rung"]] = row

    scores = clf["model"].predict_proba(np.stack([emb[row_of[t["id"]]] for t in tasks]))[:, 1]

    out = []
    for t, score in zip(tasks, scores):
        rows = grid[t["id"]]
        assert set(rows) == set(LADDER), t["id"]
        out.append({
            "id": t["id"],
            "cat": t["category"],
            "catp": preds["category_pred"][t["id"]],
            "tier": t["tier"],
            "grader": t["grader"],
            "p": t["task"],
            "exp": expected_text(t),
            "cls": preds["pred_class_test"][t["id"]],
            "req": labels["req"].get(t["id"]),
            "score": round(float(score), 4),
            "a": [[int(bool(rows[r]["correct"])), round(rows[r]["latency_s"], 3),
                   rows[r]["tokens_out"], clip(rows[r]["answer"])] for r in LADDER],
        })
    out.sort(key=lambda d: (d["cat"], d["tier"], d["id"]))

    data = {
        "meta": {
            "n": len(out),
            "threshold": clf["threshold"],
            "tau": labels["tau"],
            "n_flagged": int(sum(d["score"] >= clf["threshold"] for d in out)),
        },
        "tasks": out,
    }
    path = ROOT / "dashboard-site" / "demo-data.js"
    path.write_text("window.QLDEMO = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
                    encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB, {len(out)} tasks, "
          f"{data['meta']['n_flagged']} flagged by the safety classifier)")


if __name__ == "__main__":
    main()
