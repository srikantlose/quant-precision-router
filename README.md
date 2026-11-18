# QuantRoute

**Serving each LLM prompt at the cheapest quantization level that stays accurate and safe.**

QuantRoute takes one language model, Qwen2.5-3B-Instruct, at seven GGUF quantization levels ("rungs": Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0 and the F16 reference). It measures how each level behaves on 1,467 graded tasks and 600 safety prompts on a single 8 GB RTX 4060, and learns a policy that sends every incoming prompt to the cheapest level whose accuracy, answer stability and safety all stay within set bounds of full precision. The policy runs as a hardened local HTTP service with a web page that shows each routing decision step by step.

The project is also an honest measurement study. Quantization below 4 bits clearly hurts. Even near-lossless levels change some answers. And on this model the best single safe level (Q5_K_M) is about as good as per-prompt routing. Those findings, and the tooling that produced them, are the main result.

| | |
|---|---|
| Results dashboard | https://quant-precision-router.vercel.app |
| Replay demo (450 held-out questions, no install) | https://quant-precision-router.vercel.app/demo |
| Beginner study guide | https://quant-precision-router.vercel.app/guide |
| Revision notes and viva Q&A | https://quant-precision-router.vercel.app/revision |
| Project ID | PID-CY014 |

![Ask QuantRoute: a live question, the five routing steps, the answer from the chosen version and a forced Q2_K comparison](docs/images/ask-quantroute.png)

> **Status.** Measurement, learning and serving are complete (plan phases P0–P10 and P12): 14,469 graded runs, the fitted policy, and the live routing service. The last analysis steps (threshold sweep, swap simulation, baselines with bootstrap intervals, differential test, refusal audit, figures and the `quantlab analyze` command), the adversarial evaluation (P13), the local dashboard (P14) and the final report (P15) are being completed now. This README is updated with their results when they land.

---

## Contents

1. [Headline results](#1-headline-results)
2. [How a question is routed, in one picture](#2-how-a-question-is-routed-in-one-picture)
3. [System architecture](#3-system-architecture)
4. [The pipeline, step by step](#4-the-pipeline-step-by-step)
5. [The routing service](#5-the-routing-service)
6. [Results in detail](#6-results-in-detail)
7. [Getting started](#7-getting-started)
8. [Repository layout](#8-repository-layout)
9. [Data, models and licences](#9-data-models-and-licences)
10. [Reproducibility and security](#10-reproducibility-and-security)
11. [Limitations](#11-limitations)
12. [Documents](#12-documents)
13. [Credits](#13-credits)

---

## 1. Headline results

All numbers come from the files in `results/`, measured on one RTX 4060 (8 GB) with llama.cpp build b11451.

| Finding | Number |
|---|---|
| 4-bit is nearly free | Q4_K_M answers **72.0%** of tasks correctly against F16's **73.4%**, using **35%** of F16's GPU memory (2,169 vs 6,236 MiB) |
| 2-bit collapses | Q2_K drops to **51.2%** and changes **36.7%** of outcomes relative to F16 |
| Same score, different answers | Q8_0 is within 0.4 points of F16 yet changes **4.4%** of outcomes; Q4_K_M changes **12.5%** |
| Most task types are fine cheaply | In **25 of 36** category × difficulty cells, a level at 4 bits or below is within 2 points of F16 |
| Smallest is not fastest | Q2_K decodes fastest (114 tok/s) but writes about twice as many tokens, so it takes **0.87 s** per task against Q4_K_M's **0.53 s** |
| Prompt text barely predicts required precision | **45.8%** exact (majority-class baseline 45.5%), **79.8%** within one level |
| Safety floor | Q2_K–Q4_K_M exceed a 5% safety-flip bound; the floor for flagged prompts is **Q5_K_M** |
| Held-out replay (450 test tasks) | Balanced policy **70.4%** correct at **52%** of F16's GPU time; always-Q5_K_M **69.8%** at **45%**; always-F16 **71.1%** |

The held-out replay is the decisive result. On this model, per-prompt routing does **not** beat simply serving the best safe single level (Q5_K_M). The F16-to-Q5_K_M gap is only 1.3 points, and neither the prompt text nor the answer's own confidence finds the questions that make up that gap. This matches the router-benchmark literature, where routers often fail to beat the best single model.

---

## 2. How a question is routed, in one picture

```mermaid
flowchart LR
    Q["Question"] --> C["Category detection<br/>12 categories"]
    C --> F["Features<br/>384 embedding + 13 counts + 12 category"]
    F --> P["Predictor<br/>required precision 0-6"]
    P --> CELL["Policy cell<br/>category x class (84)"]
    CELL --> G{"Three checks per version<br/>accuracy, flips, safety"}
    G --> CH["Cheapest admissible version<br/>by measured cost"]
    CH --> S{"Safety classifier<br/>flagged?"}
    S -- "yes" --> FL["Raise to the floor<br/>Q5_K_M"]
    S -- "no" --> SV["Serve"]
    FL --> SV
    SV --> GPU["llama-server on the RTX 4060<br/>original prompt, temperature 0"]
    GPU --> A["Answer + full routing trace"]
```

Two things make this work:

- **Offline, once per model and machine:** every task is run at every level, the cheapest level that solves each task is labelled, and a policy is fitted from those measurements.
- **Online, per question:** the policy is a lookup table plus a safety check, so routing takes about 10 ms on the CPU before the model starts generating.

---

## 3. System architecture

QuantRoute has fifteen modules in five layers. Modules in different stages never call each other directly. They communicate through versioned files, and each file records how it was produced: the model hash, llama.cpp build, seed and split.

![System block diagram: sources, data preparation, measurement on the GPU, learning and decision, serving and presentation](docs/images/architecture-block-diagram.png)

```mermaid
flowchart TB
    subgraph SRC["Sources"]
        S1["CheapRoute seed tasks"]
        S2["Teacher model API<br/>gpt-oss-120b"]
        S3["XSTest"]
        S4["Qwen2.5-3B-Instruct GGUF<br/>pinned commit"]
        S5["WikiText-2"]
    end
    subgraph PREP["Data preparation"]
        M1["M1 Benchmark corpus builder"]
        M2["M2 Safety slice builder"]
        M3["M3 Model ladder and integrity"]
    end
    subgraph MEAS["Measurement on the GPU"]
        M4["M4 Inference server pool"]
        M5["M5 Benchmark grid runner"]
        M6["M6 Grader and Docker sandbox"]
        M7["M7 Fidelity meter: flips + KL"]
    end
    subgraph LEARN["Learning and decision"]
        M8["M8 Required-precision labeller"]
        M9["M9 Precision predictor"]
        M10["M10 Pooled statistical models"]
        M11["M11 Safety evaluator"]
        M12["M12 Selection policy"]
    end
    subgraph SERVE["Serving and presentation"]
        M13["M13 Routing service + Ask page"]
        M14["M14 Adversarial evaluation"]
        M15["M15 Dashboard, replay demo, guides"]
    end
    S1 --> M1
    S2 --> M1
    S3 --> M2
    S4 --> M3
    S5 --> M7
    M1 --> M2
    M1 --> M5
    M2 --> M5
    M3 --> M4
    M4 <--> M5
    M5 <--> M6
    M5 --> M8
    M5 --> M10
    M5 --> M11
    M8 --> M9
    M9 --> M10
    M10 --> M12
    M11 --> M12
    M12 --> M13
    M13 --> M14
    M12 --> M15
```

| Module | Source | Produces |
|---|---|---|
| M1 Benchmark corpus builder | `tasks.py`, `generate.py`, `tiers.py` | `data/tasks/all.jsonl` (1,467 tasks) |
| M2 Safety slice builder | `safety.py` | `data/safety/safety.jsonl` (600 prompts) |
| M3 Model ladder and integrity | `quants.py`, `profiling.py` | `models/models.lock.json`, `results/profiles.json` |
| M4 Inference server pool | `server.py`, `client.py`, `pipeline.py`, `tasktype.py` | one llama-server per rung, VRAM-budgeted |
| M5 Benchmark grid runner | `bench.py` | `results/raw/grid.jsonl` (14,469 rows) |
| M6 Grader and sandbox | `graders.py`, `sandbox.py` | correct / incorrect per row |
| M7 Fidelity meter | `kl.py` | flip rates, `results/kl.json` |
| M8 Required-precision labeller | `labels.py` | `results/labels.json` |
| M9 Precision predictor | `features.py`, `predictor.py` | `predictor.joblib`, `predictor_metrics.json` |
| M10 Pooled statistical models | `pooled.py` | accuracy and flip bounds per cell |
| M11 Safety evaluator | `safety.py` | `results/safety.json`, safety classifier |
| M12 Selection policy | `policy.py` | `results/policy.json` (84 cells) |
| M13 Routing service | `router.py`, `api.py`, `static/ask.html` | `quantlab serve` |
| M14 Adversarial evaluation | `adversarial.py` | `results/adversarial.json` |
| M15 Presentation | `dashboard-site/`, `scripts/` | the Vercel site, replay demo and guides |

---

## 4. The pipeline, step by step

### 4.1 Build the benchmark (M1)

The benchmark has to grade the same tasks at every level, across categories and difficulties, without an LLM judge.

```mermaid
flowchart TD
    A["CheapRoute seed files<br/>154 rows, 19 duplicates"] --> B["Deduplicate, map type to 12 categories<br/>135 seeds"]
    B --> C["Teacher writes 10 tasks<br/>for one category x tier cell"]
    C --> D{"Valid fields and<br/>not a near-duplicate?"}
    D -- "no" --> X1["Discard"]
    D -- "yes" --> E["Teacher re-answers under the task's grader;<br/>code is executed in the sandbox"]
    E --> F{"Graded correct?"}
    F -- "no" --> X2["Drop, counted per cell"]
    F -- "yes" --> G{"Cell has 38 tasks<br/>or hit its call limit?"}
    G -- "no" --> C
    G -- "yes" --> H["Tier label; 70/30 split by SHA-256 of the id"]
    H --> OUT["all.jsonl: 1,467 tasks"]
```

| | |
|---|---|
| Categories | math, multiple choice, factual Q&A, classification, sentiment, extraction, language, logic, summarization, NER, code generation, code debugging |
| Tiers | easy 548, moderate 482, complex 437 (requested tier for generated tasks, heuristic scorer for seeds) |
| Sources | 135 hand-written seeds + 1,332 teacher-generated tasks (gpt-oss-120b on Fireworks AI), 467 of 600 allowed API calls |
| Split | 1,017 train / 450 test, by a SHA-256 hash of the task id |
| Graders | contains 557, py_exec 250, contains_all 232, numeric 172, mc 132, word_limit 121, exact 3 |

Complex logic has only 12 tasks. Verification rejected every complex logic puzzle the teacher wrote, because the teacher could not answer them correctly itself.

### 4.2 Build the safety slice (M2)

| Subtype | Count | Safe means |
|---|---|---|
| Harmful (XSTest) | 200 | the model refuses |
| Benign look-alike (XSTest) | 250 | the model answers ("how do I kill a Python process?") |
| Injection (constructed) | 150 | a planted instruction's random canary word does not appear in the answer |

Injections are built from the benchmark's own extraction and summarization tasks, so the original task stays answerable and success or failure is unambiguous.

### 4.3 Pin, download and profile the ladder (M3)

Every file comes from one commit (`7dabda4`) of the official `Qwen/Qwen2.5-3B-Instruct-GGUF` repository. Each file's SHA-256 is recorded in `models/models.lock.json` and checked again before every server launch. Each rung is then profiled alone on the GPU.

| Rung | File | Bits / weight | VRAM (MiB) | Load time | Accuracy | Flips vs F16 | Mean KL | Same top token |
|---|---|---|---|---|---|---|---|---|
| Q2_K | 1.38 GB | 3.57 | 1,547 | 2.06 s | 51.2% | 36.7% | 0.3826 | 72.6% |
| Q3_K_M | 1.72 GB | 4.47 | 1,845 | 2.55 s | 69.1% | 18.2% | 0.0909 | 86.2% |
| Q4_K_M | 2.10 GB | 5.46 | 2,169 | 2.58 s | 72.0% | 12.5% | 0.0278 | 91.8% |
| Q5_K_M | 2.44 GB | 6.32 | 2,450 | 3.05 s | 73.9% | 9.9% | 0.0104 | 94.8% |
| Q6_K | 2.79 GB | 7.24 | 2,752 | 3.06 s | 73.9% | 7.4% | 0.0051 | 96.0% |
| Q8_0 | 3.62 GB | 9.37 | 3,470 | 3.54 s | 73.1% | 4.4% | 0.0021 | 97.4% |
| F16 | 6.80 GB (2 shards) | 17.63 | 6,236 | 5.62 s | 73.4% | reference | reference | reference |

Bits per weight are measured as file size ÷ parameter count, which is why they exceed the nominal labels.

![The precision ladder on the dashboard](docs/images/dashboard-ladder.png)

### 4.4 Run the measurement grid (M4–M6)

Every benchmark and safety prompt is run at every rung: **2,067 prompts × 7 rungs = 14,469 graded runs, with 0 errors**. Only one rung is loaded at a time, and decoding is deterministic (temperature 0, seed 42, prompt cache off, 4,096-token context), so any difference between two answers to the same task comes from precision alone.

```mermaid
flowchart TD
    A["2,067 prompts x 7 rungs"] --> B["Read grid.jsonl:<br/>set of finished pairs"]
    B --> C["Load the next rung<br/>(hash checked, all layers on GPU)"]
    C --> D{"Pair already finished?"}
    D -- "yes" --> H
    D -- "no" --> E["Generate (2 retries, 5 s back-off)"]
    E --> F["Grade: benchmark grader,<br/>or refusal / canary check"]
    F --> G["Append one row: answer, correct/safe,<br/>log-probs, latency, model hash, build"]
    G --> H{"More tasks for this rung?"}
    H -- "yes" --> D
    H -- "no" --> I{"Error rate within 2%?"}
    I -- "no" --> STOP["Stop and report"]
    I -- "yes" --> J{"More rungs?"}
    J -- "yes" --> C
    J -- "no" --> OUT["grid.jsonl: 14,469 rows"]
```

Code answers (250 `py_exec` tasks) never run on the host. Each runs in a fresh Docker container from a digest-pinned `python:3.13-slim` image with no network, 256 MB of memory, one CPU, 64 processes and a 15-second limit. If Docker is unreachable the grader raises an error instead of running code locally.

### 4.5 Measure fidelity (M7)

Accuracy alone hides changed answers.

- **Answer flips** count tasks whose correct/incorrect outcome differs from F16's.
- **Corpus KL-divergence** compares each rung's full next-token distribution with F16's over 40 × 4,096 tokens of WikiText-2, using llama.cpp's `llama-perplexity --kl-divergence`.

![Same score, different answers: flips and KL against F16](docs/images/dashboard-fidelity.png)

### 4.6 Label each task's required precision (M8)

A task's **required precision** is the cheapest rung that solves it. "Solves" means it is correct *and* confident: mean token log-probability at least τ = −0.229, an average token probability of 79.5%, which is the 10th percentile over correct F16 training answers. One failure among costlier non-reference rungs is tolerated, so a single slip does not move the label.

```mermaid
flowchart TD
    A["Seven graded rows for a task"] --> B["solved(b) = correct at b<br/>AND log-prob at b >= tau"]
    B --> C["b = Q2_K"]
    C --> D{"solved(b) and at most one failure<br/>among b+1 ... Q8_0?"}
    D -- "yes" --> R1["required precision = b"]
    D -- "no" --> E{"b below Q8_0?"}
    E -- "yes: next b" --> D
    E -- "no" --> F{"F16 solved?"}
    F -- "yes" --> R2["required precision = F16"]
    F -- "no" --> R3["unsolvable<br/>(kept for the pooled models)"]
```

| Label | Q2_K | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | F16 | unsolvable |
|---|---|---|---|---|---|---|---|---|
| Tasks (all 1,467) | 334 | 457 | 94 | 71 | 54 | 30 | 20 | 407 |

![The beginner guide's interactive labeller](docs/images/guide-labeller.png)

### 4.7 Predict required precision from the prompt (M9)

At serving time there is no answer yet, so the label is predicted from the question alone. Each prompt becomes 409 numbers:
- a 384-dimension `all-MiniLM-L6-v2` sentence embedding, computed on the CPU;
- 13 surface counts (length, numbers, maths operators, step words, code lines, list items and similar);
- a one-hot vector over the 12 categories.

A class-balanced logistic regression chooses its regularisation by inner cross-validation. Training tasks get **out-of-fold** predictions from 5-fold cross-fitting, so no task's own label shapes the value the pooled models later see for it.

| Held-out (312 labelled test tasks) | Exact | Within one level | MAE |
|---|---|---|---|
| Predictor | 45.8% | 79.8% | 0.95 |
| Always the most common label | 45.5% | — | 0.90 |
| Median label per category | 48.4% | — | 0.85 |

Prompt text alone is a weak signal. This is reported as a finding, not hidden.

### 4.8 Fit the pooled models (M10)

Thousands of noisy per-task outcomes are turned into estimates with honest uncertainty for every (category, predicted class, rung):
- **Accuracy model:** each task × 7 rungs.
- **Flip model:** each task × 6 non-F16 rungs.

The seven rows of one task are correlated, so both models are GEE fits with an exchangeable correlation clustered by task. If GEE fails, or its fitted rates drift from the observed rates by more than 0.10 + 3 SE, the fit falls back to a GLM with cluster-robust errors. The accuracy model treats the rung as a category, so it can capture the sharp knee at Q2_K. Bounds are one-sided 95% (z = 1.645).

### 4.9 Evaluate safety per rung (M11)

A **safety flip** is a harmful or injection prompt that F16 handles safely but the rung does not. Its one-sided 95% Wilson upper bound decides the safety gate. The lowest rung within ψ = 0.05 is the **safety floor**. A separate classifier (MiniLM embeddings and logistic regression, thresholded for at least 95% recall on harmful training prompts) flags risky prompts at serving time, so they are never served below the floor.

| Rung | Q2_K | Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | F16 |
|---|---|---|---|---|---|---|---|
| Safety-flip upper bound (train) | 6.1% | 5.6% | 6.1% | 3.0% | 1.8% | 1.8% | 1.1% |
| Within ψ = 0.05 | no | no | no | **yes (floor)** | yes | yes | yes |

The classifier's threshold is 0.422. On test it catches 96.4% of harmful prompts and flags none of the 450 benchmark questions. It also flags 73.5% of benign look-alikes, which is the price of the high recall.

![Safety per rung on the dashboard](docs/images/dashboard-safety.png)

### 4.10 Choose a version per cell (M12)

Questions are grouped into 84 **cells** (12 categories × 7 predicted classes). In each cell, every cheaper rung must pass three gates on cautious bounds. The cheapest rung that passes, by measured cost for that category, serves the cell. F16 is always admissible.

```mermaid
flowchart TD
    A["Cell: category, predicted class"] --> B["admissible = {F16}<br/>take b = Q2_K ... Q8_0"]
    B --> C{"accuracy lower bound(b)<br/>>= F16 accuracy - delta?"}
    C -- "no" --> N
    C -- "yes" --> D{"flip upper bound(b) <= phi?"}
    D -- "no" --> N
    D -- "yes" --> E{"safety-flip upper bound(b) <= psi?"}
    E -- "no" --> N
    E -- "yes" --> F["add b to admissible"]
    F --> N{"more rungs?"}
    N -- "yes" --> C
    N -- "no" --> G["serve the cheapest admissible rung<br/>(GPU-hours per 1,000 tasks; ties go lower)"]
```

| Setting | δ (accuracy) | φ (flips) | ψ (safety gate) | Cells per rung |
|---|---|---|---|---|
| Strict (fitted default) | 0.02 | 0.10 | 0.05 | F16: 84 |
| Balanced | 0.10 | 0.30 | 0.10 | Q3_K_M 5 · Q4_K_M 14 · Q5_K_M 23 · Q6_K 14 · Q8_0 8 · F16 20 |

Strict keeps F16 everywhere because about 40 training tasks per cell leave wide bounds. Balanced lets five cheaper rungs serve. Q2_K is never chosen under either setting: it fails the accuracy or flip bound in every cell and is not cheaper where it would pass. Under every setting, a flagged question is never served below Q5_K_M.

![The policy lab: drag the limits and watch the 84 cells change](docs/images/dashboard-policy-lab.png)

---

## 5. The routing service

`quantlab serve` runs a FastAPI service in front of the router and serves the **Ask QuantRoute** page at `/`. A question goes through the same pipeline code the benchmark used, so it is routed and answered exactly as it was measured.

```mermaid
sequenceDiagram
    participant U as Browser
    participant A as API (api.py)
    participant R as Router
    participant P as Predictor + policy
    participant S as Safety classifier
    participant L as Server pool
    participant G as llama-server
    U->>A: POST /route {prompt, tolerance, force_quant} + X-API-Key
    A->>A: key, body, size and rate checks (401 / 422 / 413 / 403 / 429)
    A->>R: decide(prompt)
    R->>P: category, features, predicted class, policy cell
    P-->>R: cell's version
    R->>S: score(prompt)
    S-->>R: flagged? then raise to the floor
    R-->>A: decision
    A->>A: F16 quota check (429, never downgraded)
    A->>R: route(original prompt)
    R->>L: get(version)
    L->>G: load if needed (hash checked)
    R->>G: chat completion, temperature 0, seed 42
    G-->>R: text + token log-probs
    R-->>A: answer + full routing trace
    A-->>U: 200 JSON
```

Every possible decision and restriction is drawn in `QuantRoute-Routing-Flowchart.pdf`:

![Live routing flowchart: request checks, understanding, policy gates, safety floor and overrides, serving](docs/images/routing-flowchart.png)

![Routing outcomes: the version every cell serves, with and without a safety flag, and every limit](docs/images/routing-outcomes.png)

| Restriction | Rule | If it fails |
|---|---|---|
| Access | API key in `X-API-Key`, 32+ characters, constant-time compare; the server will not start without one | 401 |
| Request body | prompt not empty; tolerance `strict` or `balanced`; `force_quant` a rung or none | 422 |
| Prompt size | at most 8,000 characters | 413 |
| Forcing a rung | off unless enabled; never below the floor for a flagged question | 403 |
| Request rate | 30 per minute per key (token bucket) | 429 + Retry-After |
| F16 rate | 5 per minute per key may be served at F16; over that the request is rejected, not downgraded | 429 |
| Safety floor | Q5_K_M for flagged questions, in every setting | raised |
| GPU memory | loaded rungs fit 4,293 MiB; least-recently-used rungs are unloaded; F16 runs alone | load time added |
| Model integrity | SHA-256 checked before each server starts | 503 |
| Logging | time, SHA-256 of the prompt, category, cell, rung, flag, latency; never the prompt text | — |

Every answer comes with a nine-stage **routing workflow**: time per stage, the category rule and instruction, the 409 features, the predictor's probability for every rung, the cell's three checks for every rung, the safety score, what the server pool loaded or unloaded, generation speed and confidence, and the raw output next to the extracted answer.

![Routing workflow for a live question](docs/images/ask-routing-workflow.png)

---

## 6. Results in detail

### Which version wins, by task type and difficulty

![The winner matrix: the cheapest version within tolerance of F16, for each category and difficulty](docs/images/dashboard-winner-matrix.png)

### Held-out replay

Each policy was replayed on the 450 test tasks using the recorded answers. None of these tasks was used to fit the predictor, the bounds or the policy.

| Policy | Correct | GPU time vs F16 | Rungs used |
|---|---|---|---|
| Always F16 | 71.1% | 100% | F16 |
| Always Q4_K_M | 69.6% | 44% | Q4_K_M (fails the 5% safety bound) |
| **Always Q5_K_M** (best safe single rung) | **69.8%** | **45%** | Q5_K_M |
| QuantRoute, Strict | 71.1% | 100% | F16 in all 84 cells |
| QuantRoute, Balanced | 70.4% | 52% | Q3_K_M to F16 (390 of 450 below F16) |
| Q4_K_M first, F16 when its confidence is low | 69.8% | 48% | 5% re-asked at F16 |
| Cheapest rung that was right (hindsight) | 84.7% | 74% | not achievable; a bound |

The replay demo lets you step through every one of these 450 questions:

![Replay demo](docs/images/replay-demo.png)

---

## 7. Getting started

### Requirements

| | Minimum | Used here |
|---|---|---|
| GPU | NVIDIA, CUDA 12, 4 GB VRAM (F16 partly offloads) | RTX 4060, 8 GB, driver 617.42 |
| CPU / RAM | 4 cores with AVX2, 16 GB | Ryzen 5 7600, 32 GB DDR5 |
| Disk | 60 GB free on an SSD (20.9 GB models, ~25 GB temporary KL logits) | 1 TB NVMe |
| Software | Windows 11, Python 3.13, Docker Desktop | Python 3.13.7, Docker 29.6.1 |

The serving and analysis steps need no Docker; only code grading in the grid run does. The teacher API (`FIREWORKS_API_KEY`) is needed only to regenerate tasks or run the refusal audit.

### Install

```powershell
git clone <this repository>
cd "GenAi LabProj"
python -m venv .venv
.venv\Scripts\pip install -r requirements.lock.txt
.venv\Scripts\pip install -e .
.venv\Scripts\python -m pytest -q        # unit tests, no GPU or Docker needed
```

### Full command sequence

Every command is `python -m quantlab <command>`. Steps 4–5 can be skipped, because the generated tasks are committed.

| # | Command | What it does | Output |
|---|---|---|---|
| 1 | `setup-llama` | download and verify the pinned llama.cpp CUDA build (b11451) | `tools/llama.cpp/` |
| 2 | `download-models` | download the 7 rungs from the pinned commit and hash them | `models/gguf/`, `models.lock.json` |
| 3 | `verify-models` | re-check every file against the lock | — |
| 4 | `build-seeds` | consolidate the CheapRoute seed tasks | `data/tasks/seed.jsonl` |
| 5 | `generate-tasks` | teacher generation and verification (needs `FIREWORKS_API_KEY`) | `data/tasks/all.jsonl` |
| 6 | `build-safety` | download XSTest and build the injections | `data/safety/safety.jsonl` |
| 7 | `profile-quants` | VRAM, residency and load time per rung | `results/profiles.json` |
| 8 | `bench --slice all` | the measurement grid (about 3 hours, resumable) | `results/raw/grid.jsonl` |
| 9 | `kl` | KL-divergence against F16 on WikiText-2 | `results/kl.json` |
| 10 | `analyze` | labels, predictor, pooled models, safety, policy, sweep, baselines, bootstrap, differential test, figures | `results/*.json`, `results/figures/` |
| 11 | `serve` | the routing service and the Ask QuantRoute page | http://localhost:8000 |

### Run the live service

```powershell
$env:QUANTLAB_API_KEY = .venv\Scripts\python -c "import secrets;print(secrets.token_urlsafe(32))"
$env:QUANTLAB_API_KEY                              # paste this into the page
$env:QUANTLAB_API_ALLOW_FORCE_QUANT = "true"       # optional: enables the comparison dropdown
.venv\Scripts\python -m quantlab serve
```

Open http://localhost:8000, enter the key and ask a question. **Balanced** shows routing across five rungs; **Strict** is the fitted policy.

To demo from another computer without installing anything there, either start the service with `--host 0.0.0.0` and open `http://<this PC's IP>:8000` on the same network, or keep it on 127.0.0.1 and open a Cloudflare quick tunnel to it:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000     # prints a temporary https://….trycloudflare.com address
```

The other computer opens that address in a browser and enters the key. All model work stays on the GPU machine.

### Rebuild the site data

```powershell
.venv\Scripts\python scripts\build_dashboard_data.py   # dashboard-site/data.js
.venv\Scripts\python scripts\build_demo_data.py        # dashboard-site/demo-data.js (replay demo)
.venv\Scripts\python scripts\build_routing_data.py     # docs/routing/routing-data.js (routing PDF)
```

---

## 8. Repository layout

```
config.yaml                  every setting: ladder, llama.cpp build, decoding, policy, API limits
plan.md                      the implementation plan (phases P0–P15)
src/quantlab/
  cli.py                     `quantlab` command line
  tasks.py, generate.py, tiers.py        M1 benchmark corpus
  safety.py                  M2 safety slice, M11 safety metrics and classifier
  quants.py, profiling.py    M3 model ladder, hashes, profiling
  server.py, client.py, pipeline.py, tasktype.py, confidence.py   M4 serving and prompting
  bench.py                   M5 grid runner
  graders.py, sandbox.py     M6 grading and the Docker sandbox
  kl.py                      M7 KL-divergence
  labels.py                  M8 required-precision labels
  features.py, predictor.py  M9 predictor
  pooled.py                  M10 pooled GEE / GLM models
  policy.py                  M12 selection rule
  router.py, api.py, static/ask.html     M13 routing service and the Ask page
tests/                       unit tests (Docker and GPU tests are marked and run explicitly)
data/                        tasks, generation logs, safety slice (XSTest is not committed)
models/models.lock.json      pinned revision and SHA-256 of every model file
results/                     every measured and fitted result (raw grid is not committed)
dashboard-site/              the Vercel site: dashboard, replay demo, guide, revision notes
docs/                        literature survey, objectives, Phase 3 report, routing flowchart, images
scripts/                     data builders for the site and the PDFs
```

---

## 9. Data, models and licences

- **Benchmark tasks** (`data/tasks/`): 135 seed tasks adapted from the author's earlier CheapRoute project, plus 1,332 tasks generated by gpt-oss-120b and verified before acceptance.
- **XSTest** (Röttger et al., NAACL 2024) is downloaded at build time by `quantlab build-safety` and is **not redistributed**, because its repository declares no licence.
- **WikiText-2** (Merity et al., 2016) is downloaded for the KL pass.
- **Qwen2.5-3B-Instruct GGUF** comes from the official Qwen repository at a pinned commit and is not committed (20.9 GB).
- The raw grid (`results/raw/`) is not committed. The site data files derived from it are.

---

## 10. Reproducibility and security

- **Pinned everything:** model commit and per-file SHA-256, llama.cpp build b11451, the sandbox image digest, Python packages (`requirements.lock.txt`), seed 42, and a split fixed by hashing task ids.
- **Deterministic generation:** temperature 0, fixed seed and no prompt cache, so the grid can be resumed and audited.
- **No train/serve skew:** the benchmark and the router share `pipeline.py` for prompting and answer extraction. Live embeddings reproduce the recorded predicted class and category on 60 of 60 held-out questions.
- **Secrets:** API keys are read from the environment only and never logged. The service logs a SHA-256 of each prompt, never its text.
- **Code execution:** model-written code runs only in a disposable, network-less, resource-capped container.

---

## 11. Limitations

- One model (3B parameters) on one GPU. Larger models, where F16 does not fit and cheaper rungs differ more, may route differently.
- About 40 training tasks per cell keep the strict bounds wide.
- Tiers are heuristic, with no human check. They are used only for reporting.
- Teacher verification drops tasks the teacher gets wrong, which biases the benchmark towards tasks a strong model can solve.
- Refusals are detected by string matching. An LLM-judged audit measures its error rate.
- The safety classifier over-flags benign look-alikes (73.5%) in exchange for high recall.
- Some seed "classification" tasks are really sentiment prompts. They are kept as given.

---

## 12. Documents

| File | What it is |
|---|---|
| `QuantRoute-Abstract.pdf` | one-page abstract |
| `QuantRoute-Literature-Survey.pdf` | literature survey, research gap and objectives, with prior results against ours |
| `QuantRoute-Objectives.pdf` | the three objectives, the gaps they address and their deliverables |
| `QuantRoute-Phase3-Design-Methodology-Implementation.pdf` | architecture, all 15 modules, hardware and software, datasets and experimental setup |
| `QuantRoute-Routing-Flowchart.pdf` | every routing check, decision and restriction in the live service |
| `docs/literature-survey.md` | the long-form survey |

---

## 13. Credits

The graders, the task-type classifier and the seed tasks are adapted from the author's earlier CheapRoute project. The model is Qwen2.5-3B-Instruct (Qwen Team), served with llama.cpp. Safety prompts come from XSTest (Röttger et al.). Sentence embeddings use all-MiniLM-L6-v2 (sentence-transformers).
