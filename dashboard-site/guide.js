(function () {
  "use strict";

  const D = window.QL;
  const R = D.rungs;
  const LADDER = D.meta.ladder;
  const CATS = D.meta.categories;
  const $ = (id) => document.getElementById(id);
  const pct = (x, d = 0) => (x == null || isNaN(x) ? "—" : (x * 100).toFixed(d) + "%");
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");

  // ------------------------------------------------------------ derived numbers
  const reqTotals = (() => {
    const t = { solved: 0, total: 0, q2: 0, upToQ4: 0, unsolvable: 0 };
    CATS.forEach((c) => {
      const h = D.req_hist[c];
      for (let k = 0; k < 7; k++) t.solved += h[String(k)];
      t.q2 += h["0"]; t.upToQ4 += h["0"] + h["1"] + h["2"];
      t.unsolvable += h.unsolvable;
    });
    t.total = t.solved + t.unsolvable;
    return t;
  })();
  const winnerCount = (() => {
    let low = 0, n = 0;
    CATS.forEach((c) => D.meta.tiers.forEach((t) => {
      const cell = D.cat_tier[c][t];
      const w = LADDER.find((r) => cell[r].acc >= cell.F16.acc - 0.02 - 1e-9);
      n++; if (LADDER.indexOf(w) <= 2) low++;
    }));
    return { low, n };
  })();
  const lowSupport = Object.values(D.policy.cells).filter((c) => c.low_support).length;
  const gb = (mb) => (mb / 1024).toFixed(1);

  // ------------------------------------------------------------ TL;DR
  $("tldr-found").textContent = "Q4 is the sweet spot";
  $("tldr-found-p").innerHTML = `Q4_K_M scores ${pct(R.Q4_K_M.accuracy)} vs F16's ${pct(R.F16.accuracy)} using ${pct(R.Q4_K_M.vram_mb / R.F16.vram_mb)} of the memory; Q2_K collapses to ${pct(R.Q2_K.accuracy)}. But even rungs that match F16's score change ${pct(R.Q8_0.flip_rate)}&ndash;${pct(R.Q4_K_M.flip_rate)} of answers, and guessing a task's needs from its text is hard.`;

  // ------------------------------------------------------------ progress
  const SECTIONS = ["tldr", "flow", "ideas", "numbers", "viva", "bugs", "quiz"];
  let done = [];
  try { done = JSON.parse(localStorage.getItem("ql-guide-done") || "[]"); } catch (e) { done = []; }
  function renderProgress() {
    $("pbar").style.width = (done.length / SECTIONS.length) * 100 + "%";
    $("ptext").textContent = `${done.length}/${SECTIONS.length}`;
    document.querySelectorAll(".done-btn").forEach((b) => {
      const on = done.includes(b.dataset.done);
      b.classList.toggle("is-done", on);
      b.textContent = on ? "Learned ✓" : "Mark learned";
    });
  }
  document.querySelectorAll(".done-btn").forEach((b) => b.addEventListener("click", () => {
    const id = b.dataset.done;
    done = done.includes(id) ? done.filter((x) => x !== id) : [...done, id];
    try { localStorage.setItem("ql-guide-done", JSON.stringify(done)); } catch (e) { /* storage unavailable: progress lasts this visit */ }
    renderProgress();
  }));
  renderProgress();

  // ------------------------------------------------------------ flowchart content
  const LANES = {
    data: { label: "DATA", color: "#ffc832" },
    measure: { label: "MEASURE", color: "#2ed3a0" },
    analyze: { label: "ANALYZE", color: "#bcd3ff" },
    decide: { label: "DECIDE & SERVE", color: "#ffd9c7" },
  };

  const INFO = {
    tasks: { what: `A benchmark of ${D.meta.n_bench_tasks.toLocaleString()} graded tasks across 12 categories (math, code, extraction, logic&hellip;) at three difficulty tiers, plus ${D.meta.n_safety_tasks} safety prompts.`, why: "You can't decide precision per task without measuring many kinds of task.", out: "data/tasks/all.jsonl, data/safety/safety.jsonl", key: (D.meta.n_bench_tasks + D.meta.n_safety_tasks).toLocaleString(), keyLabel: "prompts in total" },
    models: { what: `Qwen2.5-3B-Instruct in seven precisions, from Q2_K (${R.Q2_K.bpw.toFixed(1)} bits per weight) to F16 (the 16-bit reference), all from one official repo at one pinned commit, every file SHA-256 verified.`, why: "Same model, different precision &mdash; so any behaviour change is caused by quantization alone.", out: "models/models.lock.json, results/profiles.json", key: `${R.Q2_K.bpw.toFixed(1)} → ${R.F16.bpw.toFixed(1)}`, keyLabel: "bits per weight" },
    grid: { what: "Every prompt on every model with greedy decoding (temperature 0, fixed seed), recording the answer, whether it's right, its token log-probabilities and latency. Code answers are executed inside a locked-down Docker container. The run is resumable &mdash; it survived a crash and picked up where it stopped.", why: "This grid is the ground truth every later number is computed from.", out: "results/raw/grid.jsonl", key: D.meta.n_grid_rows.toLocaleString(), keyLabel: "graded runs, 0 errors" },
    labels: { what: `For each task, <b>required precision</b> = the cheapest rung that solves it <em>confidently</em> (correct, and mean token log-prob &ge; &tau; = ${D.meta.tau.toFixed(3)}), allowing at most one failure among the costlier rungs. If even F16 fails, it's "unsolvable".`, why: "Turns seven right/wrong results into one target to learn: how much precision does this task need?", out: "results/labels.json", key: pct(reqTotals.q2 / reqTotals.solved), keyLabel: "of solvable tasks already solved at Q2_K" },
    predictor: { what: "A classifier that guesses required precision from the prompt alone: a 384-number MiniLM sentence embedding + 13 surface counts (length, numbers, math operators, code lines&hellip;) + the predicted category. Cross-fitted, so no task's own label leaks into its prediction.", why: "At serving time you only have the prompt &mdash; you can't run all seven models first.", out: "results/predictor_metrics.json", key: pct(D.predictor.test_exact_accuracy, 1), keyLabel: `exact class (${pct(D.predictor.test_within_one_accuracy)} within one rung)` },
    policy: { what: "For each (category, predicted class) cell, a rung is allowed only if <b>(1)</b> its accuracy lower bound is within &delta; of F16, <b>(2)</b> its answer-flip upper bound is &le; &phi;, and <b>(3)</b> its safety-flip upper bound is &le; &psi;. The cheapest allowed rung wins; F16 is always allowed.", why: "Accuracy alone hides answer churn; stability alone could pick a cheap rung that's stably wrong; neither covers safety.", out: "results/policy.json", key: "84", keyLabel: "cells (12 categories × 7 classes)" },
    router: { what: "A hardened API: takes a prompt, predicts its cell, applies the policy plus a safety floor for prompts the safety classifier flags, and answers with the chosen model. Then it gets attacked: padding to force expensive F16 (cost inflation) and disguises to push harmful prompts onto weak rungs (downgrade).", why: "A router is an attack surface; it must be tested adversarially, not just on clean prompts.", out: "planned: API service, results/adversarial.json", key: "next", keyLabel: "phase of the project" },

    seed: { what: "135 hand-written tasks from an earlier project, deduplicated from 154 rows across four files. Each carries its own grader: numeric, exact, contains, multiple-choice, word-limit, or run-the-code.", why: "A trusted, human-made core.", out: "data/tasks/seed.jsonl", key: "135", keyLabel: "seed tasks" },
    gen: { what: "A large teacher model (gpt-oss-120b) wrote new tasks for 12 categories × 3 tiers (target 38 per cell), then answered them itself; only tasks it answered correctly with the task's own grader were kept. Hard cap of 600 API calls.", why: "Scales the benchmark to ~1,500 tasks with balanced coverage.", out: "data/tasks/generated.jsonl", key: (D.meta.n_bench_tasks - 135).toLocaleString(), keyLabel: "generated tasks kept" },
    bench: { what: "Seeds + generated tasks, each with a difficulty tier (as requested for generated tasks, from a fitted heuristic for seeds) and a train/test split decided by hashing the task id with SHA-256 &mdash; deterministic, no peeking.", why: "Everything is fit on train and judged on test.", out: "data/tasks/all.jsonl", key: "70 / 30", keyLabel: "train / test split" },
    safety: { what: "200 harmful prompts (should refuse) and 250 harmless look-alikes (should answer) from XSTest, plus 150 prompt-injection traps: a hidden instruction carrying a unique canary word, planted inside an extraction or summarization passage.", why: "Low-bit quantization can erode safety while accuracy looks fine.", out: "data/safety/safety.jsonl", key: String(D.meta.n_safety_tasks), keyLabel: "safety prompts" },
    pinned: { what: "Seven GGUF files from the official repo at one commit, run by llama.cpp build b11451. Hashes are recorded at download and re-checked before every server launch.", why: "Quantized files can be weaponised to misbehave only when quantized; pinning makes tampering detectable and results reproducible.", out: "models/models.lock.json", key: "7", keyLabel: "rungs, 8 files" },
    profile: { what: "Each rung loaded three times on the RTX 4060: VRAM footprint (fresh baseline before every load), load time, layers on GPU, bits per weight.", why: "Cost must be measured on the real machine, not guessed from file size.", out: "results/profiles.json", key: `${gb(R.Q2_K.vram_mb)} → ${gb(R.F16.vram_mb)} GB`, keyLabel: "VRAM per rung" },
    kl: { what: "llama-perplexity records F16's next-token probabilities on WikiText-2, then measures how far each rung drifts (KL divergence) and how often its top token agrees.", why: "A task-independent fidelity signal &mdash; and a sanity check: it must shrink as bits grow. It does, strictly.", out: "results/kl.json", key: `${R.Q2_K.kld.toFixed(2)} → ${R.Q8_0.kld.toFixed(3)}`, keyLabel: "KL, Q2_K → Q8_0" },
    features: { what: "A 384-number sentence embedding of the prompt (all-MiniLM-L6-v2, run on CPU) plus 13 counts: characters, words, sentences, numbers, biggest number, math operators, conjunctions, step words, question marks, has-code, code lines, quotes, list items.", why: "Exactly what the predictor gets to see: the prompt, nothing else.", out: "results/raw/embeddings.npy", key: "384 + 13", keyLabel: "features + category" },
    pooled: { what: `Two logistic models across all rungs: P(correct) and P(answer flips vs F16), from category, predicted class and log bits-per-parameter. GEE treats a task's rows as correlated. ${lowSupport} of 84 cells have under 5 tasks and fall back to a category-only model; a calibration check rejects any fit whose predictions don't match observed rates.`, why: "Gives every (category, class, rung) an accuracy and flip estimate with honest confidence bounds.", out: "results/pooled_models.json", key: `${lowSupport} / 84`, keyLabel: "cells using the fallback" },
    safetym: { what: "Per rung: directional safety flips (F16 behaved safely, this rung didn't) with a Wilson upper bound; refusal, over-refusal and injection resistance. Plus a safety classifier (embedding → logistic) tuned to catch 95% of harmful prompts.", why: "Feeds the third policy condition and the safety floor.", out: "results/safety.json", key: D.policy.safety_floor, keyLabel: "safety floor at ψ = 0.05" },
    matrix: { what: "Straight from the grid, no model: for each category and difficulty tier, the cheapest rung whose measured accuracy is within 2 points of F16.", why: "A model-free reality check on the statistical policy &mdash; and it disagrees, which is itself a finding.", out: "dashboard, section 03", key: `${winnerCount.low} / ${winnerCount.n}`, keyLabel: "cells fine at 4-bit or less" },
    dash: { what: "The interactive dashboard and this guide, built over the same result files &mdash; including a live re-implementation of the selection rule you can tune.", why: "Results nobody can read don't count.", out: "dashboard-site/", key: "live", keyLabel: "on Vercel" },
    attack: { what: "Cost-inflation attacks (padding, stuffing, greedy suffix gadgets) that push prompts to F16, and downgrade attacks that push harmful prompts to weak rungs &mdash; each measured with and without defences (safety floor, input normalisation, F16 rate limits).", why: "Routers can be steered by query-independent gadgets that perplexity filters don't catch.", out: "planned: results/adversarial.json", key: "next", keyLabel: "phase of the project" },
  };

  const FLOWS = {
    simple: {
      layout: { w: 192, h: 76, colStep: 234, rowStep: 124, x0: 20, y0: 30, lanes: false },
      nodes: [
        { id: "tasks", col: 0, row: 0, lane: "data", title: "Tasks", sub: `${D.meta.n_bench_tasks.toLocaleString()} graded + ${D.meta.n_safety_tasks} safety` },
        { id: "models", col: 0, row: 1, lane: "data", title: "7 quantized models", sub: "Q2_K → F16, one repo" },
        { id: "grid", col: 1, row: 0.5, lane: "measure", title: "Run everything", sub: `${D.meta.n_grid_rows.toLocaleString()} graded runs` },
        { id: "labels", col: 2, row: 0, lane: "analyze", title: "Label each task", sub: "its cheapest working rung" },
        { id: "predictor", col: 2, row: 1, lane: "analyze", title: "Predict from prompt", sub: "embeddings + 13 features" },
        { id: "policy", col: 3, row: 0.5, lane: "decide", title: "Pick cheapest safe rung", sub: "3 checks per rung" },
        { id: "router", col: 4, row: 0.5, lane: "decide", title: "Route live prompts", sub: "planned: API + attacks", planned: true },
      ],
      edges: [["tasks", "grid"], ["models", "grid"], ["grid", "labels"], ["labels", "predictor"], ["grid", "policy"], ["predictor", "policy"], ["policy", "router"]],
    },
    detailed: {
      layout: { w: 250, h: 70, colStep: 300, rowStep: 104, x0: 20, y0: 64, lanes: true },
      nodes: [
        { id: "seed", col: 0, row: 0, lane: "data", title: "Seed tasks", sub: "154 rows → 135 unique" },
        { id: "gen", col: 0, row: 1, lane: "data", title: "Teacher generation", sub: "gpt-oss-120b, self-verified" },
        { id: "bench", col: 0, row: 2, lane: "data", title: "Bench set", sub: `${D.meta.n_bench_tasks.toLocaleString()} tasks · tiers · 70/30` },
        { id: "safety", col: 0, row: 3, lane: "data", title: "Safety slice", sub: "XSTest 450 + 150 injections" },
        { id: "pinned", col: 1, row: 0, lane: "measure", title: "Pinned models", sub: "llama.cpp b11451 · SHA-256" },
        { id: "profile", col: 1, row: 1, lane: "measure", title: "Profiling", sub: "VRAM · load time · bits/weight" },
        { id: "grid", col: 1, row: 2, lane: "measure", title: "Benchmark grid", sub: `${D.meta.n_grid_rows.toLocaleString()} runs · Docker grading` },
        { id: "kl", col: 1, row: 3, lane: "measure", title: "Corpus KL", sub: "vs F16 on WikiText-2" },
        { id: "labels", col: 2, row: 0, lane: "analyze", title: "Labels", sub: "τ + required precision" },
        { id: "features", col: 2, row: 1, lane: "analyze", title: "Features", sub: "MiniLM 384-d + 13 counts" },
        { id: "predictor", col: 2, row: 2, lane: "analyze", title: "Predictor", sub: "cross-fitted, 5 folds" },
        { id: "pooled", col: 2, row: 3, lane: "analyze", title: "Pooled models", sub: "GEE · clustered by task" },
        { id: "safetym", col: 2, row: 4, lane: "analyze", title: "Safety metrics", sub: "flip bounds + classifier" },
        { id: "matrix", col: 2, row: 5, lane: "analyze", title: "Winner matrix", sub: "category × tier, model-free" },
        { id: "dash", col: 3, row: 1, lane: "decide", title: "Dashboard + guide", sub: "you are here" },
        { id: "policy", col: 3, row: 3, lane: "decide", title: "Selection policy", sub: "3 checks, cheapest wins" },
        { id: "router", col: 3, row: 4, lane: "decide", title: "Router + API", sub: "planned", planned: true },
        { id: "attack", col: 3, row: 5, lane: "decide", title: "Adversarial eval", sub: "planned", planned: true },
      ],
      edges: [
        ["seed", "bench"], ["gen", "bench"], ["bench", "safety"], ["bench", "grid"], ["safety", "grid"],
        ["pinned", "profile"], ["profile", "grid"], ["pinned", "kl"],
        ["grid", "labels"], ["labels", "predictor"], ["features", "predictor"], ["predictor", "pooled"],
        ["grid", "pooled"], ["grid", "safetym"], ["grid", "matrix"],
        ["pooled", "policy"], ["safetym", "policy"], ["policy", "router"], ["router", "attack"],
        ["matrix", "dash"], ["policy", "dash"],
      ],
    },
  };

  // ------------------------------------------------------------ flowchart engine
  const svg = $("fc");
  let mode = "simple", selected = null;

  function geom(n, L) {
    return { x: L.x0 + n.col * L.colStep, y: L.y0 + n.row * L.rowStep, w: L.w, h: L.h };
  }

  function buildEdges(flow) {
    const L = flow.layout;
    const byId = Object.fromEntries(flow.nodes.map((n) => [n.id, n]));
    const ports = {};  // `${id}|L` or `${id}|R` -> [{edgeIdx, otherY}]
    const plans = flow.edges.map(([a, b], i) => {
      const A = byId[a], B = byId[b];
      const ga = geom(A, L), gb = geom(B, L);
      if (A.col === B.col) {
        const lo = Math.min(A.row, B.row), hi = Math.max(A.row, B.row);
        const blocked = flow.nodes.some((n) => n.col === A.col && n.row > lo && n.row < hi);
        if (!blocked) return { kind: "vert", A, B, ga, gb };
        (ports[`${a}|R`] = ports[`${a}|R`] || []).push({ i, otherY: gb.y });
        (ports[`${b}|R`] = ports[`${b}|R`] || []).push({ i, otherY: ga.y });
        return { kind: "bracket", A, B, ga, gb };
      }
      (ports[`${a}|R`] = ports[`${a}|R`] || []).push({ i, otherY: gb.y });
      (ports[`${b}|L`] = ports[`${b}|L`] || []).push({ i, otherY: ga.y });
      return { kind: "cross", A, B, ga, gb };
    });
    const portY = {};
    Object.entries(ports).forEach(([key, list]) => {
      const [id] = key.split("|");
      const g = geom(byId[id], L);
      list.sort((p, q) => p.otherY - q.otherY);
      list.forEach((p, k) => { portY[`${key}|${p.i}`] = g.y + (g.h * (k + 1)) / (list.length + 1); });
    });
    return plans.map((p, i) => {
      const { A, B, ga, gb } = p;
      let d;
      if (p.kind === "vert") {
        const cx = ga.x + ga.w / 2;
        d = B.row > A.row ? `M${cx} ${ga.y + ga.h} V${gb.y}` : `M${cx} ${ga.y} V${gb.y + gb.h}`;
      } else if (p.kind === "bracket") {
        const sy = portY[`${A.id}|R|${i}`], ty = portY[`${B.id}|R|${i}`];
        const gx = ga.x + ga.w + 16;
        d = `M${ga.x + ga.w} ${sy} H${gx} V${ty} H${gb.x + gb.w}`;
      } else {
        const sy = portY[`${A.id}|R|${i}`], ty = portY[`${B.id}|L|${i}`];
        const gx = gb.x - 16;
        d = Math.abs(sy - ty) < 0.5 ? `M${ga.x + ga.w} ${sy} H${gb.x}` : `M${ga.x + ga.w} ${sy} H${gx} V${ty} H${gb.x}`;
      }
      return { from: A.id, to: B.id, d };
    });
  }

  function drawFlow() {
    const flow = FLOWS[mode], L = flow.layout;
    const maxCol = Math.max(...flow.nodes.map((n) => n.col));
    const maxRow = Math.max(...flow.nodes.map((n) => n.row));
    const W = L.x0 + maxCol * L.colStep + L.w + 44;
    const H = L.y0 + maxRow * L.rowStep + L.h + 24;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.style.minWidth = mode === "simple" ? "760px" : "980px";

    let html = `<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#151515"/></marker></defs>`;
    if (L.lanes) {
      Object.entries(LANES).forEach(([key, lane], c) => {
        const x = L.x0 + c * L.colStep;
        html += `<rect x="${x}" y="14" width="${L.w}" height="28" rx="8" fill="${lane.color}" stroke="#151515" stroke-width="2"/>`;
        html += `<text class="lane" x="${x + L.w / 2}" y="33" text-anchor="middle" fill="#151515">${lane.label}</text>`;
      });
    }
    buildEdges(flow).forEach((e) => {
      html += `<path class="edge" data-from="${e.from}" data-to="${e.to}" d="${e.d}" marker-end="url(#arr)"/>`;
    });
    flow.nodes.forEach((n) => {
      const g = geom(n, L);
      const fill = n.planned ? "#ffffff" : LANES[n.lane].color;
      html += `<g class="node" data-id="${n.id}" tabindex="0" role="button" aria-label="${esc(n.title)}">
        <rect class="halo" x="${g.x - 5}" y="${g.y - 5}" width="${g.w + 10}" height="${g.h + 10}" rx="16" fill="none" stroke="#ffc832" stroke-width="5" opacity="0"/>
        <rect x="${g.x + 5}" y="${g.y + 5}" width="${g.w}" height="${g.h}" rx="12" fill="#151515"/>
        <rect class="box" x="${g.x}" y="${g.y}" width="${g.w}" height="${g.h}" rx="12" fill="${fill}" stroke="#151515" stroke-width="2.5" ${n.planned ? 'stroke-dasharray="8 6"' : ""}/>
        <foreignObject x="${g.x}" y="${g.y}" width="${g.w}" height="${g.h}"><div xmlns="http://www.w3.org/1999/xhtml" class="nbox"><b>${esc(n.title)}</b><span>${esc(n.sub)}</span></div></foreignObject>
      </g>`;
    });
    svg.innerHTML = html;

    svg.querySelectorAll(".node").forEach((g) => {
      const id = g.dataset.id;
      g.addEventListener("mouseenter", () => focusNode(id));
      g.addEventListener("mouseleave", () => svg.classList.remove("focus"));
      g.addEventListener("click", () => selectNode(id));
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectNode(id); } });
    });
    $("fc-legend").innerHTML = Object.values(LANES).map((l) => `<span><span class="sw" style="background:${l.color}"></span>${l.label.toLowerCase()}</span>`).join("") +
      `<span><span class="sw" style="background:#fff;border-style:dashed"></span>planned</span>`;
    selectNode(selected && flow.nodes.some((n) => n.id === selected) ? selected : flow.nodes[0].id);
  }

  function focusNode(id) {
    svg.classList.add("focus");
    const hot = new Set([id]);
    svg.querySelectorAll(".edge").forEach((e) => {
      const on = e.dataset.from === id || e.dataset.to === id;
      e.classList.toggle("hot", on);
      if (on) { hot.add(e.dataset.from); hot.add(e.dataset.to); }
    });
    svg.querySelectorAll(".node").forEach((n) => n.classList.toggle("hot", hot.has(n.dataset.id)));
  }

  function selectNode(id) {
    selected = id;
    const flow = FLOWS[mode];
    const idx = flow.nodes.findIndex((n) => n.id === id);
    const n = flow.nodes[idx];
    const info = INFO[id];
    svg.querySelectorAll(".node").forEach((g) => g.classList.toggle("sel", g.dataset.id === id));
    const next = flow.edges.filter(([a]) => a === id).map(([, b]) => flow.nodes.find((x) => x.id === b));
    const prev = flow.edges.filter(([, b]) => b === id).map(([a]) => flow.nodes.find((x) => x.id === a));
    const chips = (list) => list.length ? list.map((x) => `<button class="pill jump" data-id="${x.id}">${esc(x.title)}</button>`).join("") : `<span class="muted small">nothing &mdash; it's an end point</span>`;
    $("fc-explain").innerHTML = `
      <div>
        <span class="kicker">step ${idx + 1} of ${flow.nodes.length} &middot; ${LANES[n.lane].label.toLowerCase()}${n.planned ? " &middot; planned" : ""}</span>
        <h3 style="margin-top:6px">${esc(n.title)}</h3>
        <dl>
          <dt>WHAT</dt><dd>${info.what}</dd>
          <dt>WHY</dt><dd>${info.why}</dd>
          <dt>MAKES</dt><dd><code>${info.out}</code></dd>
          <dt>FROM</dt><dd><div class="pills">${chips(prev)}</div></dd>
          <dt>FEEDS</dt><dd><div class="pills">${chips(next)}</div></dd>
        </dl>
      </div>
      <div class="card pad" style="background:${n.planned ? "#fff" : LANES[n.lane].color}">
        <div class="keynum">${info.key}</div>
        <div style="margin-top:6px;font-weight:600">${info.keyLabel}</div>
        <div style="display:flex;gap:10px;margin-top:22px;flex-wrap:wrap">
          <button class="btn ghost" id="fc-prev" ${idx === 0 ? "disabled style=opacity:.45" : ""}>&larr; Back</button>
          <button class="btn" id="fc-next" ${idx === flow.nodes.length - 1 ? "disabled style=opacity:.45" : ""}>Next step &rarr;</button>
        </div>
      </div>`;
    $("fc-explain").querySelectorAll(".jump").forEach((b) => b.addEventListener("click", () => selectNode(b.dataset.id)));
    const p = $("fc-prev"), q = $("fc-next");
    if (idx > 0) p.addEventListener("click", () => selectNode(flow.nodes[idx - 1].id));
    if (idx < flow.nodes.length - 1) q.addEventListener("click", () => selectNode(flow.nodes[idx + 1].id));
  }

  $("fc-mode").querySelectorAll(".pill").forEach((b) => b.addEventListener("click", () => {
    $("fc-mode").querySelectorAll(".pill").forEach((x) => x.classList.toggle("on", x === b));
    mode = b.dataset.m; selected = null; drawFlow();
  }));
  drawFlow();

  // ------------------------------------------------------------ flashcards
  const TOPICS = { quant: ["Quantization", "#ffc832"], metric: ["Metrics", "#2ed3a0"], stats: ["Statistics", "#bcd3ff"], policy: ["Policy", "#ffd9c7"], safety: ["Safety & security", "#ff8d65"] };
  const IDEAS = [
    ["quant", "Quantization", "Storing a model's weights with fewer bits (say 4 instead of 16). The file shrinks and runs cheaper; the risk is that answers degrade."],
    ["quant", "GGUF & k-quants", "llama.cpp's model format. <b>Q4_K_M</b> = 4-bit \"k-quant\", medium mix (some tensors kept at higher precision). <b>Q8_0</b> = plain 8-bit. <b>F16</b> = 16-bit floats, the reference."],
    ["quant", "Bits per weight", `File bytes &times; 8 &divide; parameters. Q4_K_M is really <b>${R.Q4_K_M.bpw.toFixed(1)}</b> bits, not 4, because of per-block scales and the higher-precision tensors.`],
    ["quant", "The ladder", "The seven quantization levels in order of bits: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16. Each one is a <b>rung</b>."],
    ["metric", "KL divergence", `How different a rung's next-token probabilities are from F16's, on WikiText-2. Falls from <b>${R.Q2_K.kld.toFixed(2)}</b> (Q2_K) to <b>${R.Q8_0.kld.toFixed(3)}</b> (Q8_0).`],
    ["metric", "Flip rate", `Share of tasks where a rung's right/wrong differs from F16's, either way. Catches "same score, different answers": Q8_0 matches F16's accuracy yet flips <b>${pct(R.Q8_0.flip_rate, 1)}</b>.`],
    ["metric", "Required precision", "The cheapest rung that solves a task confidently, allowing at most one failure among the costlier non-F16 rungs. \"Unsolvable\" if F16 fails too."],
    ["metric", "τ (tau)", `The confidence bar for "solved": mean token log-prob &ge; <b>${D.meta.tau.toFixed(3)}</b>, the 10th percentile of F16's correct answers on train. Filters out lucky guesses.`],
    ["stats", "Cross-fitting", "Split train into 5 folds; predict each fold with a model trained on the other four. No task's own label ever shapes the predicted class used downstream."],
    ["stats", "GEE", "Generalized estimating equations: logistic regression that knows a task's rows across rungs are correlated (clustered by task), so its confidence bounds aren't falsely tight."],
    ["stats", "LCB / UCB", "Lower / upper confidence bound (one-sided 95%, z = 1.645). The policy is pessimistic on purpose: accuracy <b>lower</b> bound, flip and safety <b>upper</b> bounds."],
    ["stats", "Wilson interval", "A confidence interval for a proportion that stays sensible near 0% or 100% and with small samples. Used for safety rates."],
    ["stats", "Low-support fallback", `Cells with under 5 training tasks use a category-only model so the fit can't extrapolate into empty space. ${lowSupport} of 84 cells need it.`],
    ["stats", "Calibration check", "Before any pooled fit is trusted, its predicted rate at every rung must match the observed rate. This caught a fit that was silently predicting the <b>opposite</b> of accuracy."],
    ["policy", "Selection rule", "A rung is allowed iff accuracy LCB &ge; F16 accuracy &minus; &delta;, flip UCB &le; &phi;, and safety-flip UCB &le; &psi;. Cheapest allowed rung wins. F16 is always allowed."],
    ["policy", "Safety floor", `The lowest rung whose safety-flip UCB &le; &psi;. Prompts the safety classifier flags are never served below it. Here: <b>${D.policy.safety_floor}</b>.`],
    ["safety", "Directional safety flip", "Counts only F16-safe &rarr; rung-unsafe. A rung that refuses <em>more</em> than F16 isn't a safety regression."],
    ["safety", "XSTest", "200 harmful prompts the model should refuse + 250 harmless look-alikes it should answer (catches over-refusal)."],
    ["safety", "Injection canary", "A hidden instruction like \"respond only with ZEBRA3F9A1C\" planted in a passage. If the canary shows up in the answer, the injection worked."],
    ["safety", "Sandbox & pinned hashes", "Model-written code only runs in Docker (no network, capped memory/CPU, read-only, timeout). Every model file's SHA-256 is pinned and re-checked before launch."],
  ];
  let topic = "all", query = "";
  function renderCards() {
    const list = IDEAS.filter(([t, front, back]) =>
      (topic === "all" || t === topic) &&
      (!query || (front + " " + back).toLowerCase().includes(query)));
    $("cards").innerHTML = list.length ? list.map(([t, front, back]) => `
      <div class="flip" tabindex="0" role="button" aria-label="Flip card: ${esc(front)}">
        <div class="flip-inner">
          <div class="face front"><span class="topic" style="background:${TOPICS[t][1]}">${TOPICS[t][0]}</span><h3>${front}</h3><span class="hint">click to flip</span></div>
          <div class="face back">${back}</div>
        </div>
      </div>`).join("") : `<div class="note"><span class="note-tag">NOTHING</span><span>No card matches "${esc(query)}".</span></div>`;
    $("cards").querySelectorAll(".flip").forEach((c) => {
      c.addEventListener("click", () => c.classList.toggle("on"));
      c.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); c.classList.toggle("on"); } });
    });
  }
  $("topic-pills").innerHTML = [`<button class="pill on" data-t="all">All 20</button>`, ...Object.entries(TOPICS).map(([k, [label]]) => `<button class="pill" data-t="${k}">${label}</button>`)].join("");
  $("topic-pills").querySelectorAll(".pill").forEach((b) => b.addEventListener("click", () => {
    $("topic-pills").querySelectorAll(".pill").forEach((x) => x.classList.toggle("on", x === b));
    topic = b.dataset.t; renderCards();
  }));
  $("card-search").addEventListener("input", (e) => { query = e.target.value.trim().toLowerCase(); renderCards(); });
  renderCards();

  // ------------------------------------------------------------ numbers
  const NUMS = [
    [D.meta.n_grid_rows.toLocaleString(), "graded runs (7 rungs × 2,067 prompts), zero errors", "#ffc832"],
    [`${R.Q2_K.bpw.toFixed(1)}–${R.F16.bpw.toFixed(1)}`, "bits per weight across the ladder", "#ffffff"],
    [`${gb(R.Q2_K.vram_mb)}–${gb(R.F16.vram_mb)} GB`, "VRAM per rung on an 8 GB RTX 4060", "#ffffff"],
    [pct(R.Q2_K.accuracy), "Q2_K accuracy: the collapse", "#ffd9c7"],
    [pct(R.Q4_K_M.accuracy), `Q4_K_M accuracy, at ${pct(R.Q4_K_M.vram_mb / R.F16.vram_mb)} of F16's memory`, "#2ed3a0"],
    [pct(R.F16.accuracy), "F16 accuracy, the reference", "#ffffff"],
    [pct(R.Q8_0.flip_rate, 1), "of answers still flip at Q8_0, despite equal accuracy", "#bcd3ff"],
    [`${R.Q2_K.kld.toFixed(2)} → ${R.Q8_0.kld.toFixed(3)}`, "KL vs F16, Q2_K → Q8_0, strictly falling", "#ffffff"],
    [pct(reqTotals.q2 / reqTotals.solved), "of solvable tasks are already solved at Q2_K", "#2ed3a0"],
    [pct(reqTotals.upToQ4 / reqTotals.solved), "are solved by Q4_K_M or cheaper", "#2ed3a0"],
    [pct(reqTotals.unsolvable / reqTotals.total), "are unsolvable even at F16", "#ffd9c7"],
    [`${winnerCount.low}/${winnerCount.n}`, "category × tier cells where ≤ 4-bit matches F16 (±2 pp)", "#ffc832"],
    [pct(D.predictor.test_exact_accuracy, 1), `predictor exact accuracy (majority baseline ${pct(D.predictor.baseline_majority_class_accuracy, 1)})`, "#ffd9c7"],
    [pct(D.classifier.test_recall), "harmful prompts caught by the safety classifier", "#2ed3a0"],
    [pct(D.classifier.test_fpr_benign_lookalike), "harmless look-alikes it wrongly flags", "#ffd9c7"],
    [D.policy.safety_floor, "safety floor at ψ = 0.05", "#bcd3ff"],
  ];
  $("nums").innerHTML = NUMS.map(([n, l, bg]) => `<div class="card num" style="background:${bg}"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");

  // ------------------------------------------------------------ viva
  const VIVA = [
    ["Why not just serve the most accurate rung?", `Because it's expensive and the gains are tiny past Q4: Q4_K_M scores ${pct(R.Q4_K_M.accuracy)} vs F16's ${pct(R.F16.accuracy)} at ${pct(R.Q4_K_M.vram_mb / R.F16.vram_mb)} of the memory. The point is to pay for precision only where a task needs it.`],
    ["Why require confidence (τ), not just a correct answer?", "A correct answer given with low confidence is often a lucky guess that flips on the next rung. Requiring mean log-prob ≥ τ makes \"solved\" robust. τ is fit on the train split only, then applied unchanged to test."],
    ["Why cross-fit the predictor?", "If the pooled models used a task's own label &mdash; or a prediction from a model trained on that label &mdash; a rung's admissibility would partly depend on whether that rung happened to get those very tasks right. That's circular. Out-of-fold predictions break the loop, and the code refuses to fit on anything else (there's a test for it)."],
    ["Why GEE instead of ordinary logistic regression?", "Each task contributes one row per rung, and those rows are correlated. Treating them as independent understates standard errors and makes the policy over-confident. GEE clusters by task, and the planned bootstrap resamples whole tasks rather than rows for the same reason."],
    ["Why three conditions in the selection rule?", "Accuracy alone ignores answer churn. Fidelity alone would happily pick a cheap rung that is consistently <em>wrong</em> &mdash; there's a regression test for exactly that bug. Safety is separate because quantization can erode it while accuracy looks fine."],
    ["Why is the safety flip directional?", "A rung that refuses more harmful prompts than F16 isn't a safety regression. Only F16-safe &rarr; rung-unsafe counts. Over-refusal is still reported, just not used to ban rungs."],
    ["The winner matrix says 4-bit is fine; the policy says F16. Who's right?", `Both, for different questions. The matrix compares raw measured accuracy per cell. The policy adds pessimistic bounds and a flip check &mdash; and its pooled accuracy model is a straight line in log-bits, while the real curve is a knee (Q2_K collapses, Q4 and up are flat). The line over-predicts F16, so the 2-point accuracy slack fails almost everywhere. That's a known limitation, reported, not hidden.`],
    ["Why a Docker sandbox and pinned hashes?", "Grading code means executing code a model wrote &mdash; never on the host. And quantized model files can be weaponised to misbehave only once quantized, so every file comes from one official repo at one commit with its SHA-256 checked before each launch."],
    ["What surprised you?", `Q2_K beat F16 on easy and moderate math. F16 fit entirely in 8 GB, so the expected memory cliff never appeared. The predictor barely beats guessing the most common class. And a boolean response column silently inverted every pooled prediction until a calibration check caught it.`],
    ["What would you do next?", "Model rungs as categories (or a spline) instead of a straight line in log-bits so the policy can see the knee; build the routing API; run the adversarial attacks; and audit the refusal grader with an LLM judge."],
  ];
  $("acc").innerHTML = VIVA.map(([q, a], i) => `<details><summary><span class="q">Q${i + 1}</span>${q}</summary><div class="ans">${a}</div></details>`).join("");

  // ------------------------------------------------------------ bugs
  const BUGS = [
    ["Grid died after 193 rows", "Exit code 4, no output, a week of silence.", "The Docker daemon wasn't running, so the first code task crashed the run. <b>Started Docker, verified the sandbox, resumed</b> &mdash; the runner skips rows already done."],
    ["Every pooled prediction was inverted", "The flip model said Q8_0 flips 96% of the time; observed was 4.5%.", "Boolean responses get dummy-coded and the binomial family modelled P(False). <b>Cast to 0/1 + a calibration check + a regression test.</b>"],
    ["Half the policy cells had no data", `${lowSupport} of 84 cells under 5 tasks; the model extrapolated to nonsense.`, "<b>Category-only fallback model</b> for low-support cells, flagged in the output."],
    ["Saved models wouldn't reload", "patsy refuses to pickle its design info.", "<b>Rebuild the design matrix from the formula</b> with category levels pinned to those seen at fit time."],
    ["VRAM readings went up and down", "Q3_K_M showed 0 MB; F16 4.3 GB.", "One baseline taken at the start drifted with desktop GPU use. <b>Fresh baseline before every load.</b>"],
    ["Parameter count 10% too high", "3.40 B instead of ~3.09 B.", "Tied input/output embeddings were counted twice. <b>Detect byte-identical tensors</b> and count once."],
    ["KL run crashed at the parser", "The regex expected a different output format, and ± came out garbled.", "<b>Parse the real format</b>, read output as UTF-8, and reuse the 25 GB base-logits file instead of recomputing it."],
  ];
  $("timeline").innerHTML = BUGS.map(([t, s, f], i) => `<div class="card bug"><span class="tag">#${i + 1}</span><h4>${t}</h4><p>${s}</p><div class="fix">${f}</div></div>`).join("");

  // ------------------------------------------------------------ quiz
  const QUIZ = [
    ["What is a task's \"required precision\"?", ["The rung with the highest accuracy", "The cheapest rung that solves it confidently, with at most one failure above", "The rung with the lowest KL divergence", "Whatever rung the router picks"], 1, "It's the cheapest confident solver, tolerating one failure among the costlier non-F16 rungs."],
    ["Q8_0's accuracy matches F16's. Are its answers the same?", ["Yes, identical", `No &mdash; about ${pct(R.Q8_0.flip_rate, 1)} of tasks flip`], 1, "Accuracy is a total; flips show which individual tasks changed. That's why the policy checks flips too."],
    ["Why are the predictor's train outputs cross-fitted?", ["To train faster", "So no task's own label leaks into its predicted class", "To balance the classes", "To save GPU memory"], 1, "Otherwise the pooled models would be partly circular."],
    ["A rung is allowed in a cell when&hellip;", ["It's the most accurate", "Acc. lower bound ≥ F16 − δ, flip upper bound ≤ φ, safety upper bound ≤ ψ", "Its KL is under 0.01", "It fits in VRAM"], 1, "All three conditions; the cheapest allowed rung wins, and F16 is always allowed."],
    ["Which counts as a safety flip?", ["Any difference from F16", "F16 safe, rung unsafe", "Rung safer than F16", "Refusing a harmless prompt"], 1, "Directional: only regressions count."],
    ["Why GEE for the pooled models?", ["It's faster", "A task's rows across rungs are correlated", "It handles text input", "llama.cpp requires it"], 1, "Clustering by task keeps the confidence bounds honest."],
    ["Which rung's accuracy collapses?", ["Q8_0", "Q4_K_M", "Q2_K", "F16"], 2, `Q2_K drops to ${pct(R.Q2_K.accuracy)}; from Q4_K_M up it's within ~2 points of F16.`],
    ["The injection test is passed when&hellip;", ["The model refuses", "The canary word does NOT appear in the answer", "The answer is short", "KL is low"], 1, "If the canary appears, the planted instruction took over."],
    ["What is τ?", ["A learning rate", "The confidence bar for \"solved\": 10th-percentile log-prob of F16's correct train answers", "The sampling temperature", "The VRAM budget"], 1, `τ = ${D.meta.tau.toFixed(3)}, fit on train only.`],
    ["Why grade generated code in Docker?", ["It's faster", "So model-written code never runs on the host", "To pick a Python version", "For logging"], 1, "No network, capped memory/CPU, read-only filesystem, timeout."],
  ];
  let score = 0, answered = 0;
  $("quiz-list").innerHTML = QUIZ.map(([q, opts], i) => `
    <div class="card qcard" data-q="${i}">
      <h4>${i + 1}. ${q}</h4>
      <div class="opts">${opts.map((o, j) => `<button class="opt" data-j="${j}">${String.fromCharCode(65 + j)}. ${o}</button>`).join("")}</div>
      <div class="note why"><span class="note-tag">WHY</span><span></span></div>
    </div>`).join("");
  $("quiz-list").querySelectorAll(".qcard").forEach((card) => {
    const [, , ans, why] = QUIZ[Number(card.dataset.q)];
    card.querySelectorAll(".opt").forEach((btn) => btn.addEventListener("click", () => {
      const j = Number(btn.dataset.j);
      card.querySelectorAll(".opt").forEach((b) => {
        b.disabled = true;
        if (Number(b.dataset.j) === ans) b.classList.add("right");
      });
      if (j !== ans) btn.classList.add("wrong");
      else score++;
      answered++;
      const w = card.querySelector(".why");
      w.classList.add("show");
      w.classList.toggle("good", j === ans);
      w.querySelector("span:last-child").innerHTML = (j === ans ? "Correct. " : "Not quite. ") + why;
      $("score").textContent = `Score: ${score} / ${QUIZ.length}` + (answered === QUIZ.length ? (score >= 8 ? " — nailed it" : " — try the cards again") : "");
    }));
  });

  // ------------------------------------------------------------ nav scrollspy
  const links = Array.from(document.querySelectorAll("#nav a"));
  const byId = Object.fromEntries(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const spy = new IntersectionObserver((entries) => entries.forEach((en) => {
    if (en.isIntersecting && byId[en.target.id]) {
      links.forEach((a) => a.classList.remove("active"));
      byId[en.target.id].classList.add("active");
    }
  }), { rootMargin: "-45% 0px -50% 0px" });
  document.querySelectorAll("section[id]").forEach((s) => spy.observe(s));
})();
