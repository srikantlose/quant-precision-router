(function () {
  "use strict";

  const D = window.QL;
  const LADDER = D.meta.ladder;
  const NONREF = LADDER.slice(0, 6);
  const CATS = D.meta.categories;
  const TIERS = D.meta.tiers;

  // Rung identity: one warm hue light->dark (validated ordinal ramp), F16 = ink.
  const RC = { Q2_K: "#ff8d65", Q3_K_M: "#ec6d3d", Q4_K_M: "#d05320", Q5_K_M: "#b13d02", Q6_K: "#8e2e00", Q8_0: "#6b2101", F16: "#151515" };
  const textOn = (r) => (r === "Q2_K" || r === "Q3_K_M" ? "#151515" : "#ffffff");
  const BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
  const INK = "#151515", INK2 = "#4b4840", MUTED = "#7d786d", GRID = "#ece5d8", SUN = "#ffc832", GREY = "#c9c1b2";
  const MONO = '"JetBrains Mono", ui-monospace, monospace';

  const SHORT = { Q2_K: "Q2", Q3_K_M: "Q3", Q4_K_M: "Q4", Q5_K_M: "Q5", Q6_K: "Q6", Q8_0: "Q8", F16: "F16" };
  const pretty = (c) => c.replace(/_/g, " ");
  const pct = (x, d = 0) => (x == null || isNaN(x) ? "—" : (x * 100).toFixed(d) + "%");
  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- tooltip
  const tip = $("tip");
  function placeTip(e) {
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = e.clientX + 16, y = e.clientY + 16;
    if (x + w > window.innerWidth - 8) x = e.clientX - w - 16;
    if (y + h > window.innerHeight - 8) y = e.clientY - h - 16;
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }
  function bindTip(el, html) {
    el.addEventListener("mouseenter", (e) => { tip.innerHTML = html(); tip.classList.add("show"); placeTip(e); });
    el.addEventListener("mousemove", placeTip);
    el.addEventListener("mouseleave", () => tip.classList.remove("show"));
  }
  const tipBar = (r, v, max) =>
    `<div style="display:flex;align-items:center;gap:8px;margin-top:3px"><span style="font-family:${MONO};width:58px">${r}</span>` +
    `<span style="display:inline-block;height:8px;border-radius:3px;width:${Math.max(2, (v / max) * 110)}px;background:${r === "F16" ? "#fff" : RC[r]}"></span>` +
    `<span style="font-family:${MONO}">${pct(v)}</span></div>`;

  // ---------------------------------------------------------------- Chart.js setup
  Chart.defaults.font.family = '"DM Sans", system-ui, sans-serif';
  Chart.defaults.font.size = 12.5;
  Chart.defaults.color = INK2;
  Chart.defaults.borderColor = GRID;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.plugins.legend.display = false;
  Object.assign(Chart.defaults.plugins.tooltip, { backgroundColor: INK, padding: 10, cornerRadius: 8, titleFont: { weight: "700" }, displayColors: false });

  // Direct value labels. Formatting lives on the dataset object (raw user data),
  // because Chart.js would treat a function inside plugin options as scriptable.
  Chart.register({
    id: "vlabels",
    afterDatasetsDraw(chart) {
      const ctx = chart.ctx;
      chart.data.datasets.forEach((ds, di) => {
        if (!ds.vfmt) return;
        const meta = chart.getDatasetMeta(di);
        if (meta.hidden) return;
        meta.data.forEach((el, i) => {
          const v = ds.data[i];
          if (v == null) return;
          const t = ds.vfmt(v, i);
          if (!t) return;
          ctx.save();
          ctx.font = `700 11.5px ${MONO}`;
          ctx.fillStyle = INK;
          if (ds.vmode === "h") { ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(t, el.x + 7, el.y); }
          else if (ds.vmode === "pt") {
            const [dx, dy, align] = (ds.voff && ds.voff[i]) || [12, 0, "left"];
            ctx.textAlign = align; ctx.textBaseline = "middle"; ctx.fillText(t, el.x + dx, el.y + dy);
          }
          else { ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(t, el.x, el.y - 6); }
          ctx.restore();
        });
      });
    },
  });
  Chart.register({
    id: "hline",
    afterDatasetsDraw(chart) {
      const o = chart.$hline;
      if (!o || o.value == null) return;
      const y = chart.scales.y.getPixelForValue(o.value);
      const { left, right } = chart.chartArea;
      const ctx = chart.ctx;
      ctx.save();
      ctx.setLineDash([6, 5]); ctx.strokeStyle = INK; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = `700 11px ${MONO}`; ctx.fillStyle = INK; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(o.label, right + 6, y);
      ctx.restore();
    },
  });

  const xCat = { grid: { display: false }, border: { color: INK, width: 2 }, ticks: { color: INK, font: { family: MONO, size: 11, weight: "700" } } };
  const yVal = (extra) => Object.assign({ grid: { color: GRID }, border: { display: false }, ticks: { color: MUTED } }, extra || {});
  const bars = (data, colors, extra) => Object.assign({ data, backgroundColor: colors, borderColor: INK, borderWidth: 2, borderRadius: 6, borderSkipped: "start", maxBarThickness: 48 }, extra || {});

  // ---------------------------------------------------------------- hero
  $("hero-stats").innerHTML = [
    [D.meta.n_grid_rows.toLocaleString(), "scored runs"],
    ["7", "precisions"],
    [String(CATS.length), "task types"],
    ["0", "failed runs"],
  ].map(([n, l]) => `<div class="s"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");

  const maxBpw = D.rungs.F16.bpw;
  (function heroSteps() {
    const g = $("hero-steps");
    let html = "";
    LADDER.forEach((r, i) => {
      const h = 50 + (D.rungs[r].bpw / maxBpw) * 230;
      const x = 40 + i * 64, y = 362 - h;
      html += `<rect x="${x + 5}" y="${y + 5}" width="54" height="${h}" rx="9" fill="#151515"/>`;
      html += `<rect x="${x}" y="${y}" width="54" height="${h}" rx="9" fill="${RC[r]}" stroke="#151515" stroke-width="3"/>`;
      html += `<text x="${x + 27}" y="${y + 22}" text-anchor="middle" font-family="JetBrains Mono" font-weight="700" font-size="12" fill="${textOn(r)}">${r.replace("_K_M", "").replace("_K", "").replace("_0", "")}</text>`;
    });
    g.innerHTML = html;
  })();

  // ---------------------------------------------------------------- ladder
  let selRung = "Q4_K_M";
  const stairs = $("stairs");
  const STEP_W = 84, BASE = 290;
  const stepX = (i) => 18 + i * 97;
  const stepH = (r) => 46 + (D.rungs[r].bpw / maxBpw) * 196;

  (function drawStairs() {
    let html = `<line x1="8" y1="${BASE}" x2="692" y2="${BASE}" stroke="#151515" stroke-width="3" stroke-linecap="round"/>`;
    LADDER.forEach((r, i) => {
      const x = stepX(i), h = stepH(r), y = BASE - h;
      html += `<g class="step" data-rung="${r}" role="option">
        <rect class="shadow" x="${x + 6}" y="${y + 6}" width="${STEP_W}" height="${h}" rx="12" fill="#151515" opacity="0"/>
        <rect class="body" x="${x}" y="${y}" width="${STEP_W}" height="${h}" rx="12" fill="${RC[r]}" stroke="#151515" stroke-width="3"/>
        <text x="${x + STEP_W / 2}" y="${y + 26}" text-anchor="middle" font-family="JetBrains Mono" font-weight="700" font-size="14" fill="${textOn(r)}">${D.rungs[r].bpw.toFixed(1)}</text>
        <text x="${x + STEP_W / 2}" y="${y + 42}" text-anchor="middle" font-family="DM Sans" font-weight="600" font-size="11" fill="${textOn(r)}" opacity=".85">bits/wt</text>
        <text x="${x + STEP_W / 2}" y="${BASE + 26}" text-anchor="middle" font-family="JetBrains Mono" font-weight="700" font-size="14" fill="#151515">${r}</text>
      </g>`;
    });
    html += `<g id="token">
      <circle cx="0" cy="0" r="18" fill="#ffc832" stroke="#151515" stroke-width="3"/>
      <circle cx="-6" cy="-3" r="2.6" fill="#151515"/><circle cx="6" cy="-3" r="2.6" fill="#151515"/>
      <path d="M-6 5 Q0 10 6 5" fill="none" stroke="#151515" stroke-width="2.5" stroke-linecap="round"/>
    </g>`;
    stairs.innerHTML = html;
    stairs.querySelectorAll(".step").forEach((g) => g.addEventListener("click", () => selectRung(g.dataset.rung)));
    stairs.addEventListener("keydown", (e) => {
      const i = LADDER.indexOf(selRung);
      if (e.key === "ArrowRight" && i < LADDER.length - 1) { selectRung(LADDER[i + 1]); e.preventDefault(); }
      if (e.key === "ArrowLeft" && i > 0) { selectRung(LADDER[i - 1]); e.preventDefault(); }
    });
  })();

  (function vramTicks() {
    let html = "";
    for (let k = 1; k < 8; k++) {
      const x = 127 + (226 * k * 1024) / D.meta.vram_total_mb;
      html += `<line x1="${x}" y1="47" x2="${x}" y2="95"/>`;
    }
    $("vram-ticks").innerHTML = html;
  })();

  const scatter = new Chart($("c-scatter"), {
    type: "scatter",
    data: { datasets: [{
      data: LADDER.map((r) => ({ x: D.rungs[r].vram_mb, y: D.rungs[r].accuracy * 100 })),
      pointBackgroundColor: LADDER.map((r) => RC[r]),
      pointBorderColor: INK, pointBorderWidth: 2, pointRadius: 7, pointHoverRadius: 10,
      vfmt: (v, i) => LADDER[i], vmode: "pt",
      // hand-placed so the four rungs clustered at 72-74% don't collide
      voff: [[12, 0, "left"], [-12, 2, "right"], [4, 22, "left"], [-6, -18, "right"], [8, -18, "left"], [0, 20, "center"], [-12, -16, "right"]],
    }] },
    options: {
      layout: { padding: { right: 24, top: 10 } },
      scales: {
        x: { title: { display: true, text: "VRAM (MB)", color: INK2 }, grid: { color: GRID }, ticks: { color: MUTED }, min: 1000, max: 7000 },
        y: yVal({ title: { display: true, text: "accuracy (%)", color: INK2 }, min: 45, max: 80, ticks: { color: MUTED, stepSize: 5 } }),
      },
      plugins: { tooltip: { callbacks: { label: (c) => `${LADDER[c.dataIndex]}: ${c.parsed.y.toFixed(1)}% at ${c.parsed.x.toLocaleString()} MB` } } },
    },
  });

  function plainEnglish(r) {
    const R = D.rungs[r], F = D.rungs.F16;
    if (r === "F16") return `F16 stores every weight as a 16-bit float &mdash; the full-precision reference. The model file is ${F.file_gb} GB and everything else is judged against it.`;
    const gap = (F.accuracy - R.accuracy) * 100;
    const verdict = gap > 10 ? "and it shows: accuracy falls off a cliff" : gap > 3 ? "with a noticeable accuracy cost" : "for practically no accuracy cost";
    return `${r} squeezes each weight into about ${R.bpw.toFixed(1)} bits instead of 16, shrinking the file from ${F.file_gb} GB to ${R.file_gb} GB, ${verdict} (${gap >= 0 ? "&minus;" : "+"}${Math.abs(gap).toFixed(1)} pp). But ${pct(R.flip_rate)} of individual answers still change vs F16.`;
  }

  function readout(label, value, delta, cls) {
    return `<div class="readout"><div class="kicker">${label}</div><div class="v tabular">${value}</div><div class="d ${cls || "flat"}">${delta || "&nbsp;"}</div></div>`;
  }

  function selectRung(r) {
    selRung = r;
    const i = LADDER.indexOf(r);
    stairs.querySelectorAll(".step").forEach((g) => {
      const on = g.dataset.rung === r;
      g.querySelector(".shadow").setAttribute("opacity", on ? "1" : "0");
      g.querySelector(".body").setAttribute("stroke-width", on ? "4" : "3");
      g.setAttribute("aria-selected", on ? "true" : "false");
    });
    $("token").setAttribute("transform", `translate(${stepX(i) + STEP_W / 2} ${BASE - stepH(r) - 26})`);

    const R = D.rungs[r], F = D.rungs.F16;
    const dAcc = (R.accuracy - F.accuracy) * 100;
    $("readouts").innerHTML = [
      readout("bits / weight", R.bpw.toFixed(2), r === "F16" ? "reference" : `${((1 - R.bpw / F.bpw) * 100).toFixed(0)}% fewer than F16`, "up"),
      readout("VRAM", R.vram_mb.toLocaleString() + " MB", r === "F16" ? "reference" : `${((1 - R.vram_mb / F.vram_mb) * 100).toFixed(0)}% less than F16`, "up"),
      readout("accuracy", pct(R.accuracy, 1), r === "F16" ? "reference" : `${dAcc >= 0 ? "+" : ""}${dAcc.toFixed(1)} pp vs F16`, Math.abs(dAcc) < 1.5 ? "flat" : dAcc > 0 ? "up" : "down"),
      readout("answers flipped", r === "F16" ? "0%" : pct(R.flip_rate, 1), r === "F16" ? "reference" : "vs F16", R.flip_rate > 0.1 ? "down" : "flat"),
      readout("KL vs F16", r === "F16" ? "0" : R.kld.toPrecision(2), r === "F16" ? "reference" : `top token agrees ${R.same_top_pct.toFixed(0)}%`, "flat"),
      readout("avg latency", R.latency_s.toFixed(2) + " s", `${R.load_s.toFixed(1)} s to load`, "flat"),
    ].join("") + `<div class="note info" style="grid-column:1/-1"><span class="note-tag">IN ENGLISH</span><span>${plainEnglish(r)}</span></div>`;

    $("vram-fill").style.width = Math.max(4, (226 * R.vram_mb) / D.meta.vram_total_mb) + "px";
    $("vram-fill").setAttribute("fill", RC[r]);
    $("vram-sub").innerHTML = `<strong>${r}</strong> uses ${R.vram_mb.toLocaleString()} MB &mdash; ${pct(R.vram_mb / D.meta.vram_total_mb)} of the card.`;

    const ds = scatter.data.datasets[0];
    ds.pointRadius = LADDER.map((x) => (x === r ? 11 : 7));
    ds.pointBorderColor = LADDER.map((x) => (x === r ? SUN : INK));
    ds.pointBorderWidth = LADDER.map((x) => (x === r ? 4 : 2));
    scatter.update();
  }

  // ---------------------------------------------------------------- winner matrix
  let mxMode = "match", tol = 0.02, selCell = ["math", "easy"];
  const cellChart = new Chart($("c-cell"), {
    type: "bar",
    data: { labels: LADDER, datasets: [bars([], LADDER.map((r) => RC[r]), { vfmt: (v) => v.toFixed(0) + "%" })] },
    options: { layout: { padding: { right: 86 } }, scales: { x: xCat, y: yVal({ min: 0, max: 105, ticks: { color: MUTED, callback: (v) => (v <= 100 ? v + "%" : "") } }) },
      plugins: { tooltip: { callbacks: { label: (c) => `${c.parsed.y.toFixed(1)}% (n=${D.cat_tier[selCell[0]][selCell[1]][LADDER[c.dataIndex]].n})` } } } },
  });

  function winner(cell) {
    const ref = cell.F16.acc;
    if (mxMode === "match") return LADDER.find((r) => cell[r].acc >= ref - tol - 1e-9);
    let best = LADDER[0];
    for (const r of LADDER) if (cell[r].acc > cell[best].acc + 1e-9) best = r;
    return best;
  }

  function renderMatrix() {
    const mx = $("mx");
    let html = `<thead><tr><th></th>${TIERS.map((t) => `<th>${t}</th>`).join("")}</tr></thead><tbody>`;
    const wins = [];
    CATS.forEach((cat) => {
      html += `<tr><th class="rowh">${pretty(cat)}</th>`;
      TIERS.forEach((t) => {
        const cell = D.cat_tier[cat][t];
        const w = winner(cell);
        wins.push(w);
        const sel = selCell[0] === cat && selCell[1] === t ? " sel" : "";
        html += `<td class="${sel}" data-cat="${cat}" data-tier="${t}" style="background:${RC[w]};color:${textOn(w)}">${w}<span class="sub">${pct(cell[w].acc)} &middot; n=${cell[w].n}</span></td>`;
      });
      html += `</tr>`;
    });
    mx.innerHTML = html + `</tbody>`;
    mx.querySelectorAll("td").forEach((td) => {
      const cat = td.dataset.cat, t = td.dataset.tier;
      td.addEventListener("click", () => { selCell = [cat, t]; renderMatrix(); renderCellDetail(); });
      bindTip(td, () => {
        const cell = D.cat_tier[cat][t];
        const max = Math.max(...LADDER.map((r) => cell[r].acc), 0.01);
        return `<b>${pretty(cat)} &middot; ${t}</b> &mdash; ${cell.F16.n} tasks<br>use <b>${winner(cell)}</b>` + LADDER.map((r) => tipBar(r, cell[r].acc, max)).join("");
      });
    });

    const low = wins.filter((w) => LADDER.indexOf(w) <= 2).length;
    const full = wins.filter((w) => w === "F16").length;
    const saved = wins.reduce((s, w) => s + (1 - D.rungs[w].vram_mb / D.rungs.F16.vram_mb), 0) / wins.length;
    $("mx-kpis").innerHTML = [
      [`${low}/${wins.length}`, "cells where 4-bit or less is enough", "var(--mint)"],
      [`${full}/${wins.length}`, "cells that need full F16", "var(--blush)"],
      [pct(saved), "average VRAM saved vs always-F16", "var(--sun)"],
    ].map(([n, l, bg]) => `<div class="k" style="background:${bg}"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
  }

  function renderCellDetail() {
    const [cat, t] = selCell;
    const cell = D.cat_tier[cat][t];
    const w = winner(cell);
    $("mx-detail-title").innerHTML = `${pretty(cat)} &middot; ${t} &rarr; <span class="rchip" style="background:${RC[w]};color:${textOn(w)}">${w}</span>`;
    $("mx-detail-sub").textContent = mxMode === "match"
      ? `Cheapest rung within ${Math.round(tol * 100)} pp of F16 (dashed line). ${cell.F16.n} tasks in this cell.`
      : `Highest measured accuracy (dashed line = F16). ${cell.F16.n} tasks in this cell.`;
    cellChart.data.datasets[0].data = LADDER.map((r) => cell[r].acc * 100);
    cellChart.data.datasets[0].borderWidth = LADDER.map((r) => (r === w ? 4 : 2));
    cellChart.$hline = mxMode === "match"
      ? { value: Math.max(0, (cell.F16.acc - tol) * 100), label: `F16 − ${Math.round(tol * 100)}pp` }
      : { value: cell.F16.acc * 100, label: "F16" };
    cellChart.update();
  }

  $("mx-mode").querySelectorAll(".pill").forEach((b) => b.addEventListener("click", () => {
    $("mx-mode").querySelectorAll(".pill").forEach((x) => x.classList.toggle("on", x === b));
    mxMode = b.dataset.mode;
    $("tol-row").style.opacity = mxMode === "match" ? "1" : ".4";
    $("tol").disabled = mxMode !== "match";
    renderMatrix(); renderCellDetail();
  }));
  $("tol").addEventListener("input", (e) => {
    tol = Number(e.target.value) / 100;
    $("tol-val").textContent = `${e.target.value} pp`;
    renderMatrix(); renderCellDetail();
  });
  $("rung-legend").innerHTML = LADDER.map((r) => `<span><span class="sw" style="background:${RC[r]}"></span>${r}</span>`).join("") +
    `<span class="muted">lighter = fewer bits</span>`;

  // ---------------------------------------------------------------- accuracy heatmap
  let accSort = "drop";
  function renderAcc() {
    const rows = CATS.slice();
    if (accSort === "drop") rows.sort((a, b) => (D.cat_acc[b].F16 - D.cat_acc[b].Q2_K) - (D.cat_acc[a].F16 - D.cat_acc[a].Q2_K));
    if (accSort === "f16") rows.sort((a, b) => D.cat_acc[b].F16 - D.cat_acc[a].F16);
    let html = `<thead><tr><th></th>${LADDER.map((r) => `<th><span class="rchip" style="background:${RC[r]};color:${textOn(r)}">${r}</span></th>`).join("")}</tr></thead><tbody>`;
    rows.forEach((cat) => {
      html += `<tr><th class="rowh">${pretty(cat)}</th>`;
      LADDER.forEach((r) => {
        const v = D.cat_acc[cat][r];
        const idx = Math.max(0, Math.min(12, Math.round(((v - 0.3) / 0.7) * 12)));
        html += `<td data-cat="${cat}" data-rung="${r}" style="background:${BLUE[idx]};color:${idx >= 7 ? "#fff" : INK}">${pct(v)}</td>`;
      });
      html += `</tr>`;
    });
    $("acc-mx").innerHTML = html + `</tbody>`;
    $("acc-mx").querySelectorAll("td").forEach((td) => bindTip(td, () => {
      const { cat, rung } = td.dataset;
      const v = D.cat_acc[cat][rung], f = D.cat_acc[cat].F16;
      const flip = rung === "F16" ? null : D.cat_flip[cat][rung];
      return `<b>${pretty(cat)} on ${rung}</b><br>accuracy ${pct(v, 1)}` +
        (rung === "F16" ? " (reference)" : `<br>${((v - f) * 100 >= 0 ? "+" : "")}${((v - f) * 100).toFixed(1)} pp vs F16<br>answers flipped vs F16: ${pct(flip, 1)}`);
    }));
  }
  $("acc-sort").querySelectorAll(".pill").forEach((b) => b.addEventListener("click", () => {
    $("acc-sort").querySelectorAll(".pill").forEach((x) => x.classList.toggle("on", x === b));
    accSort = b.dataset.sort; renderAcc();
  }));

  // ---------------------------------------------------------------- fidelity
  new Chart($("c-flip"), {
    type: "bar",
    data: { labels: NONREF, datasets: [bars(NONREF.map((r) => D.rungs[r].flip_rate * 100), NONREF.map((r) => RC[r]), { vfmt: (v) => v.toFixed(1) + "%" })] },
    options: { scales: { x: xCat, y: yVal({ beginAtZero: true, ticks: { color: MUTED, callback: (v) => v + "%" } }) },
      plugins: { tooltip: { callbacks: { label: (c) => `${c.parsed.y.toFixed(1)}% of tasks flip vs F16` } } } },
  });
  new Chart($("c-kl"), {
    type: "bar",
    data: { labels: NONREF, datasets: [bars(NONREF.map((r) => D.rungs[r].kld), NONREF.map((r) => RC[r]), { vfmt: (v) => v.toPrecision(2) })] },
    options: { scales: { x: xCat, y: yVal({ type: "logarithmic", min: 0.001, max: 1, ticks: { color: MUTED, callback: (v) => ([0.001, 0.01, 0.1, 1].includes(v) ? v : "") } }) },
      plugins: { tooltip: { callbacks: { label: (c) => `mean KL ${c.parsed.y.toPrecision(3)} · top token agrees ${D.rungs[NONREF[c.dataIndex]].same_top_pct.toFixed(1)}%` } } } },
  });
  (function fidNote() {
    const q8 = D.rungs.Q8_0, f = D.rungs.F16, q4 = D.rungs.Q4_K_M;
    $("fid-note").innerHTML = `<span class="note-tag">KEY IDEA</span><span><strong>Q8_0</strong> scores ${pct(q8.accuracy, 1)} vs F16's ${pct(f.accuracy, 1)} &mdash; practically identical &mdash; yet ${pct(q8.flip_rate, 1)} of tasks flip. At <strong>Q4_K_M</strong> it's ${pct(q4.flip_rate, 1)}. Matching accuracy does not mean matching behaviour, which is why the router checks flips as well as accuracy.</span>`;
  })();

  // ---------------------------------------------------------------- what tasks need
  const NEED_LABELS = [...LADDER, "none"];
  const needChart = new Chart($("c-need"), {
    type: "bar",
    data: { labels: NEED_LABELS, datasets: [bars([], [...LADDER.map((r) => RC[r]), GREY], { vfmt: (v) => (v ? String(v) : "") })] },
    options: { scales: { x: xCat, y: yVal({ beginAtZero: true, title: { display: true, text: "tasks", color: INK2 } }) },
      plugins: { tooltip: { callbacks: { title: (c) => (c[0].label === "none" ? "unsolvable even at F16" : `cheapest rung that solves it: ${c[0].label}`), label: (c) => `${c.parsed.y} tasks` } } } },
  });
  let needCat = "all";
  function renderNeed() {
    const counts = NEED_LABELS.map((_, k) => {
      const key = k < 7 ? String(k) : "unsolvable";
      return needCat === "all" ? CATS.reduce((s, c) => s + D.req_hist[c][key], 0) : D.req_hist[needCat][key];
    });
    needChart.data.datasets[0].data = counts;
    needChart.update();
    const total = counts.reduce((a, b) => a + b, 0);
    const solvable = total - counts[7];
    const atQ2 = counts[0], upToQ4 = counts[0] + counts[1] + counts[2], highEnd = counts[5] + counts[6];
    $("need-side").innerHTML = `
      <div class="kpis" style="grid-template-columns:1fr 1fr;margin:0">
        <div class="k" style="background:var(--mint)"><div class="n">${pct(atQ2 / Math.max(1, solvable))}</div><div class="l">of solvable tasks are already solved at Q2_K</div></div>
        <div class="k" style="background:var(--sun)"><div class="n">${pct(upToQ4 / Math.max(1, solvable))}</div><div class="l">are solved by Q4_K_M or cheaper</div></div>
        <div class="k" style="background:var(--blush)"><div class="n">${pct(highEnd / Math.max(1, solvable))}</div><div class="l">genuinely need Q8_0 or F16</div></div>
        <div class="k" style="background:#e9e3d6"><div class="n">${pct(counts[7] / Math.max(1, total))}</div><div class="l">are unsolvable even at F16</div></div>
      </div>
      <div class="note info"><span class="note-tag">DEFINITION</span><span>"Solved" means correct <em>and</em> confident: the answer's mean token log-probability is at least &tau; = ${D.meta.tau.toFixed(3)} (the 10th percentile of F16's correct answers). Required precision is the cheapest solved rung, allowing at most one failure among the rungs above it.</span></div>`;
  }
  $("need-pills").innerHTML = [`<button class="pill on" data-c="all">All categories</button>`, ...CATS.map((c) => `<button class="pill" data-c="${c}">${pretty(c)}</button>`)].join("");
  $("need-pills").querySelectorAll(".pill").forEach((b) => b.addEventListener("click", () => {
    $("need-pills").querySelectorAll(".pill").forEach((x) => x.classList.toggle("on", x === b));
    needCat = b.dataset.c; renderNeed();
  }));

  // ---------------------------------------------------------------- policy lab
  const PRESETS = { strict: { d: 0.02, f: 0.1, s: 0.05 }, balanced: { d: 0.1, f: 0.3, s: 0.1 }, loose: { d: 0.15, f: 1, s: 1 } };
  const pol = Object.assign({}, PRESETS.strict);
  const cells = D.policy.cells;

  function chooseCell(cell) {
    const adm = ["F16"], why = {};
    for (const b of NONREF) {
      const s = D.rungs[b].safety_flip_ucb;
      if (cell.acc_lcb[b] < cell.acc_ref - pol.d) why[b] = `accuracy LB ${pct(cell.acc_lcb[b])} &lt; ${pct(cell.acc_ref - pol.d)}`;
      else if (cell.flip_ucb[b] > pol.f) why[b] = `flip UB ${pct(cell.flip_ucb[b])} &gt; ${pct(pol.f)}`;
      else if (s > pol.s) why[b] = `safety UB ${pct(s, 1)} &gt; ${pct(pol.s, 1)}`;
      else { adm.push(b); why[b] = "allowed"; }
    }
    let best = adm[0];
    for (const b of adm) {
      const cb = cell.cost[b], cc = cell.cost[best];
      if (cb < cc - 1e-12 || (Math.abs(cb - cc) < 1e-12 && LADDER.indexOf(b) < LADDER.indexOf(best))) best = b;
    }
    return { rung: best, why };
  }

  function renderPolicy() {
    $("v-d").textContent = pol.d.toFixed(3);
    $("v-f").textContent = pol.f.toFixed(2);
    $("v-s").textContent = pol.s.toFixed(3);
    $("sl-d").value = pol.d; $("sl-f").value = pol.f; $("sl-s").value = pol.s;

    let html = `<thead><tr><th></th>${[0, 1, 2, 3, 4, 5, 6].map((k) => `<th>${k}</th>`).join("")}</tr></thead><tbody>`;
    let below = 0, wsum = 0, wcost = 0;
    CATS.forEach((cat) => {
      html += `<tr><th class="rowh">${pretty(cat)}</th>`;
      for (let k = 0; k < 7; k++) {
        const cell = cells[`${cat}|${k}`];
        const { rung } = chooseCell(cell);
        if (rung !== "F16") below++;
        const w = cell.n_support_tasks;
        wsum += w; wcost += w * (cell.cost[rung] / cell.cost.F16);
        html += `<td class="${cell.low_support ? "dashed" : ""}" data-key="${cat}|${k}" style="background-color:${RC[rung]};color:${textOn(rung)}">${rung}</td>`;
      }
      html += `</tr>`;
    });
    $("pol-mx").innerHTML = html + `</tbody>`;
    $("pol-mx").querySelectorAll("td").forEach((td) => bindTip(td, () => {
      const cell = cells[td.dataset.key];
      const [cat, k] = td.dataset.key.split("|");
      const { rung, why } = chooseCell(cell);
      return `<b>${pretty(cat)} &middot; class ${k}</b> &rarr; <b>${rung}</b><br>${cell.n_support_tasks} training tasks${cell.low_support ? " (coarse fallback)" : ""}<br>` +
        NONREF.map((b) => `<div style="margin-top:2px"><span style="font-family:${MONO}">${b}</span> &middot; ${why[b]}</div>`).join("");
    }));

    let floor = "F16";
    for (const b of NONREF) if (D.rungs[b].safety_flip_ucb <= pol.s) { floor = b; break; }
    $("pol-kpis").innerHTML = [
      [`${below}/84`, "cells served below F16", "var(--mint)"],
      [pct(wsum ? wcost / wsum : 1), "cost vs always-F16 (task-weighted)", "var(--sun)"],
      [floor, "safety floor for flagged prompts", "var(--blush)"],
      [String(NONREF.filter((b) => D.rungs[b].safety_flip_ucb > pol.s).length), "rungs banned by safety", "#e9e3d6"],
    ].map(([n, l, bg]) => `<div class="k" style="background:${bg}"><div class="n" style="font-size:22px">${n}</div><div class="l">${l}</div></div>`).join("");

    const accBlocked = NONREF.reduce((n, b) => n + CATS.reduce((m, cat) => {
      let k = 0;
      for (let c = 0; c < 7; c++) { const cell = cells[`${cat}|${c}`]; if (cell.acc_lcb[b] < cell.acc_ref - pol.d) k++; }
      return m + k;
    }, 0), 0);
    const flipBlocked = NONREF.reduce((n, b) => n + CATS.reduce((m, cat) => {
      let k = 0;
      for (let c = 0; c < 7; c++) { const cell = cells[`${cat}|${c}`]; if (cell.flip_ucb[b] > pol.f) k++; }
      return m + k;
    }, 0), 0);
    $("pol-hint").innerHTML = `<span class="note-tag">WHAT'S BLOCKING</span><span>Of 504 (cell, rung) checks, the accuracy bar blocks <strong>${accBlocked}</strong> and the flip bar blocks <strong>${flipBlocked}</strong>.` +
      (below === 0 ? " Every cell falls back to F16 &mdash; try <strong>Balanced</strong> or <strong>Loose</strong>." : "") + `</span>`;

    if (window.__sflip) { window.__sflip.$hline = { value: pol.s * 100, label: `ψ ${pct(pol.s, 1)}` }; window.__sflip.update(); }
  }
  [["sl-d", "d"], ["sl-f", "f"], ["sl-s", "s"]].forEach(([id, key]) => $(id).addEventListener("input", (e) => {
    pol[key] = Number(e.target.value);
    $("presets").querySelectorAll(".pill").forEach((x) => x.classList.remove("on"));
    renderPolicy();
  }));
  $("presets").querySelectorAll(".pill").forEach((b) => b.addEventListener("click", () => {
    $("presets").querySelectorAll(".pill").forEach((x) => x.classList.toggle("on", x === b));
    Object.assign(pol, PRESETS[b.dataset.p]); renderPolicy();
  }));

  // ---------------------------------------------------------------- safety
  function safetyBars(id, key) {
    new Chart($(id), {
      type: "bar",
      data: { labels: LADDER, datasets: [bars(LADDER.map((r) => (D.rungs[r][key] ?? 0) * 100), LADDER.map((r) => RC[r]), { vfmt: (v) => v.toFixed(0) + "%" })] },
      options: { layout: { padding: { top: 16 } }, scales: { x: Object.assign({}, xCat, { ticks: { color: INK, font: { family: MONO, size: 11, weight: "700" }, callback: (v, i) => SHORT[LADDER[i]] } }), y: yVal({ beginAtZero: true, ticks: { color: MUTED, callback: (v) => v + "%" } }) } },
    });
  }
  safetyBars("c-s1", "harmful_refusal");
  safetyBars("c-s2", "over_refusal");
  safetyBars("c-s3", "injection_resistance");
  window.__sflip = new Chart($("c-sflip"), {
    type: "bar",
    data: { labels: NONREF, datasets: [bars(NONREF.map((r) => D.rungs[r].safety_flip_ucb * 100), NONREF.map((r) => RC[r]), { vfmt: (v) => v.toFixed(1) + "%" })] },
    options: { layout: { padding: { top: 16, right: 70 } }, scales: { x: xCat, y: yVal({ beginAtZero: true, suggestedMax: 8, ticks: { color: MUTED, callback: (v) => v + "%" } }) } },
  });
  (function clf() {
    const c = D.classifier;
    $("clf-box").innerHTML = `<div class="chart-title">Safety classifier</div><div class="chart-sub">Flags risky prompts so they are never served below the safety floor. Tuned for 95% recall.</div>
      <div class="kpis" style="grid-template-columns:1fr 1fr;margin:0">
        <div class="k" style="background:var(--mint)"><div class="n">${pct(c.test_recall)}</div><div class="l">of harmful test prompts caught</div></div>
        <div class="k" style="background:var(--mint)"><div class="n">${pct(c.test_fpr_bench)}</div><div class="l">normal tasks wrongly flagged</div></div>
        <div class="k" style="background:var(--blush)"><div class="n">${pct(c.test_fpr_benign_lookalike)}</div><div class="l">harmless look-alikes wrongly flagged</div></div>
        <div class="k" style="background:#e9e3d6"><div class="n">${(c.threshold ?? 0).toFixed(2)}</div><div class="l">decision threshold</div></div>
      </div>`;
  })();

  // ---------------------------------------------------------------- predictor
  const P = D.predictor;
  const PRED_LABELS = ["This predictor", "Most-common class", "Category median"];
  function predBars(id, vals, fmt, max) {
    new Chart($(id), {
      type: "bar",
      data: { labels: PRED_LABELS, datasets: [bars(vals, [INK, GREY, GREY], { borderSkipped: "start", maxBarThickness: 34, vfmt: fmt, vmode: "h" })] },
      options: { indexAxis: "y", layout: { padding: { right: 54 } },
        scales: { y: { grid: { display: false }, border: { color: INK, width: 2 }, ticks: { color: INK, font: { weight: "700" } } },
          x: { grid: { color: GRID }, border: { display: false }, ticks: { color: MUTED }, beginAtZero: true, max } } },
    });
  }
  predBars("c-p1", [P.test_exact_accuracy * 100, P.baseline_majority_class_accuracy * 100, P.baseline_per_category_median_accuracy * 100], (v) => v.toFixed(1) + "%", 100);
  predBars("c-p2", [P.test_mae, P.baseline_majority_class_mae, P.baseline_per_category_median_mae], (v) => v.toFixed(2), undefined);
  $("pred-note").innerHTML = `<span class="note-tag">VERDICT</span><span>The predictor lands within one rung ${pct(P.test_within_one_accuracy)} of the time, but its exact hit-rate (${pct(P.test_exact_accuracy, 1)}) only ties always guessing the most common class (${pct(P.baseline_majority_class_accuracy, 1)}), and simply guessing each category's median beats it on <em>both</em> measures (${pct(P.baseline_per_category_median_accuracy, 1)} exact, ${P.baseline_per_category_median_mae.toFixed(2)} MAE). Prompt text alone is a weak signal for required precision &mdash; reported as measured.</span>`;

  // ---------------------------------------------------------------- nav scrollspy
  const links = Array.from(document.querySelectorAll("#nav a"));
  const byId = Object.fromEntries(links.map((a) => [a.getAttribute("href").slice(1), a]));
  const spy = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting && byId[en.target.id]) {
        links.forEach((a) => a.classList.remove("active"));
        byId[en.target.id].classList.add("active");
      }
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  document.querySelectorAll("section.block").forEach((s) => spy.observe(s));

  // ---------------------------------------------------------------- initial render
  selectRung(selRung);
  renderMatrix();
  renderCellDetail();
  renderAcc();
  renderNeed();
  renderPolicy();
})();
