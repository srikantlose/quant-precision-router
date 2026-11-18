(function () {
  "use strict";

  const D = window.QL;          // policy cells and per-rung safety bounds (data.js)
  const X = window.QLDEMO;      // 450 held-out questions with recorded answers (demo-data.js)
  const LADDER = D.meta.ladder;
  const NONREF = LADDER.slice(0, 6);
  const RC = { Q2_K: "#ff8d65", Q3_K_M: "#ec6d3d", Q4_K_M: "#d05320", Q5_K_M: "#b13d02", Q6_K: "#8e2e00", Q8_0: "#6b2101", F16: "#151515" };
  const textOn = (r) => (r === "Q2_K" || r === "Q3_K_M" ? "#151515" : "#ffffff");
  const pretty = (c) => c.replace(/_/g, " ");
  const pct = (x, d = 0) => (x * 100).toFixed(d) + "%";
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const chip = (r) => `<span class="rchip" style="background:${RC[r]};color:${textOn(r)}">${r}</span>`;
  const mark = (ok) => (ok ? '<span class="ok">✓ correct</span>' : '<span class="no">✗ wrong</span>');
  const idx = (r) => LADDER.indexOf(r);

  // ------------------------------------------------------------ policy (same rule as the dashboard's policy lab)
  const PRESETS = { strict: { d: 0.02, f: 0.1, s: 0.05 }, balanced: { d: 0.1, f: 0.3, s: 0.1 }, loose: { d: 0.15, f: 1, s: 1 } };
  const pol = Object.assign({}, PRESETS.balanced);

  function gates(cell) {
    const out = {};
    for (const b of NONREF) {
      const acc = cell.acc_lcb[b] >= cell.acc_ref - pol.d;
      const flip = cell.flip_ucb[b] <= pol.f;
      const safe = D.rungs[b].safety_flip_ucb <= pol.s;
      out[b] = { acc, flip, safe, ok: acc && flip && safe };
    }
    return out;
  }
  function chooseCell(cell) {
    const g = gates(cell), adm = ["F16"].concat(NONREF.filter((b) => g[b].ok));
    let best = adm[0];
    for (const b of adm) {
      const cb = cell.cost[b], cc = cell.cost[best];
      if (cb < cc - 1e-12 || (Math.abs(cb - cc) < 1e-12 && idx(b) < idx(best))) best = b;
    }
    return { rung: best, gates: g, nAllowed: adm.length - 1 };
  }
  function floorRung() {
    for (const b of NONREF) if (D.rungs[b].safety_flip_ucb <= pol.s) return b;
    return "F16";
  }
  function route(t, pretendFlag) {
    const cell = D.policy.cells[`${t.catp}|${t.cls}`];
    const c = chooseCell(cell);
    const realFlag = t.score >= X.meta.threshold;
    const flagged = realFlag || pretendFlag;
    const floor = floorRung();
    const served = flagged && idx(c.rung) < idx(floor) ? floor : c.rung;
    return { cell, choice: c.rung, gates: c.gates, nAllowed: c.nAllowed, flagged, realFlag, floor, served };
  }
  const ans = (t, r) => { const a = t.a[idx(r)]; return { ok: !!a[0], lat: a[1], tok: a[2], text: a[3] }; };

  // ------------------------------------------------------------ controls
  function syncControls() {
    $("v-d").textContent = pol.d.toFixed(3);
    $("v-f").textContent = pol.f.toFixed(2);
    $("v-s").textContent = pol.s.toFixed(3);
    $("sl-d").value = pol.d; $("sl-f").value = pol.f; $("sl-s").value = pol.s;
    $("presets").querySelectorAll(".pill").forEach((b) => {
      const p = PRESETS[b.dataset.p];
      b.classList.toggle("on", p.d === pol.d && p.f === pol.f && p.s === pol.s);
    });
  }
  [["sl-d", "d"], ["sl-f", "f"], ["sl-s", "s"]].forEach(([id, key]) => $(id).addEventListener("input", (e) => { pol[key] = Number(e.target.value); refresh(); }));
  $("presets").querySelectorAll(".pill").forEach((b) => b.addEventListener("click", () => { Object.assign(pol, PRESETS[b.dataset.p]); refresh(); }));
  $("flag").addEventListener("change", () => renderSelected());

  // ------------------------------------------------------------ picker
  const cats = [...new Set(X.tasks.map((t) => t.cat))].sort();
  $("cat").innerHTML = `<option value="">All categories</option>` + cats.map((c) => `<option value="${c}">${pretty(c)}</option>`).join("");
  let selected = (X.tasks.find((t) => t.id === "ext-07") || X.tasks[0]).id;

  function matches(t) {
    const q = $("q").value.trim().toLowerCase();
    if (q && !t.p.toLowerCase().includes(q)) return false;
    if ($("cat").value && t.cat !== $("cat").value) return false;
    const only = $("only").value;
    if (only === "all") return true;
    if (only === "miscat") return t.catp !== t.cat;
    const r = route(t, false);
    if (r.served === "F16") return false;
    const s = ans(t, r.served), f = ans(t, "F16");
    if (only === "cheap") return true;
    if (only === "savedright") return s.ok;
    if (only === "flip") return !s.ok && f.ok;
    return true;
  }
  function renderList() {
    const list = X.tasks.filter(matches);
    $("count").textContent = `${list.length} of ${X.tasks.length} questions`;
    $("qlist").innerHTML = list.map((t) => {
      const r = route(t, false), a = ans(t, r.served);
      return `<button type="button" class="qitem${t.id === selected ? " sel" : ""}" data-id="${t.id}">
        <span class="t">${esc(t.p.length > 110 ? t.p.slice(0, 110) + "…" : t.p)}</span>
        <span class="m">${chip(r.served)}<span class="${a.ok ? "ok" : "no"}">${a.ok ? "✓" : "✗"}</span><span>${pretty(t.cat)} · ${t.tier}</span></span></button>`;
    }).join("") || `<p class="small muted">No questions match. Clear the search or pick another filter.</p>`;
    $("qlist").querySelectorAll(".qitem").forEach((b) => b.addEventListener("click", () => { selected = b.dataset.id; $("flag").checked = false; renderList(); renderSelected(); }));
  }
  ["q", "cat", "only"].forEach((id) => $(id).addEventListener("input", renderList));
  $("rand").addEventListener("click", () => {
    const list = X.tasks.filter(matches), pool = list.length ? list : X.tasks;
    selected = pool[Math.floor(Math.random() * pool.length)].id;
    $("flag").checked = false; renderList(); renderSelected();
    const el = $("qlist").querySelector(".qitem.sel"); if (el) el.scrollIntoView({ block: "nearest" });
  });

  // ------------------------------------------------------------ selected question
  function renderSelected() {
    const t = X.tasks.find((x) => x.id === selected);
    const r = route(t, $("flag").checked);
    const s = ans(t, r.served), f = ans(t, "F16");
    const need = t.req == null ? null : LADDER[t.req];

    $("qcard").innerHTML = `<div class="kicker">Question ${esc(t.id)}</div>
      <div class="qtext">${esc(t.p)}</div>
      <div class="meta"><span>Category <b>${pretty(t.cat)}</b></span><span>Difficulty <b>${t.tier}</b></span><span>Graded by <b class="mono">${t.grader}</b></span><span>Expected <b>${esc(t.exp)}</b></span></div>`;

    const steps = [
      { k: "1 · Detected category", v: pretty(t.catp), warn: t.catp !== t.cat,
        d: t.catp === t.cat ? "Matches the question's true category." : `The true category is ${pretty(t.cat)}, so it is routed as ${pretty(t.catp)}.` },
      { k: "2 · Predicted precision", v: `needs ~${LADDER[t.cls]}`,
        d: `From the question text alone. Its recorded answers show it actually needed ${need ? need : "no version (unsolvable)"}.` },
      { k: "3 · Policy cell", v: `${pretty(t.catp)} · ${t.cls}`,
        d: `${r.nAllowed} of 6 cheaper versions pass all three checks${r.cell.low_support ? " (sparse cell: category-level estimate)" : ""}. Cheapest allowed: ${r.choice}.` },
      { k: "4 · Safety check", v: r.flagged ? "Flagged" : "Not flagged", warn: r.flagged,
        d: `Classifier score ${t.score.toFixed(3)} vs threshold ${X.meta.threshold.toFixed(3)}${$("flag").checked && !r.realFlag ? " (flag forced by the switch)" : ""}.` +
           (r.flagged ? (r.served !== r.choice ? ` Raised to the safety floor ${r.floor}.` : ` Already at or above the floor ${r.floor}.`) : "") +
           (r.flagged && r.floor === "Q2_K" ? " At this safety limit every version passes, so the floor is the bottom rung; choose Strict to see the floor rise to Q5_K_M." : "") },
      { k: "5 · Served by", v: chip(r.served), final: true, d: `Answered in ${s.lat.toFixed(3)} s with ${s.tok} tokens. ${mark(s.ok)}` },
    ];
    $("pipe").innerHTML = steps.map((st) => `<div class="st${st.final ? " final" : ""}${st.warn ? " warn" : ""}"><span class="k">${st.k}</span><span class="v">${st.v}</span><span class="d">${st.d}</span></div>`).join("");

    let other = "F16", otherTitle = "Full precision (F16) for comparison";
    if (r.served === "F16") {
      const cheapest = NONREF.find((b) => ans(t, b).ok);
      if (cheapest) { other = cheapest; otherTitle = "Cheapest version that happened to be right"; }
      else { other = "Q4_K_M"; otherTitle = "A common fixed choice (Q4_K_M)"; }
    }
    const o = ans(t, other);
    const card = (title, rung, a) => `<div class="card ans"><div class="kicker">${title}</div><div class="hd">${chip(rung)}${mark(a.ok)}</div><div class="txt">${esc(a.text) || "<em>(empty answer)</em>"}</div><div class="st2">${a.lat.toFixed(3)} s · ${a.tok} tokens</div></div>`;
    $("compare").innerHTML = card("QuantRoute serves", r.served, s) + card(otherTitle, other, o);

    const v = $("verdict");
    let cls = "info", tag = "RESULT", msg;
    if (r.served === "F16") {
      msg = r.nAllowed === 0 ? "Served at F16: no cheaper version passes all three checks for this cell at these thresholds." : "Served at F16.";
      if (other !== "Q4_K_M" && ans(t, other).ok) msg += ` In hindsight ${other} was also right, but the policy can only act on what it could be sure of in advance for this kind of question.`;
    } else if (s.ok && f.ok) {
      cls = "good";
      msg = s.lat < f.lat ? `Same result as F16 with ${pct(1 - s.lat / f.lat)} less GPU time (${s.lat.toFixed(3)} s vs ${f.lat.toFixed(3)} s), on a model ${pct(D.rungs[r.served].vram_mb / D.rungs.F16.vram_mb)} the size in memory.`
        : `Same result as F16 on a model ${pct(D.rungs[r.served].vram_mb / D.rungs.F16.vram_mb)} the size in memory, though this particular answer took a little longer (${s.lat.toFixed(3)} s vs ${f.lat.toFixed(3)} s).`;
    } else if (!s.ok && f.ok) {
      cls = ""; tag = "A FLIP";
      msg = "The cheaper version got this wrong where F16 was right. The flip limit φ caps how often this may happen in each cell; tighten it to make the policy more careful.";
    } else if (s.ok && !f.ok) {
      cls = "good"; msg = "The cheaper version was right where F16 was wrong. Flips go both ways.";
    } else {
      msg = "Both the served version and F16 got this one wrong.";
    }
    v.className = `note verdict ${cls}`;
    v.innerHTML = `<span class="note-tag">${tag}</span><span>${msg}</span>`;

    const gateChip = (ok, name) => `<span class="gate ${ok ? "pass" : "fail"}">${name} ${ok ? "✓" : "✗"}</span>`;
    $("all").innerHTML = `<thead><tr><th>Version</th><th>Recorded answer</th><th>Result</th><th>Time</th><th>Checks for this cell</th></tr></thead><tbody>` +
      LADDER.map((rg) => {
        const a = ans(t, rg), g = r.gates[rg];
        const checks = rg === "F16" ? '<span class="gate ref">reference</span>' : `${gateChip(g.acc, "accuracy")} ${gateChip(g.flip, "flips")} ${gateChip(g.safe, "safety")}`;
        return `<tr class="${rg === r.served ? "served" : ""}"><td class="n">${chip(rg)}${rg === need ? '<span class="need">NEEDED</span>' : ""}</td>
          <td class="a"><div>${esc(a.text) || "<em>(empty)</em>"}</div></td><td class="n">${a.ok ? '<span class="ok">✓</span>' : '<span class="no">✗</span>'}</td>
          <td class="n mono">${a.lat.toFixed(3)} s</td><td>${checks}</td></tr>`;
      }).join("") + `</tbody>`;
  }

  // ------------------------------------------------------------ totals over all 450
  function renderTotals() {
    const N = X.tasks.length;
    const use = Object.fromEntries(LADDER.map((r) => [r, 0]));
    const tot = { me: [0, 0], f16: [0, 0], q4: [0, 0], best: [0, 0] };
    let below = 0, flips = 0;
    for (const t of X.tasks) {
      const r = route(t, false), s = ans(t, r.served), f = ans(t, "F16"), q4 = ans(t, "Q4_K_M");
      use[r.served]++;
      if (r.served !== "F16") below++;
      if (s.ok !== f.ok) flips++;
      tot.me[0] += s.ok; tot.me[1] += s.lat;
      tot.f16[0] += f.ok; tot.f16[1] += f.lat;
      tot.q4[0] += q4.ok; tot.q4[1] += q4.lat;
      const cheapest = LADDER.find((b) => ans(t, b).ok) || "F16", h = ans(t, cheapest);
      tot.best[0] += h.ok; tot.best[1] += h.lat;
    }
    const F = tot.f16[1];
    const cardT = (name, [ok, time], extra, me) => `<div class="card tot${me ? " me" : ""}"><div class="name">${name}</div><div class="big2">${pct(ok / N, 1)}</div><div class="sub">correct</div>
      <div class="sub"><b>${time.toFixed(0)} s</b> GPU time · ${pct(time / F)} of F16</div>${extra ? `<div class="sub">${extra}</div>` : ""}</div>`;
    $("totals").innerHTML =
      cardT("QuantRoute", tot.me, `${below} of ${N} served below F16 · ${flips} outcomes differ from F16`, true) +
      cardT("Always F16", tot.f16, "the full-precision reference") +
      cardT("Always Q4_K_M", tot.q4, "a common one-size-fits-all choice") +
      cardT("Cheapest correct (hindsight)", tot.best, "needs answers in advance");
    $("usage").innerHTML = LADDER.filter((r) => use[r]).map((r) => `<div style="width:${(use[r] / N) * 100}%;background:${RC[r]};color:${textOn(r)}" title="${r}: ${use[r]} questions">${use[r] / N > 0.07 ? `${r} · ${use[r]}` : ""}</div>`).join("");
    $("ulegend").innerHTML = LADDER.map((r) => `<span><span class="sw" style="background:${RC[r]}"></span>${r} ${use[r]}</span>`).join("");
  }

  function refresh() { syncControls(); renderList(); renderSelected(); renderTotals(); }
  refresh();
})();
