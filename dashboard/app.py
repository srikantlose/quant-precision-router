"""QuantRoute results dashboard.

Run:  .venv/Scripts/python.exe -m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501

Every page reads the files in results/ (lazily, cached). A missing file shows a notice
instead of an error. The Inspector page calls router.decide only: it never generates,
and no page shows raw completions from the safety slice.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
LADDER = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]
SHORT = {"Q2_K": "Q2", "Q3_K_M": "Q3", "Q4_K_M": "Q4", "Q5_K_M": "Q5", "Q6_K": "Q6", "Q8_0": "Q8", "F16": "F16"}
RC = {"Q2_K": "#ff8d65", "Q3_K_M": "#ec6d3d", "Q4_K_M": "#d05320", "Q5_K_M": "#b13d02", "Q6_K": "#8e2e00",
      "Q8_0": "#6b2101", "F16": "#151515"}
BLUE, AMBER, MUTED = "#3d74e0", "#c98a00", "#77726a"
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), font=dict(size=12),
              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))

st.set_page_config(page_title="QuantRoute results", layout="wide")


# ------------------------------------------------------------------------------ loading
@st.cache_data(show_spinner=False)
def load_json(name: str):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


@st.cache_data(show_spinner=False)
def load_csv(name: str):
    p = R / name
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return pd.DataFrame(list(csv.DictReader(f)))


def need(*names: str):
    """Load results files, or show a notice and return None if any is missing."""
    out = []
    for n in names:
        d = load_csv(n) if n.endswith(".csv") else load_json(n)
        if d is None:
            st.info(f"`results/{n}` is not there yet. Run the step that writes it "
                    f"(see the README's command sequence), then reload this page.")
            return None
        out.append(d)
    return out


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(x, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * float(x):.{digits}f}%"


def ops_and_maps():
    """Strict and Balanced operating points and the rung each policy cell serves."""
    from quantlab.policy import choose
    from quantlab.router import TOLERANCES

    policy = load_json("policy.json")
    safety = load_json("safety.json")
    base = {k: float(policy["operating_point"][k]) for k in ("delta", "phi", "psi")}
    ops = {t: {**base, **(v or {})} for t, v in TOLERANCES.items()}
    ucb = {r: float(safety[f"{r}|train"]["safety_flip_ucb"]) for r in LADDER}
    maps = {}
    for t, op in ops.items():
        maps[t] = {k: choose(c["acc_ref"], c["acc_lcb"], c["flip_ucb"], ucb, c["cost"], op["delta"], op["phi"], op["psi"])
                   for k, c in policy["cells"].items()}
    return ops, maps, ucb


# ------------------------------------------------------------------------------ pages
POLICY_ROWS = [("always_" + r, f"always {r}") for r in LADDER] + [
    ("best_fixed", "best fixed rung"), ("resident_subset", "resident subset"),
    ("learned", "learned policy"), ("oracle", "oracle (hindsight)")]


def page_overview():
    got = need("baselines.json")
    if not got:
        return
    (b,) = got
    boot = load_json("bootstrap.json")
    st.title("QuantRoute")
    st.caption("Qwen2.5-3B-Instruct served at seven GGUF precisions on one RTX 4060. Each prompt is routed to the "
               "cheapest precision whose accuracy, flip and safety bounds hold. Numbers are on the 450 held-out test tasks.")
    t = b["results"]
    c = st.columns(4)
    c[0].metric("Safety floor", b["safety_floor"])
    c[1].metric("Learned (Strict) accuracy", pct(t["strict"]["test"]["learned"]["accuracy"]),
                help="Strict serves F16 everywhere: no cell clears the fitted bounds.")
    c[2].metric("Learned (Balanced) accuracy", pct(t["balanced"]["test"]["learned"]["accuracy"]))
    c[3].metric("Always Q5_K_M accuracy", pct(t["strict"]["test"]["always_Q5_K_M"]["accuracy"]))

    for tol in ("strict", "balanced"):
        op = b["operating_points"][tol]
        st.subheader(f"{tol.capitalize()}  ·  δ {op['delta']}, φ {op['phi']}, ψ {op['psi']}")
        rows = []
        bo = boot["metric_intervals"]["by_operating_point"][tol] if boot else {}
        for key, name in POLICY_ROWS:
            m = t[tol]["test"][key]
            ci = bo.get(key, {})
            rows.append({
                "policy": name + (f" ({b['best_fixed'][tol]['rung']})" if key == "best_fixed" else "")
                + (f" ({'+'.join(b['resident_subset'][tol]['chosen'])})" if key == "resident_subset" else ""),
                "accuracy": pct(m["accuracy"]),
                "95% CI": f"{pct(ci['accuracy_ci']['lo'])} – {pct(ci['accuracy_ci']['hi'])}" if ci else "–",
                "flip rate": pct(m["flip_rate"]),
                "inference GPU-h/1k": round(m["inference_gpu_h_per_1k"], 3),
                "with swaps GPU-h/1k": round(m["total_gpu_h_per_1k"], 3),
                "mean bits/weight": round(m["mean_bpw"], 2),
                "harmful refusal": pct(m.get("harmful_refusal")),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(b.get("notes", ""))


def page_accuracy():
    got = need("grid_summary.json", "profiles.json")
    if not got:
        return
    grid, prof = got
    st.title("Accuracy vs bits")
    st.caption("All 1,467 bench tasks graded at every rung. Bands are 95% Wilson intervals.")
    bpw = {r: prof["rungs"][r]["bpw"] for r in LADDER}
    cats = sorted(grid["rungs"]["F16"]["by_category"])
    pick = st.multiselect("Categories", cats, default=["math", "code_gen", "sentiment", "logic"])
    cols = st.columns(2)
    for i, cat in enumerate(pick):
        xs = [bpw[r] for r in LADDER]
        acc = [grid["rungs"][r]["by_category"][cat]["accuracy"] for r in LADDER]
        ns = [grid["rungs"][r]["by_category"][cat]["n"] for r in LADDER]
        lo, hi = zip(*[wilson(a, n) for a, n in zip(acc, ns)])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[100 * v for v in hi] + [100 * v for v in lo][::-1], fill="toself",
                                 fillcolor="rgba(119,114,106,0.15)", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=[100 * a for a in acc], mode="lines+markers", line=dict(color=MUTED, width=2),
                                 marker=dict(size=10, color=[RC[r] for r in LADDER], line=dict(color="white", width=1)),
                                 text=LADDER, hovertemplate="%{text}: %{y:.1f}%<extra></extra>", showlegend=False))
        fig.update_layout(**LAYOUT, title=f"{cat.replace('_', ' ')} (n={ns[-1]})", height=320,
                          xaxis=dict(type="log", title="bits per weight", tickvals=xs, ticktext=[SHORT[r] for r in LADDER]),
                          yaxis=dict(title="accuracy %", range=[0, 100]))
        cols[i % 2].plotly_chart(fig, width="stretch")


def page_cost():
    got = need("grid_summary.json", "policy.json")
    if not got:
        return
    grid, policy = got
    st.title("Cost vs accuracy")
    st.caption("Cost is the measured mean latency per rung and category, as GPU-hours per 1,000 tasks. "
               "Filled points are Pareto-optimal; hollow points are dominated by a cheaper, more accurate rung.")
    cats = sorted(grid["rungs"]["F16"]["by_category"])
    cat = st.selectbox("Category", cats, index=cats.index("math"))
    cost = policy["cells"][f"{cat}|0"]["cost"]
    pts = [(cost[r], 100 * grid["rungs"][r]["by_category"][cat]["accuracy"], r) for r in LADDER]
    pareto = {p[2] for p in pts if not any(q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]) for q in pts)}
    front = sorted(p for p in pts if p[2] in pareto)
    fig = go.Figure(go.Scatter(x=[p[0] for p in front], y=[p[1] for p in front], mode="lines",
                               line=dict(color=MUTED, width=1.5), hoverinfo="skip", showlegend=False))
    for c, a, r in pts:
        on = r in pareto
        fig.add_trace(go.Scatter(x=[c], y=[a], mode="markers+text", text=[SHORT[r]], textposition="top right",
                                 marker=dict(size=14, color=RC[r] if on else "white", line=dict(color=RC[r], width=2.5)),
                                 name=r, hovertemplate=f"{r}<br>%{{x:.3f}} GPU-h/1k<br>%{{y:.1f}}%<extra></extra>"))
    fig.update_layout(**LAYOUT, height=460, showlegend=False,
                      xaxis=dict(title="GPU-hours per 1,000 tasks"), yaxis=dict(title="accuracy %"))
    st.plotly_chart(fig, width="stretch")


def page_policy():
    got = need("policy.json", "safety.json")
    if not got:
        return
    policy, _ = got
    st.title("Policy")
    ops, maps, ucb = ops_and_maps()
    tol = st.radio("Operating point", list(ops), horizontal=True, format_func=str.capitalize)
    op = ops[tol]
    st.caption(f"δ {op['delta']}, φ {op['phi']}, ψ {op['psi']}. Rows are task categories, columns the predicted "
               f"required-precision class (0 = Q2_K … 6 = F16). Hover for the admissible set and the bounds. "
               f"Flagged prompts are then raised to the {policy['safety_floor']} floor.")
    cats = sorted({k.split("|")[0] for k in policy["cells"]})
    z, text, hover = [], [], []
    for cat in cats:
        zr, tr, hr = [], [], []
        for k in range(7):
            key = f"{cat}|{k}"
            rung, adm = maps[tol][key]
            c = policy["cells"][key]
            bounds = "<br>".join(f"{r}: acc LCB {c['acc_lcb'][r]:.3f}, flip UCB {c['flip_ucb'][r]:.3f}, safety UCB {ucb[r]:.3f}"
                                 for r in LADDER[:-1])
            zr.append(LADDER.index(rung)); tr.append(SHORT[rung])
            hr.append(f"<b>{key} → {rung}</b><br>admissible: {', '.join(adm)}<br>support: {c.get('n_support_tasks')} tasks"
                      f"{' (coarse model)' if c.get('low_support') else ''}<br>acc bar: {c['acc_ref'] - op['delta']:.3f}<br>{bounds}")
        z.append(zr); text.append(tr); hover.append(hr)
    scale = [[i / 6, RC[r]] for i, r in enumerate(LADDER)]
    fig = go.Figure(go.Heatmap(z=z, x=[str(k) for k in range(7)], y=[c.replace("_", " ") for c in cats], text=text,
                               texttemplate="%{text}", customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                               colorscale=scale, zmin=0, zmax=6, showscale=False, xgap=2, ygap=2))
    fig.update_layout(**LAYOUT, height=560, xaxis=dict(title="predicted class"), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    counts = pd.Series([maps[tol][k][0] for k in policy["cells"]]).value_counts().reindex(LADDER).fillna(0).astype(int)
    st.write("Cells per rung:", ", ".join(f"{SHORT[r]} {n}" for r, n in counts.items() if n))


def page_sweep():
    got = need("sweep.csv", "baselines.json")
    if not got:
        return
    df, b = got
    st.title("Threshold sweep")
    for col in df.columns:
        if col != "split":
            df[col] = pd.to_numeric(df[col])
    split = st.radio("Split", ["test", "train"], horizontal=True)
    d = df[df["split"] == split]
    c = st.columns(3)
    delta = c[0].select_slider("δ accuracy tolerance", sorted(d["delta"].unique()), value=0.1)
    phi = c[1].select_slider("φ flip tolerance", sorted(d["phi"].unique()), value=0.3)
    psi = c[2].select_slider("ψ safety tolerance", sorted(d["psi"].unique()), value=0.1)
    row = d[(d["delta"] == delta) & (d["phi"] == phi) & (d["psi"] == psi)].iloc[0]
    f16 = b["results"]["strict"][split]["always_F16"]["inference_gpu_h_per_1k"]
    fig = go.Figure(go.Scatter(x=100 * d["inference_gpu_h_per_1k"] / f16, y=100 * d["accuracy"], mode="markers",
                               marker=dict(size=7, color=BLUE, opacity=0.4), name="learned policy, all operating points",
                               customdata=d[["delta", "phi", "psi"]].values,
                               hovertemplate="δ %{customdata[0]}, φ %{customdata[1]}, ψ %{customdata[2]}<br>%{x:.1f}% cost, %{y:.1f}%<extra></extra>"))
    for r in LADDER:
        m = b["results"]["strict"][split][f"always_{r}"]
        fig.add_trace(go.Scatter(x=[100 * m["inference_gpu_h_per_1k"] / f16], y=[100 * m["accuracy"]], mode="markers+text",
                                 text=[f"always {SHORT[r]}"], textposition="bottom right", name=r, showlegend=False,
                                 marker=dict(symbol="diamond", size=12, color=RC[r], line=dict(color="white", width=1))))
    fig.add_trace(go.Scatter(x=[100 * row["inference_gpu_h_per_1k"] / f16], y=[100 * row["accuracy"]], mode="markers",
                             marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color=AMBER, width=3)), name="selected"))
    fig.update_layout(**LAYOUT, height=480, xaxis=dict(title="inference time vs always-F16 %"), yaxis=dict(title="accuracy %"))
    st.plotly_chart(fig, width="stretch")
    m = st.columns(5)
    m[0].metric("accuracy", pct(row["accuracy"]))
    m[1].metric("flip rate", pct(row["flip_rate"]))
    m[2].metric("GPU-h/1k incl. swaps", f"{row['total_gpu_h_per_1k']:.3f}")
    m[3].metric("cells below F16", int(row["cells_below_f16"]))
    m[4].metric("harmful refusal", pct(row["harmful_refusal"]))


def page_safety():
    got = need("safety.json", "profiles.json")
    if not got:
        return
    safety, prof = got
    st.title("Safety per version")
    st.caption("Refusals are string-matched, as in the benchmark grid. The audit below shows how much that matcher misses.")
    bpw = {r: prof["rungs"][r]["bpw"] for r in LADDER}
    psi = load_json("policy.json")["operating_point"]["psi"]
    metrics = [("safety_flip_point", "safety flips vs F16 % (harmful + injection)"),
               ("over_refusal_rate", "over-refusal on benign look-alikes %"),
               ("injection_resistance_rate", "injection resistance %")]
    cols = st.columns(3)
    for col, (key, label) in zip(cols, metrics):
        fig = go.Figure()
        for split, color in (("train", "#151515"), ("test", MUTED)):
            ys = [100 * safety[f"{r}|{split}"][key] for r in LADDER]
            err = None
            if key == "safety_flip_point":
                err = dict(type="data", symmetric=False, array=[100 * safety[f"{r}|{split}"]["safety_flip_ucb"] - y
                                                               for r, y in zip(LADDER, ys)], arrayminus=[0] * 7)
            fig.add_trace(go.Scatter(x=[bpw[r] for r in LADDER], y=ys, mode="lines+markers", name=split,
                                     line=dict(color=color), error_y=err))
        if key == "safety_flip_point":
            fig.add_hline(y=100 * psi, line=dict(color=RC["Q4_K_M"], dash="dash"), annotation_text=f"ψ = {psi}")
        fig.update_layout(**LAYOUT, height=340, title=label,
                          xaxis=dict(type="log", tickvals=[bpw[r] for r in LADDER], ticktext=[SHORT[r] for r in LADDER]))
        col.plotly_chart(fig, width="stretch")

    audit = load_json("audit.json")
    st.subheader("Refusal-matcher audit")
    if not audit:
        st.info("`results/audit.json` is not there yet (run `quantlab audit-refusals`).")
        return
    c = st.columns(4)
    c[0].metric("agreement with judge", pct(audit["agreement"]))
    c[1].metric("Cohen's κ", f"{audit['cohen_kappa']:.2f}")
    c[2].metric("matcher misses (FN rate)", pct(audit["matcher_false_negative_rate"]))
    c[3].metric("false refusals (FP rate)", pct(audit["matcher_false_positive_rate"]))
    sub = pd.DataFrame(audit["by_subtype"]).T
    st.dataframe(sub.style.format("{:.2f}"), width="stretch")
    sens = audit.get("sensitivity")
    if sens:
        st.markdown("**Sensitivity: an extended matcher that also catches moralising refusals.** "
                    "Does the safety floor change?")
        rows = []
        for v, key in (("v1", "matcher_v1"), ("v2", "matcher_v2_extended")):
            h = sens["holdout"][key]
            rows.append({"matcher": "grid matcher (v1)" if v == "v1" else "extended matcher (v2)",
                         **{f"floor at {k.replace('psi=', 'ψ = ')}": r for k, r in sens["safety_floor"][v].items()},
                         "holdout agreement": pct(h["agreement"]), "holdout κ": round(h["cohen_kappa"], 3),
                         "holdout miss rate": pct(h["false_negative_rate"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(sens.get("note", ""))


def page_adversarial():
    got = need("adversarial.json")
    if not got:
        return
    (a,) = got
    st.title("Adversarial evaluation")
    st.caption(f"Attacks on the routing decision at the {a.get('tolerance', 'balanced')} operating point "
               f"(at Strict every cell already serves F16). Gadget suffixes are searched on train prompts only; each "
               f"defence is scored against the strongest gadget found under any defence. Floor = {a.get('floor')}.")
    tr = a.get("transfer", {})
    if "inflation" in a:
        st.subheader(f"Cost inflation ({a['sets']['inflation']} test tasks not already at F16)")
        rows = []
        for dname, d in a["inflation"].items():
            for att in ("pad", "stuff", "gadget_up"):
                m = tr[dname][att]["metrics"] if att.startswith("gadget") and tr else d[att]
                rows.append({"defence": dname, "attack": att, "rung raised": pct(m["pct_increased"]),
                             "routed to F16": pct(m["pct_routed_f16"]), "mean Δ rung": round(m["mean_delta_rung"], 2),
                             "added GPU-h/1k": round(m["added_cost_gpu_h_per_1k"], 4)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.subheader(f"Downgrade ({a['sets']['downgrade']} test harmful prompts)")
        rows = []
        for dname, d in a["downgrade"].items():
            for att in ("casual", "gadget_down"):
                m = tr[dname][att]["metrics"] if att.startswith("gadget") and tr else d[att]
                rows.append({"defence": dname, "attack": att, "below floor": pct(m["pct_below_floor"]),
                             "below floor (clean)": pct(m["clean_pct_below_floor"]), "mean Δ rung": round(m["mean_delta_rung"], 2),
                             "flag rate": pct(m["flag_rate"]), "flag rate (clean)": pct(m["clean_flag_rate"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.subheader("Gadgets found")
        st.dataframe(pd.DataFrame([{"defence": k, "direction": d, "gadget": v[d]["gadget"],
                                    "objective": f"{v[d]['initial_objective']:.2f} → {v[d]['objective']:.2f}"}
                                   for k, v in a["gadgets"].items() for d in ("gadget_up", "gadget_down")]),
                     hide_index=True, width="stretch")
        dc = a.get("defence_cost_clean")
        if dc:
            st.subheader("What normalisation costs on clean traffic")
            st.dataframe(pd.DataFrame({k: v for k, v in dc.items() if isinstance(v, dict)}).T, width="stretch")
    if "end_to_end" in a:
        st.subheader("End to end: refusal rate when generated at the decided rung")
        e = a["end_to_end"]["conditions"]
        st.dataframe(pd.DataFrame([{"condition": k, "refusal (grid matcher)": pct(v["refusal_rate"]),
                                    "refusal (extended matcher)": pct(v["refusal_rate_extended"]),
                                    "rungs": ", ".join(f"{SHORT[r]} {n}" for r, n in v["rungs"].items())}
                                   for k, v in e.items()]), hide_index=True, width="stretch")
    if "rate_limit" in a:
        rl = a["rate_limit"]
        st.subheader("Rate limit")
        st.write(f"{rl['n_requests']} padded requests decided at F16 from one key: **{rl['served']} served**, "
                 f"**{rl['rejected_429']} rejected with 429** (limits {rl['rate_per_min']}/min, F16 {rl['f16_rate_per_min']}/min).")


@st.cache_resource(show_spinner="Loading the router (predictor, classifier, embedder) …")
def get_router():
    from quantlab.router import Router
    return Router()


def page_inspector():
    st.title("Inspector")
    st.caption("Type a prompt to see the routing decision. This runs the classifier, predictor, policy and safety "
               "classifier only. Nothing is generated.")
    for f in ("policy.json", "safety.json", "profiles.json", "labels.json", "raw/predictor.joblib", "raw/safety_classifier.joblib"):
        if not (R / f).is_file():
            st.info(f"`results/{f}` is missing, so the router cannot load. Run `quantlab analyze` first.")
            return
    prompt = st.text_area("Prompt", "A train travels 120 km in 1.5 hours. What is its average speed in km/h?", height=120)
    c = st.columns(2)
    tol = c[0].radio("Tolerance", ["strict", "balanced"], horizontal=True, index=1)
    norm = c[1].checkbox("Normalise the routing input", value=False)
    if not prompt.strip():
        return
    d = get_router().decide(prompt, tol, normalize=norm)
    m = st.columns(5)
    m[0].metric("category", d.category)
    m[1].metric("predicted class", f"{d.pred_class} ({SHORT[d.predicted_required_precision]})")
    m[2].metric("cell rung", d.cell_rung)
    m[3].metric("served rung", d.rung)
    m[4].metric("safety flagged", "yes" if d.flagged else "no", f"score {d.safety_score:.2f}", delta_color="off")
    st.write(f"**Why:** {d.why}")
    gates = pd.DataFrame(d.trace["policy"]["gates"]).set_index("rung")
    st.dataframe(gates, width="stretch")


PAGES = {"Overview": page_overview, "Accuracy vs bits": page_accuracy, "Cost vs accuracy": page_cost,
         "Policy": page_policy, "Sweep": page_sweep, "Safety": page_safety, "Adversarial": page_adversarial,
         "Inspector": page_inspector}

choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption("Results from `results/`. Re-run `quantlab analyze` and reload to refresh.")
PAGES[choice]()
