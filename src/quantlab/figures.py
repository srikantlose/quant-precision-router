"""Report figures (plan P11.14), written to results/figures/.

Rung colours are one warm ordinal ramp (fewer bits = lighter, F16 = ink), the same
validated ramp the dashboard uses; single-series charts use one neutral colour. Twelve
categories are shown as small multiples rather than twelve hues.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .quants import LADDER  # noqa: E402

RC = {"Q2_K": "#ff8d65", "Q3_K_M": "#ec6d3d", "Q4_K_M": "#d05320", "Q5_K_M": "#b13d02", "Q6_K": "#8e2e00",
      "Q8_0": "#6b2101", "F16": "#151515"}
INK, MUTED, GRID, BLUE = "#151515", "#77726a", "#e6dfd2", "#3d74e0"
SHORT = {"Q2_K": "Q2", "Q3_K_M": "Q3", "Q4_K_M": "Q4", "Q5_K_M": "Q5", "Q6_K": "Q6", "Q8_0": "Q8", "F16": "F16"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight"})


def _load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fig_accuracy_vs_bpw(out: Path, grid: dict, bpw: dict) -> None:
    cats = sorted(grid["rungs"]["F16"]["by_category"])
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.5), sharex=True, sharey=True)
    for ax, cat in zip(axes.flat, cats):
        xs = [bpw[r] for r in LADDER]
        ys = [100 * grid["rungs"][r]["by_category"][cat]["accuracy"] for r in LADDER]
        ax.plot(xs, ys, color=MUTED, lw=1.5, zorder=1)
        ax.scatter(xs, ys, c=[RC[r] for r in LADDER], s=34, zorder=2, edgecolors="white", linewidths=0.8)
        ax.set_title(f"{cat.replace('_', ' ')}  (n={grid['rungs']['F16']['by_category'][cat]['n']})", fontsize=9, color=INK)
        ax.set_xscale("log"); ax.set_ylim(0, 100); ax.minorticks_off()
    for ax in axes[-1]:
        ax.set_xlabel("bits per weight (log)")
        ax.set_xticks([4, 6, 9, 17.6]); ax.set_xticklabels(["4", "6", "9", "17.6"])
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy %")
    fig.suptitle("Accuracy against bits per weight, by category (all 1,467 tasks; dots coloured Q2_K → F16)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(out / "accuracy_vs_bpw.png"); plt.close(fig)


def fig_cost_vs_accuracy(out: Path, grid: dict, cells: dict) -> None:
    cats = sorted(grid["rungs"]["F16"]["by_category"])
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.5))
    for ax, cat in zip(axes.flat, cats):
        cost = cells[f"{cat}|0"]["cost"]
        pts = [(cost[r], 100 * grid["rungs"][r]["by_category"][cat]["accuracy"], r) for r in LADDER]
        pareto = [p for p in pts if not any((q[0] <= p[0] and q[1] >= p[1]) and (q[0] < p[0] or q[1] > p[1]) for q in pts)]
        front = sorted(pareto)
        ax.plot([p[0] for p in front], [p[1] for p in front], color=MUTED, lw=1.2, zorder=1)
        for c, a, r in pts:
            on = (c, a, r) in pareto
            ax.scatter([c], [a], s=40, zorder=2, facecolors=RC[r] if on else "white", edgecolors=RC[r], linewidths=1.6)
            ax.annotate(SHORT[r], (c, a), xytext=(3, 3), textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(cat.replace("_", " "), fontsize=9, color=INK)
    for ax in axes[-1]:
        ax.set_xlabel("GPU-hours per 1,000 tasks")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy %")
    fig.suptitle("Cost against accuracy per category: filled = Pareto-optimal, hollow = dominated", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(out / "cost_vs_accuracy.png"); plt.close(fig)


def fig_policy_heatmap(out: Path, maps: dict[str, dict[str, str]]) -> None:
    cats = sorted({k.split("|")[0] for k in next(iter(maps.values()))})
    fig, axes = plt.subplots(1, len(maps), figsize=(6.2 * len(maps), 5.6))
    for ax, (name, m) in zip(axes, maps.items()):
        ax.grid(False)
        for i, cat in enumerate(cats):
            for k in range(7):
                r = m[f"{cat}|{k}"]
                ax.add_patch(plt.Rectangle((k, len(cats) - 1 - i), 0.94, 0.9, color=RC[r]))
                ax.text(k + 0.47, len(cats) - 1 - i + 0.45, SHORT[r], ha="center", va="center", fontsize=7.5,
                        color="#151515" if r in ("Q2_K", "Q3_K_M") else "white", fontweight="bold")
        ax.set_xlim(0, 7); ax.set_ylim(0, len(cats))
        ax.set_xticks([k + 0.47 for k in range(7)]); ax.set_xticklabels([str(k) for k in range(7)])
        ax.set_yticks([len(cats) - 1 - i + 0.45 for i in range(len(cats))]); ax.set_yticklabels([c.replace("_", " ") for c in cats])
        ax.set_xlabel("predicted required-precision class (0 = Q2_K … 6 = F16)")
        ax.set_title(f"{name.capitalize()} policy", fontsize=10, color=INK, loc="left")
    fig.suptitle("Version served by each of the 84 cells", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(out / "policy_heatmap.png"); plt.close(fig)


def fig_baselines(out: Path, boot: dict) -> None:
    ops = boot["metric_intervals"]["by_operating_point"]
    rows = [(f"always {SHORT[r]}", ops["balanced"][f"always_{r}"], RC[r]) for r in LADDER]
    rows += [("best fixed (strict)", ops["strict"]["best_fixed"], BLUE), ("best fixed (balanced)", ops["balanced"]["best_fixed"], BLUE),
             ("resident subset (bal.)", ops["balanced"]["resident_subset"], BLUE),
             ("learned, strict", ops["strict"]["learned"], INK), ("learned, balanced", ops["balanced"]["learned"], INK),
             ("oracle (hindsight)", ops["balanced"]["oracle"], MUTED)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    y = list(range(len(rows)))[::-1]
    for ax, key, label, scale in ((axes[0], "accuracy", "test accuracy % (95% CI)", 100), (axes[1], "cost_vs_f16", "inference time vs F16 % (95% CI)", 100)):
        for yi, (name, m, col) in zip(y, rows):
            v, ci = scale * m[key], m[f"{key}_ci"]
            ax.barh(yi, v, color=col, height=0.62)
            ax.errorbar(v, yi, xerr=[[v - scale * ci["lo"]], [scale * ci["hi"] - v]], color=INK, capsize=3, lw=1)
            ax.text(scale * ci["hi"] + 1, yi, f"{v:.1f}", va="center", fontsize=7.5, color=INK)
        ax.set_xlabel(label); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y); axes[0].set_yticklabels([r[0] for r in rows])
    fig.suptitle("Policies on the 450 held-out test tasks (bootstrap, 1,000 resamples)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(out / "baselines.png"); plt.close(fig)


def fig_safety(out: Path, safety: dict, bpw: dict, psi: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [bpw[r] for r in LADDER]
    for split, mk in (("train", "o"), ("test", "s")):
        xs = [bpw[r] * (0.97 if split == "train" else 1.03) for r in LADDER]
        pt = [100 * safety[f"{r}|{split}"]["safety_flip_point"] for r in LADDER]
        ucb = [100 * safety[f"{r}|{split}"]["safety_flip_ucb"] for r in LADDER]
        axes[0].errorbar(xs, pt, yerr=[[0] * len(pt), [u - p for u, p in zip(ucb, pt)]], fmt=mk, color=INK if split == "train" else MUTED,
                         capsize=3, label=f"{split}: point and upper bound", ms=5)
        axes[1].plot(xs, [100 * safety[f"{r}|{split}"]["over_refusal_rate"] for r in LADDER], marker=mk, color=INK if split == "train" else MUTED, label=split)
    axes[0].axhline(100 * psi, color=RC["Q4_K_M"], ls="--", lw=1.2)
    axes[0].text(xs[-1], 100 * psi + 0.3, f"ψ = {psi:.2f}", ha="right", fontsize=8, color=INK)
    for ax in axes:
        ax.set_xscale("log"); ax.minorticks_off()
        ax.set_xticks([bpw[r] for r in LADDER]); ax.set_xticklabels([SHORT[r] for r in LADDER])
    axes[0].set_ylabel("safety flips vs F16 % (harmful + injection)"); axes[0].legend(fontsize=7.5, frameon=False)
    axes[1].set_ylabel("over-refusal on benign look-alikes %"); axes[1].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Safety per version (string-matched refusals; see the refusal audit)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(out / "safety_vs_bpw.png"); plt.close(fig)


def fig_sweep(out: Path, sweep_rows: list[dict], base: dict) -> None:
    test = [r for r in sweep_rows if r["split"] == "test"]
    fig, ax = plt.subplots(figsize=(8, 5))
    f16 = base["results"]["strict"]["test"]["always_F16"]["inference_gpu_h_per_1k"]
    ax.scatter([100 * float(r["inference_gpu_h_per_1k"]) / f16 for r in test], [100 * float(r["accuracy"]) for r in test],
               s=14, color=BLUE, alpha=0.45, label="learned policy, 240 operating points")
    for r in LADDER:
        m = base["results"]["strict"]["test"][f"always_{r}"]
        ax.scatter([100 * m["inference_gpu_h_per_1k"] / f16], [100 * m["accuracy"]], marker="D", s=46, color=RC[r], edgecolors="white", zorder=3)
        off = {"Q4_K_M": (-8, -13), "Q5_K_M": (4, 5)}.get(r, (5, -10))
        ax.annotate(f"always {SHORT[r]}", (100 * m["inference_gpu_h_per_1k"] / f16, 100 * m["accuracy"]), xytext=off,
                    textcoords="offset points", fontsize=7.5, ha="right" if off[0] < 0 else "left")
    ax.set_xlabel("inference time vs always-F16 %"); ax.set_ylabel("test accuracy %")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Threshold sweep on test: operating points stay within the noise of the fixed versions", loc="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(out / "sweep_frontier.png"); plt.close(fig)


def make_figures(results_dir: str | Path = "results") -> list[str]:
    from .evaluate import EvalData
    R = Path(results_dir)
    out = R / "figures"
    out.mkdir(parents=True, exist_ok=True)
    grid, profiles, policy = _load(R / "grid_summary.json"), _load(R / "profiles.json"), _load(R / "policy.json")
    bpw = {r: profiles["rungs"][r]["bpw"] for r in LADDER}
    fig_accuracy_vs_bpw(out, grid, bpw)
    fig_cost_vs_accuracy(out, grid, policy["cells"])
    D = EvalData(R)
    fig_policy_heatmap(out, {name: {k: r for k, (r, _) in D.cell_map(op).items()} for name, op in D.ops.items()})
    if (R / "bootstrap.json").is_file():
        fig_baselines(out, _load(R / "bootstrap.json"))
    fig_safety(out, _load(R / "safety.json"), bpw, policy["operating_point"]["psi"])
    with open(R / "sweep.csv", encoding="utf-8") as f:
        fig_sweep(out, list(csv.DictReader(f)), _load(R / "baselines.json"))
    return sorted(p.name for p in out.glob("*.png"))
