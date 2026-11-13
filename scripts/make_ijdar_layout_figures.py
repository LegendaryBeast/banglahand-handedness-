#!/usr/bin/env python3
"""IJDAR re-layouts of the PRL result figures, drawn for the full two-column text width.

The data, statistics and plotted values are those of make_prl_figures_v2.py (results/final42_2026-10-19); only the
arrangement changes: the main-results strip is taller with the legend outside the data, the mechanism figure puts the
permutation nulls on a full-width row above the block ablation and the mirror test, and the STROKE-difference figure
puts the slant histogram and the hinge map above a full-width strip of all bins. Writes into manuscripts/ijdar/figures.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_prl_figures_v2 as P  # noqa: E402

RUN, OUT = ROOT / "results/final42_2026-10-19", ROOT / "manuscripts/ijdar/figures"
W = 7.16                                                    # drawn at 7.16 in, placed at \textwidth (6.30 in)
plt.rcParams.update({"font.size": 7.4, "axes.titlesize": 7.8, "axes.labelsize": 7.2, "xtick.labelsize": 6.8,
                     "ytick.labelsize": 6.8, "legend.fontsize": 6.8})
TITLE = dict(loc="left", fontsize=7.8, weight="bold")


def fig_main():
    m = pd.read_csv(RUN / "metrics_by_split.csv"); s = pd.read_csv(RUN / "secondary_metrics_by_split.csv")
    ctrl = s[(s.study == "control") & (s.clf == "svm")]
    fig, ax = plt.subplots(figsize=(W, 2.9), constrained_layout=True)
    order = ["A", "C", "B", "ALL8"]; x, ticks = 0, []; rng = np.random.default_rng(3); bounds = []
    for rep in P.REPS + P.CTRL:
        series = ({a: m[(m.rep == rep) & (m.clf == "svm") & (m.analysis == a)].ba.to_numpy() for a in order} if rep in P.REPS
                  else {"A": ctrl[ctrl.rep == rep].ba.to_numpy()})
        start = x
        for a, v in series.items():
            ax.scatter(x + rng.uniform(-0.17, 0.17, len(v)), v, s=5, color=P.ACOL[a], alpha=0.4, lw=0)
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="o", ms=4, color=P.ACOL[a], mec="white", mew=0.5, capsize=1.6, lw=1.0)
            x += 1
        ticks.append(((start + x - 1) / 2, P.NAME[rep])); x += 0.7 if rep not in ("stroke_cnn", "global_intensity") else 1.3
        if rep in ("combined", "stroke_cnn"):
            ax.axvline(x - 0.85, color="#d1d5db", lw=0.7); bounds.append(x - 0.85)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks], fontsize=6.6)
    ax.set_ylabel("writer balanced accuracy"); ax.set_ylim(0.25, 1.0); ax.set_xlim(-0.8, x - 0.5)
    for xx, t in ((-0.5, "handwriting features"), (bounds[0] + 0.3, "deep features"), (bounds[1] + 0.3, "controls")):
        ax.text(xx, 0.975, t, fontsize=6.8, color="#6b7280", va="top")
    for a in order:
        ax.scatter([], [], color=P.ACOL[a], s=16, label=P.ALAB[a])
    fig.legend(ncol=4, loc="outside upper center", frameon=False)
    P.save(fig, OUT, "fig2_main_results")


def fig_mechanism():
    perm = pd.read_csv(RUN / "permutation_tests.csv").set_index("rep")
    s = pd.read_csv(RUN / "secondary_metrics_summary.csv"); ma = pd.read_csv(RUN / "mirror_augmented_stroke.csv")
    mt = pd.read_csv(RUN / "mirror_test_by_split.csv") if (RUN / "mirror_test_by_split.csv").is_file() else None
    fig = plt.figure(figsize=(W, 4.3), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[1.0, 1.05])
    ax = sf[0].subplots()
    reps = [r for r in P.REPS + P.CTRL if r in perm.index]
    for i, rep in enumerate(reps):
        nm = pd.read_csv(RUN / f"permutation_null_{rep}.csv").null_mean_ba
        vp = ax.violinplot(nm, positions=[i], widths=0.75, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor("#d1d5db"); b.set_alpha(0.95); b.set_edgecolor("none")
        ax.scatter(i, perm.loc[rep, "observed_mean_ba"], color="#dc2626", s=22, zorder=3)
        ax.text(i, 1.02, f"p = {perm.loc[rep, 'p_mean']:.3f}", ha="center", va="bottom", fontsize=6.4)
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set_xticks(range(len(reps))); ax.set_xticklabels([P.NAME[r] for r in reps], fontsize=6.6)
    ax.set_ylim(0.25, 1.1); ax.set_ylabel("repeated-CV mean BA")
    ax.set_title("(a) Permutation null (grey) and observed value (red)", **TITLE)
    axb, axc = sf[1].subplots(1, 2, gridspec_kw=dict(width_ratios=[1, 1]))
    blocks = [("stroke_hinge", "hinge"), ("stroke_direction", "direction 360°"), ("stroke_orient_slant", "orientation / slant"),
              ("stroke_endings", "stroke ends"), ("stroke_rates", "end / junction rates"), ("stroke_turning", "turning (mirror-invariant)")]
    vals = [s[(s.study == "ablation") & (s.rep == b) & (s.analysis == "A")].iloc[0] for b, _ in blocks]
    y = np.arange(len(blocks))
    axb.barh(y, [v.ba_mean for v in vals], xerr=[v.ba_sd for v in vals], color="#c2410c", height=0.62, error_kw=dict(lw=0.7, capsize=1.6))
    axb.axvline(0.5, color="k", lw=0.6, ls=":"); axb.set_yticks(y); axb.set_yticklabels([l for _, l in blocks])
    axb.invert_yaxis(); axb.set_xlim(0.3, 0.85); axb.set_xlabel("writer BA (linear SVM, condition A)")
    axb.set_title("(b) Each STROKE block alone", **TITLE)
    bars = [("mirror-trained,\noriginal", ma.ba_original), ("mirror-trained,\nmirrored", ma.ba_mirrored)]
    if mt is not None:
        bars = [("STROKE SVM,\noriginal", mt.ba_original), ("STROKE SVM,\nmirrored", mt.ba_mirrored)] + bars
    cols = ["#c2410c", "#fdba74"] * 4
    for i, (lab, v) in enumerate(bars):
        axc.bar(i, v.mean(), yerr=v.std(ddof=1), color=cols[i], width=0.62, error_kw=dict(lw=0.7, capsize=1.6))
        axc.text(i, v.mean() + v.std(ddof=1) + 0.01, f"{100 * v.mean():.1f}", ha="center", va="bottom", fontsize=6.4)
    axc.axhline(0.5, color="k", lw=0.6, ls=":"); axc.set_xticks(range(len(bars))); axc.set_xticklabels([b[0] for b in bars], fontsize=6.4)
    axc.set_ylim(0.3, 0.85); axc.set_ylabel("writer BA (condition A)")
    axc.set_title("(c) Mirror test", **TITLE)
    P.save(fig, OUT, "fig4_mechanism")


def fig_differences():
    d = pd.read_csv(RUN / "stroke_bin_differences.csv")
    fig = plt.figure(figsize=(W, 4.6), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[1.2, 1.0])
    ax, axh = sf[0].subplots(1, 2, gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    sl = d[d.block == "slant"].reset_index(drop=True); cen = np.arange(-40, 41, 10); w = 4
    ax.bar(cen - w / 2, sl.left_mean, width=w, color=P.LEFT_C, label="left-handed")
    ax.bar(cen + w / 2, sl.right_mean, width=w, color=P.RIGHT_C, label="right-handed")
    for c, q, top in zip(cen, sl.q_bh_block, np.maximum(sl.left_mean, sl.right_mean)):
        if q < 0.10:
            ax.text(c, top + 0.004, "*", ha="center", fontsize=9)
    ax.set_xticks(cen); ax.set_xlabel("tilt from vertical (deg): $-$ forward '/', + backward '\\'")
    ax.set_ylabel("share of near-vertical stroke pixels"); ax.legend(frameon=False)
    ax.set_title("(a) Slant of near-vertical strokes", **TITLE)
    hg = d[d.block == "hinge"]; G = hg.hedges_g.to_numpy().reshape(12, 12); Q = hg.q_bh_block.to_numpy().reshape(12, 12)
    cmap = LinearSegmentedColormap.from_list("rb", [P.RIGHT_C, "#f0efec", P.LEFT_C])
    im = axh.imshow(G, cmap=cmap, vmin=-1.2, vmax=1.2, origin="lower")
    for i, j in zip(*np.where(Q < 0.10)):
        axh.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=0.8))
    ticks = [0, 3, 6, 9]; axh.set_xticks(ticks); axh.set_yticks(ticks)
    axh.set_xticklabels([f"{30 * t}°" for t in ticks]); axh.set_yticklabels([f"{30 * t}°" for t in ticks])
    axh.set_xlabel("incoming leg direction $\\phi_2$"); axh.set_ylabel("outgoing leg direction $\\phi_1$")
    cb = sf[0].colorbar(im, ax=axh, shrink=0.9, pad=0.02); cb.set_label("Hedges g (left $-$ right)")
    axh.set_title("(b) Contour hinge" + (" (boxes: q < 0.10)" if (Q < 0.10).any() else " (no cell q < 0.10)"), **TITLE)
    axc = sf[1].subplots()
    order = [("direction", "direction 360°"), ("hinge", "hinge"), ("endings", "stroke ends"), ("orientation", "orientation"),
             ("slant", "slant"), ("turning", "turning"), ("rates", "rates")]
    rng = np.random.default_rng(4)
    for i, (b, lab) in enumerate(order):
        sub = d[d.block == b]
        axc.scatter(sub.hedges_g, i + rng.uniform(-0.2, 0.2, len(sub)), s=9, c=np.where(sub.q_bh_block < 0.10, "#dc2626", "#9ca3af"), lw=0)
    axc.axvline(0, color="k", lw=0.5); axc.set_yticks(range(len(order))); axc.set_yticklabels([l for _, l in order])
    axc.invert_yaxis(); axc.set_xlabel("Hedges g per bin (left $-$ right)")
    axc.set_title("(c) Every STROKE bin (red: q < 0.10 within its block)", **TITLE)
    P.save(fig, OUT, "fig6_stroke_differences")


if __name__ == "__main__":
    fig_main(); fig_mechanism(); fig_differences(); print("written to", OUT)
