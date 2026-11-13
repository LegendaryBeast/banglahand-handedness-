#!/usr/bin/env python3
"""IJDAR figure of the CSAFE external application (fig9_csafe.pdf), drawn from the saved results.

(a) the same London Letter written by a left- and a right-handed writer (session 1, first copy);
(b) writer BA of every representation in condition A (bootstrap interval) and condition X, with the controls;
(c) the post hoc line-orientation check; (d) left-right differences of the layout measures (Hedges g, BH q);
(e) leakage: page-level BA with writer-level and image-level splits.
Inputs: results/csafe_ijdar_2026-11-12/{csafe_table, csafe_layout_tests, csafe_orientation_check}.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CS, EXT, OUT = ROOT / "results/csafe_ijdar_2026-11-12", ROOT / "external/csafe_v2", ROOT / "manuscripts/ijdar/figures"
LEFT_C, RIGHT_C, RED, GREEN, INK, GREY = "#c2410c", "#1d4ed8", "#dc2626", "#15803d", "#111827", "#6b7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "axes.titlesize": 7.6, "axes.labelsize": 7,
                     "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.6, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})
TITLE = dict(loc="left", fontsize=7.6, weight="bold")
NAME = {"hc183_tb": "HC-183, text block", "hc183_page": "HC-183, whole page", "stroke": "STROKE", "resnet18": "ResNet-18",
        "efficientnet_b0": "EffNet-B0", "vit_b_16": "ViT-B/16", "edge_direction": "Edge-direction", "edge_hinge": "Edge-hinge",
        "chain_code": "Chain code", "hog_glbp": "HOG + GLBP", "lbp": "Uniform LBP", "global_intensity": "Scan intensity",
        "background": "Paper only", "layout": "Layout measures"}
GROUPS = [("ours", ["hc183_tb", "hc183_page", "stroke"]), ("networks", ["resnet18", "efficientnet_b0", "vit_b_16"]),
          ("earlier methods", ["edge_direction", "edge_hinge", "chain_code", "hog_glbp", "lbp"]),
          ("controls", ["global_intensity", "background", "layout"])]


def page(ax, wid, title, colour):
    im = np.asarray(Image.open(EXT / f"selected/Session 1/w{wid}_s01_pLND_r01.png").convert("L"), np.float32)
    rows = np.flatnonzero((im < 160).mean(1) > 0.004); r0, r1 = max(0, rows.min() - 25), min(im.shape[0], rows.max() + 25)
    ax.imshow(im[r0:r1], cmap="gray", vmin=60, vmax=255, aspect="auto", interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(colour); s.set_linewidth(1.2)
    ax.set_title(title, loc="left", fontsize=7.2, color=colour, weight="bold")


def main():
    T = pd.read_csv(CS / "csafe_table.csv").set_index("rep")
    L = pd.read_csv(CS / "csafe_layout_tests.csv").set_index("measure")
    O = pd.read_csv(CS / "csafe_orientation_check.csv")
    meta = pd.read_csv(CS / "features/csafe_meta.csv", dtype={"wid": str})
    wl = meta.groupby("wid", sort=True).label.first()
    w_left, w_right = wl[wl == 1].index[0], wl[wl == 0].index[0]

    fig = plt.figure(figsize=(7.16, 6.0))
    ga = fig.add_gridspec(2, 1, left=0.01, right=0.315, top=0.965, bottom=0.535, hspace=0.30)
    gb = fig.add_gridspec(1, 1, left=0.505, right=0.985, top=0.965, bottom=0.535)
    page(fig.add_subplot(ga[0, 0]), w_left, "(a) Left-handed writer, London Letter", LEFT_C)
    page(fig.add_subplot(ga[1, 0]), w_right, "Right-handed writer, same prompt", RIGHT_C)

    # (b) all representations, condition A (interval) and X
    ax = fig.add_subplot(gb[0, 0]); y, ticks, labels = 0, [], []
    for g, reps in GROUPS:
        for r in reps:
            x = T.loc[r]; sig = x.A_p < 0.05
            c = GREY if g == "controls" else (INK if sig else "#9ca3af")
            ax.plot([100 * x.A_lo, 100 * x.A_hi], [y, y], color=c, lw=1.1, alpha=0.6)
            ax.scatter(100 * x.A, y, s=24, color=c, zorder=3, label="A: session 1" if y == 0 else None)
            ax.scatter(100 * x.X, y, s=24, facecolor="white", edgecolor=GREEN, lw=1.1, zorder=4, label="X: weeks later" if y == 0 else None)
            ticks.append(y); labels.append(NAME[r]); y += 1
        ax.text(41, y - len(reps) - 0.45, g, fontsize=6.4, color=GREY, style="italic", va="bottom")
        y += 0.8
    ax.axvline(50, color="k", lw=0.6, ls=":")
    ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_xlim(40, 100); ax.set_xlabel("writer balanced accuracy (%), linear SVM")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3)
    ax.set_title("(b) Every representation on unseen writers (dark: p < 0.05)", loc="right", fontsize=7.6, weight="bold")
    for t, r in zip(ax.get_yticklabels(), [r for _, rs in GROUPS for r in rs]):
        if r == "hc183_tb":
            t.set_weight("bold")

    bot = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.2, 1.0], left=0.065, right=0.985, top=0.415, bottom=0.075, wspace=0.55)
    # (c) orientation check
    ax = fig.add_subplot(bot[0, 0])
    covs = [("none", "none", "#111827"), ("orientation", "orientation", "#6b7280"), ("all layout+ink+scan", "all layout\n+ ink + scan", "#d1d5db")]
    reps = [("hc183_tb", "HC-183"), ("stroke", "STROKE"), ("hog_glbp", "HOG+GLBP")]
    w = 0.26
    for k, (c, lab, col) in enumerate(covs):
        v = [100 * O[(O.rep == r) & (O.covariates == c)].A.iloc[0] for r, _ in reps]
        ax.bar(np.arange(3) + (k - 1) * w, v, width=w, color=col, label=lab.replace("\n", " "), edgecolor="white", lw=0.4)
    ax.axhline(50, color="k", lw=0.6, ls=":"); ax.set_ylim(45, 86)
    ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in reps]); ax.set_ylabel("writer BA (%), condition A")
    ax.legend(frameon=False, fontsize=5.9, loc="upper left", title="regressed out", title_fontsize=5.9, ncol=1)
    ax.set_title("(c) Line orientation removed", **TITLE)

    # (d) layout measures
    ax = fig.add_subplot(bot[0, 1])
    show = [("ruled_lines", "long straight lines"), ("block_cx", "block centre, x"), ("block_cy", "block centre, y"),
            ("block_w", "block width"), ("block_h", "block height"), ("ink_textblock", "ink density"),
            ("n_lines", "number of lines"), ("spacing_rel", "line spacing"), ("line_slope_mean", "line slope"), ("skew_deg", "text skew")]
    gv = [L.loc[k, "hedges_g"] for k, _ in show]; qv = [L.loc[k, "q_bh"] for k, _ in show]
    ax.barh(range(len(show)), gv, color=[RED if q < 0.05 else "#9ca3af" for q in qv], height=0.65)
    for i, (g_, q_) in enumerate(zip(gv, qv)):
        ax.text(g_ + (0.03 if g_ >= 0 else -0.03), i, f"q = {q_:.2f}", va="center", ha="left" if g_ >= 0 else "right", fontsize=5.6,
                color=RED if q_ < 0.05 else GREY)
    ax.axvline(0, color="k", lw=0.6); ax.set_yticks(range(len(show))); ax.set_yticklabels([n for _, n in show])
    ax.invert_yaxis(); ax.tick_params(axis="y", length=0); ax.set_xlim(-1.15, 1.0)
    ax.set_xlabel("Hedges g (left $-$ right)")
    ax.set_title("(d) Page and layout: what differs", **TITLE)

    # (e) leakage
    ax = fig.add_subplot(bot[0, 2])
    lk = [r for r in ["vit_b_16", "efficientnet_b0", "stroke", "edge_hinge", "resnet18", "hc183_tb", "hog_glbp"]]
    lk.sort(key=lambda r: T.loc[r, "para_ba_leaky"] - T.loc[r, "para_ba_writer"])
    for i, r in enumerate(lk):
        a, b = 100 * T.loc[r, "para_ba_writer"], 100 * T.loc[r, "para_ba_leaky"]
        ax.plot([a, b], [i, i], color="#d1d5db", lw=2.4, solid_capstyle="round", zorder=1)
        ax.scatter(a, i, s=16, color=INK, zorder=2, lw=0); ax.scatter(b, i, s=16, color=RED, zorder=2, lw=0)
        ax.text(b + 1.5, i, f"+{b - a:.0f}", va="center", fontsize=6.0, color=RED, weight="bold")
    ax.set_yticks(range(len(lk))); ax.set_yticklabels([NAME[r].replace(", text block", "") for r in lk]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(50, 102); ax.set_xlabel("page-level BA (%)")
    ax.text(0.0, 1.0, "● writers split", transform=ax.transAxes, fontsize=6.0, color=INK, va="bottom")
    ax.text(0.55, 1.0, "● pages split", transform=ax.transAxes, fontsize=6.0, color=RED, va="bottom")
    ax.set_title("(e) Leakage", **TITLE, pad=15)

    fig.savefig(OUT / "fig9_csafe.pdf"); fig.savefig(OUT / "fig9_csafe.png", dpi=200); print("wrote", OUT / "fig9_csafe.pdf")


if __name__ == "__main__":
    main()
