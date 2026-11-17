#!/usr/bin/env python3
"""IJDAR Fig. 1: graphical overview of the study, drawn from the data and results themselves.

(a) the two cohorts (real pages); (b) removing the page: photograph -> printed lines found -> line-free text block
(run_layout_check.ruled_lines, run_lineless_embeddings.lineless_gray); (c) what is measured on that block: gradient
orientation (HC-183), contour direction (STROKE), headstroke cross-sections by hand (MATRA-INK writer means), CNN tiles;
(d) writer-independent folds (the real StratifiedKFold of the first seeds); (e) the whole-writer permutation null of
line-free HC-183; (f) leakage: paragraph BA with writer-level vs image-level splits; (g) the audit: accuracy as each
shortcut is removed, and the external cohort. Numbers come from the result files and the manuscript macros.
"""
import re
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import matra_features as mf  # noqa: E402
import run_final_experiments as rf  # noqa: E402
import run_layout_check as lc  # noqa: E402
import run_lineless_embeddings as le  # noqa: E402

DEV, BASE, FIG = ROOT / "results/ijdar_dev_2026-11-12", ROOT / "results/final42_2026-10-19", ROOT / "manuscripts/ijdar/figures"
LEFT_C, RIGHT_C, RED, GREEN, INK, GREY = "#c2410c", "#1d4ed8", "#dc2626", "#15803d", "#111827", "#6b7280"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.1, "axes.titlesize": 7.8, "axes.labelsize": 6.9,
                     "xtick.labelsize": 6.4, "ytick.labelsize": 6.4, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42})


def macros():
    m = {}
    for f in ("results_macros.tex", "layout_macros.tex", "csafe_macros.tex"):
        for k, v in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (FIG / f).read_text()):
            m[k] = v
    return m


def stretch(a):
    lo, hi = np.percentile(a, [1, 99.5]); return np.clip((a - lo) / (hi - lo), 0, 1)


def show(fig, rect, im, cmap="gray", edge="#d1d5db"):
    ax = fig.add_axes(rect); ax.imshow(im, cmap=cmap, interpolation="lanczos", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(edge); s.set_linewidth(0.8)
    return ax


def label(fig, x, y, letter, title):
    fig.text(x, y, letter, weight="bold", fontsize=9.8, va="bottom", ha="left", color=INK)
    fig.text(x + 0.018, y + 0.002, title, weight="bold", fontsize=7.9, va="bottom", ha="left", color=INK)


def arrow(fig, p0, p1, colour=GREY, lw=1.1, scale=8):
    fig.patches.append(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=scale,
                                       color=colour, lw=lw))


def polar(fig, rect, ang, weights, bins, colour, title):
    ax = fig.add_axes(rect, projection="polar")
    h, e = np.histogram(ang, bins=bins, range=(0, 2 * np.pi), weights=weights)
    ax.bar(e[:-1], h / h.max(), width=2 * np.pi / bins, align="edge", color=colour, alpha=0.85, lw=0)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels([]); ax.set_yticks([]); ax.spines["polar"].set_linewidth(0.4); ax.grid(lw=0.3)
    nm, sub = title.split("\n", 1)
    ax.set_title(nm, fontsize=7.2, pad=2.5, weight="bold")
    ax.text(0.5, -0.10, sub, transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color=GREY, linespacing=1.1)
    return ax


def main():
    m = macros()
    img = pd.read_csv(DEV / "layout_images.csv"); M = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    meta = M.set_index("image_id")
    pick = img[(img.label == 0) & (img.has_ruled == 1) & (img.ruled_h >= 2) & (img.ruled_v >= 1)].iloc[0]
    rel = meta.loc[pick.image_id, "relative_path"]
    a = lc.load(rel); mask, _ = lc.ruled_lines(a); inkl, _ = lc.ink_of(a); text = inkl & ~mask
    r0, r1, c0, c1 = lc.text_box(text)
    lf = np.asarray(le.lineless_gray(rel).convert("L").resize((a.shape[1], a.shape[0]), Image.Resampling.LANCZOS), np.float32)
    block = lf[r0:r1, c0:c1]

    fig = plt.figure(figsize=(7.16, 4.15))
    Y1, H1 = 0.535, 0.405                                   # upper row: images and features
    Y2, H2 = 0.030, 0.385                                   # lower row: protocol and audit

    # (a) cohorts
    label(fig, 0.005, Y1 + H1 + 0.012, "a", "Two cohorts")
    show(fig, [0.010, Y1 + 0.075, 0.098, H1 - 0.075], stretch(a))
    cs = np.asarray(Image.open(ROOT / "external/csafe_v2/selected/Session 1/w0016_s01_pLND_r01.png").convert("L"), np.float32)
    show(fig, [0.115, Y1 + 0.075, 0.098, H1 - 0.075], stretch(cs[: int(0.36 * cs.shape[0])]))
    for x, t1, t2 in ((0.059, "BanglaHand", f"{m['vnWriters']} writers\nphotographs"),
                      (0.164, "CSAFE", f"{m['vCsN']} writers\nEnglish scans")):
        fig.text(x, Y1 + 0.060, t1, ha="center", va="top", weight="bold", fontsize=7.1)
        fig.text(x, Y1 + 0.036, t2, ha="center", va="top", fontsize=6.3, color=GREY, linespacing=1.15)

    # (b) removing the page
    label(fig, 0.235, Y1 + H1 + 0.012, "b", "Removing the page")
    rgb = np.repeat(stretch(a)[..., None], 3, 2); rgb[mask] = [0.86, 0.15, 0.15]
    xs = [0.240, 0.345, 0.450]; w = 0.090
    show(fig, [xs[0], Y1 + 0.075, w, H1 - 0.075], stretch(a))
    show(fig, [xs[1], Y1 + 0.075, w, H1 - 0.075], rgb, edge=RED)
    show(fig, [xs[2], Y1 + 0.075, w, H1 - 0.075], stretch(block), edge=GREEN)
    for x0, x1 in ((xs[0] + w, xs[1]), (xs[1] + w, xs[2])):
        arrow(fig, (x0 + 0.002, Y1 + 0.22), (x1 - 0.002, Y1 + 0.22))
    for x, t, c in zip(xs, ["photograph", "printed lines\nfound (red)", "line-free\ntext block"], [GREY, RED, GREEN]):
        fig.text(x + w / 2, Y1 + 0.060, t, ha="center", va="top", fontsize=6.3, color=c, linespacing=1.15,
                 weight="normal" if c == GREY else "bold")

    # (d) writer-independent folds (real folds, first three seeds)
    label(fig, 0.005, Y2 + H2 + 0.012, "d", "Writer-independent protocol: five-fold writer cross-validation, 20 seeds, whole-writer permutation test")
    D = rf.Data(BASE); order = np.argsort(-D.wlabels, kind="stable"); L = D.wlabels[order]; n = len(L); nl = int(L.sum())
    ax = fig.add_axes([0.035, Y2 + 0.075, 0.42, H2 - 0.135])
    for s_i, seed in enumerate(rf.SEEDS[:3]):
        f = rf.writer_folds(D.writers, D.wlabels, seed)[order]
        for k in range(5):
            yy = s_i * 6.4 + k
            for i in range(n):
                ax.add_patch(Rectangle((i, yy), 0.86, 0.86, fc=INK if f[i] == k else ("#fed7aa" if L[i] else "#bfdbfe"), ec="none"))
    ax.set_xlim(-0.3, n); ax.set_ylim(3 * 6.4 - 1.3, -2.2); ax.axis("off")
    for s_i in range(3):
        ax.text(-0.8, s_i * 6.4 + 2.4, f"seed {s_i + 1}", ha="right", va="center", fontsize=5.8, color=GREY, rotation=90)
    ax.text(nl / 2, -0.6, f"{nl} left", ha="center", va="bottom", fontsize=6.2, color=LEFT_C, weight="bold")
    ax.text(nl + (n - nl) / 2, -0.6, f"{n - nl} right", ha="center", va="bottom", fontsize=6.2, color=RIGHT_C, weight="bold")
    fig.text(0.245, Y2 + 0.012, "rows: folds of the first three seeds; dark: held-out test writers",
             ha="center", va="bottom", fontsize=6.1, color=GREY, linespacing=1.15)

    # (e) permutation null of line-free HC-183
    nul = pd.read_csv(DEV / "permutation_null_hc183_lf.csv").null_mean_ba.to_numpy() * 100   # scripts/save_linefree_null.py
    obs = float(m["vHcTbA"])
    ax = fig.add_axes([0.56, Y2 + 0.095, 0.40, H2 - 0.16])
    ax.hist(nul, bins=np.arange(24, 88, 2), color="#cbd5e1", lw=0); ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.axvline(obs, color=RED, lw=1.4); ax.axvline(50, color=GREY, lw=0.6, ls=":")
    ax.text(obs + 1.2, ax.get_ylim()[1] * 0.97, f"{m['vHcTbA']}%\np = {m['vHcTbP']}", color=RED, fontsize=6.2, va="top",
            ha="left", weight="bold")
    ax.set_xlabel("writer BA (%), line-free HC-183", labelpad=1); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.text(0.01, 0.97, "null of line-free HC-183:\n999 relabellings, folds redrawn", transform=ax.transAxes, va="top", ha="left", fontsize=5.8, color=GREY)

    # (g) the audit: accuracy as each shortcut is removed, and the external cohort
    label(fig, 0.585, Y1 + H1 + 0.012, "c", "Shortcut audit, HC-183")
    steps = [("Whole photograph", float(m["vHcA"]), INK),
             ("Printed lines removed", float(m["vHcNoLineA"]), "#374151"),
             ("Line-free text block", float(m["vHcTbA"]), "#4b5563"),
             ("+ ink, photo adjusted", float(m["vHcTbAdj"]), "#9ca3af"),
             ("Framing only, no writing", float(m["vFrameA"]), RED),
             ("CSAFE, unseen writers", float(m["vCsHcA"]), GREEN),
             ("CSAFE, weeks later", float(m["vCsHcX"]), GREEN)]
    ax = fig.add_axes([0.80, Y1 + 0.075, 0.165, H1 - 0.105])
    yv = [0, 1, 2, 3, 4.5, 6.0, 7.0]
    for (nm, v, c), y in zip(steps, yv):
        ax.barh(y, v - 50, left=50, height=0.72, color=c, lw=0)
        ax.text(v + 0.8, y, f"{v:.1f}", va="center", fontsize=6.1, color=c, weight="bold")
    ax.set_yticks(yv); ax.set_yticklabels([s[0] for s in steps]); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    for t, (_, _, c) in zip(ax.get_yticklabels(), steps):
        t.set_color(c if c in (RED, GREEN) else INK)
    ax.set_xlim(50, 86); ax.set_xticks([50, 60, 70, 80]); ax.set_xlabel("writer BA (%)", labelpad=1)
    ax.text(85.5, 5.3, f"Holm p = {m['vCsHcHolm']}", ha="right", va="center", fontsize=5.8, color=GREEN, style="italic")

    out = FIG / "fig0_overview.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=220)
    print("wrote", out)


if __name__ == "__main__":
    main()
