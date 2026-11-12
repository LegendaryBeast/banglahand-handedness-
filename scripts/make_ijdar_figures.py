#!/usr/bin/env python3
"""IJDAR figures: MATRA-INK (detection and group profiles) and the printed-border shortcut.

Writes manuscripts/ijdar/figures/fig7_matra.pdf and fig8_border.pdf from results/ijdar_dev_2026-11-12.
"""
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import matra_features as mf  # noqa: E402
import run_layout_check as lc  # noqa: E402

DEV = ROOT / "results/ijdar_dev_2026-11-12"
OUT = ROOT / "manuscripts/ijdar/figures"
LEFT_C, RIGHT_C = "#c2410c", "#1d4ed8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
META = pd.read_csv(ROOT / "dataset/metadata/images.csv")


def writer_means(X, names):
    d = pd.DataFrame(X, columns=names).assign(writer=META.writer_id, label=META.label, para=META.paragraph_number)
    return d[d.para <= 6].groupby("writer").agg({**{n: "mean" for n in names}, "label": "first"})


def ci(v, rng, n=2000):
    b = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def headstroke_panel(ax, rel, title, colour):
    g = cv2.imread(str(ROOT / "dataset" / rel), cv2.IMREAD_GRAYSCALE)
    _, dbg = mf.features_from_gray(g, return_debug=True)
    H, W = g.shape
    y0, y1, x0, x1 = int(0.28 * H), int(0.58 * H), int(0.07 * W), int(0.60 * W)
    ax.imshow(g[y0:y1, x0:x1], cmap="gray", vmin=60, vmax=235)
    for a, b, yc, _ in dbg["matras"]:                          # yc: centreline row of every column a..b
        if y0 <= yc.min() and yc.max() < y1 and a >= x0 and b < x1:
            ax.plot(np.arange(a, b + 1) - x0, yc - y0, color=colour, lw=1.6, alpha=0.85, solid_capstyle="butt")
    ax.set_title(f"{title} ({dbg['n_used']} headstrokes on the page)", loc="left")
    ax.axis("off")


def fig_matra():
    z = np.load(DEV / "features/matra_features.npz", allow_pickle=True)
    names = list(z["names"]); wm = writer_means(z["matra"], names); rng = np.random.default_rng(7)
    T = pd.read_csv(DEV / "matra_feature_tests.csv").set_index("feature")
    fig = plt.figure(figsize=(7.1, 3.3))
    gs = fig.add_gridspec(2, 6, height_ratios=[0.72, 1], hspace=0.38, wspace=0.9)
    headstroke_panel(fig.add_subplot(gs[0, :3]), "Left-Handed/P1/paragraph_01.jpg", "(a) Left-handed writer L1", LEFT_C)
    headstroke_panel(fig.add_subplot(gs[0, 3:]), "Right-Handed/P2/paragraph_03.jpg", "(b) Right-handed writer R2", RIGHT_C)
    cols = [f"cross_{k:02d}" for k in range(len(mf.U))]
    ax = fig.add_subplot(gs[1, 0:2])
    for lab, c, nm in ((1, LEFT_C, "left"), (0, RIGHT_C, "right")):
        v = wm[wm.label == lab][cols].to_numpy(); m = v.mean(0)
        lo, hi = np.array([ci(v[:, j], rng) for j in range(v.shape[1])]).T
        ax.fill_between(mf.U, lo, hi, color=c, alpha=0.18, lw=0); ax.plot(mf.U, m, color=c, lw=1.2, label=f"{nm}-handed")
    ax.set_xlabel("offset (stroke widths, + below)"); ax.set_ylabel("darkness / centre darkness")
    ax.set_title("(c) Headstroke cross-section", loc="left"); ax.legend(frameon=False, loc="lower center")
    ax = fig.add_subplot(gs[1, 2:4])
    g = T.loc[cols, "hedges_g"].to_numpy(); q = T.loc[cols, "q_bh_block"].to_numpy()
    ax.bar(mf.U, g, width=0.2, color=np.where(q < 0.05, "#111827", "#9ca3af"))
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("offset (stroke widths, + below)")
    ax.set_ylabel("Hedges g, left − right"); ax.set_title("(d) Group difference per point", loc="left")
    ax.text(0.02, 0.04, "black: q < 0.05", transform=ax.transAxes, fontsize=6)
    ax = fig.add_subplot(gs[1, 4:6]); al = [f"along_dark_{k}" for k in range(mf.N_ALONG)]; x = np.arange(mf.N_ALONG) + 1
    for lab, c in ((1, LEFT_C), (0, RIGHT_C)):
        v = wm[wm.label == lab][al].to_numpy(); lo, hi = np.array([ci(v[:, j], rng) for j in range(v.shape[1])]).T
        ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0); ax.plot(x, v.mean(0), color=c, lw=1.2)
    ax.set_xlabel("bin, left end → right end"); ax.set_ylabel("darkness / stroke mean")
    ax.set_title("(e) Along the headstroke", loc="left")
    fig.savefig(OUT / "fig7_matra.pdf", bbox_inches="tight"); plt.close(fig)


def border_panel(ax, rel, title):
    a = lc.load(rel); lines, _ = lc.ruled_lines(a)
    rgb = np.repeat(np.clip(a, 0, 255).astype(np.uint8)[..., None], 3, 2)
    rgb[lines] = (220, 38, 38)
    ax.imshow(rgb); ax.set_title(title, loc="left"); ax.axis("off")


def fig_border():
    img = pd.read_csv(DEV / "layout_images.csv")
    S = pd.read_csv(DEV / "layout_cv_summary.csv").set_index("rep").ba_mean
    E = pd.read_csv(DEV / "layout_cv_adjusted.csv").set_index(["rep", "covariates"]).adjusted_ba
    fig = plt.figure(figsize=(7.1, 3.7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 0.8], hspace=0.35, wspace=0.4)
    border_panel(fig.add_subplot(gs[0, 0]), "Right-Handed/P9/paragraph_02.jpg", "(a) Right-handed R9: border visible")
    border_panel(fig.add_subplot(gs[0, 1]), "Left-Handed/P11/paragraph_02.jpg", "(b) Left-handed L11: no border")
    ax = fig.add_subplot(gs[0, 2])
    w = img[img.para <= 8].groupby("writer").agg(label=("label", "first"), frac=("has_ruled", "mean"))
    rng = np.random.default_rng(3)
    for lab, c, xpos in ((1, LEFT_C, 0), (0, RIGHT_C, 1)):
        v = w[w.label == lab].frac.to_numpy()
        ax.scatter(xpos + rng.uniform(-0.15, 0.15, len(v)), 100 * v, s=9, color=c, alpha=0.8, lw=0)
        ax.plot([xpos - 0.25, xpos + 0.25], [100 * v.mean()] * 2, color="k", lw=1)
    ax.set_xticks([0, 1], ["left", "right"]); ax.set_ylabel("pages with border (%)")
    ax.set_title("(c) Per writer", loc="left"); ax.set_xlim(-0.5, 1.5)
    ax = fig.add_subplot(gs[1, :])
    rows = [("Framing only (no handwriting)", S["framing"], "#dc2626"),
            ("Writers' own layout (skew, spacing, margins)", S["behaviour"], "#9ca3af"),
            ("HC-183, whole photograph", S["hc183"], "#111827"),
            ("HC-183, printed lines removed", S["hc183_noline"], "#374151"),
            ("HC-183, lines removed, text block", S["hc183_textblock"], "#4b5563"),
            ("  … + ink and photo intensity adjusted", E[("hc183_textblock", "ink+photo")], "#6b7280"),
            ("  … + framing adjusted as well", E[("hc183_textblock", "ink+photo+framing")], "#9ca3af")]
    f = DEV / "lineless_embeddings_summary.csv"
    if f.exists():
        L = pd.read_csv(f).set_index("rep")
        for k, nm in (("efficientnet_b0", "EffNet-B0"), ("vit_b_16", "ViT-B/16")):
            rows += [(f"{nm}, whole photograph", L.loc[k, "ba_mean"], "#60a5fa"),
                     (f"{nm}, lines removed, text block", L.loc[k + "_lineless", "ba_mean"], "#93c5fd")]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [100 * r[1] for r in rows], color=[r[2] for r in rows], height=0.7)
    for yi, r in zip(y, rows):
        ax.text(100 * r[1] + 0.5, yi, f"{100 * r[1]:.1f}", va="center", fontsize=6)
    ax.set_yticks(y, [r[0] for r in rows]); ax.axvline(50, color="k", ls=":", lw=0.8)
    ax.set_xlim(40, 85); ax.set_xlabel("writer balanced accuracy (%), linear SVM, condition A")
    ax.set_title("(d) What the printed border is worth", loc="left")
    fig.savefig(OUT / "fig8_border.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["matra", "border"]
    if "matra" in which:
        fig_matra()
    if "border" in which:
        fig_border()
    print("written:", which)
