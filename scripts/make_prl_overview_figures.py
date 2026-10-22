#!/usr/bin/env python3
"""Dataset and methodology figures for the PRL letter (fig1_dataset, fig2_method)."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image
from skimage.measure import find_contours
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stroke_features as sf  # noqa: E402

LEFT_C, RIGHT_C, INK, MUTED, GRID = "#c2410c", "#1d4ed8", "#111827", "#6b7280", "#e5e7eb"
SAMPLES_L, SAMPLES_R = ["L_p3", "L_p11", "L_p15"], ["R_p11", "R_p16", "R_p2"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "pdf.fonttype": 42, "axes.linewidth": 0.5})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=260); plt.close(fig)


def for_display(g):
    """Display-only rendering of a grey photograph (never used by any model): divide by a smooth estimate of the
    paper (flattens uneven lighting) and stretch so paper is white and ink black."""
    import cv2
    a = np.asarray(g, np.float32)
    bg = cv2.GaussianBlur(cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8)), (0, 0), 25)
    f = np.clip(a / np.maximum(bg, 1) * 255, 0, 255)
    lo, hi = np.percentile(f, 0.5), np.percentile(f, 60)
    return Image.fromarray(np.clip((f - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8))


def text_block(path, ratio, width=1400):
    """Top of the text block of a paragraph photo, cut from the full-resolution image with the given width/height
    ratio (the ratio of the axes it is shown in, so nothing is stretched), display-corrected, resized to `width`."""
    g = Image.open(path).convert("L")
    W, H = g.size; s = 1536 / W
    a = np.asarray(g.resize((1536, round(H * s)), Image.Resampling.LANCZOS)).astype(float)
    full = ndi.gaussian_filter(a, 34) - a > 17                      # ink, only to locate the text block
    rows = np.flatnonzero(full.mean(1) > 0.01); cols = np.flatnonzero(full.mean(0) > 0.005)
    r0 = int(np.percentile(rows, 0.5)); c0, c1 = np.percentile(cols, [0.5, 99.5]).astype(int)
    x0, x1, y0 = max(0, c0 - 12) / s, min(1536, c1 + 12) / s, max(0, r0 - 12) / s
    crop = for_display(g.crop((int(x0), int(y0), int(x1), int(y0 + (x1 - x0) / ratio))))
    return np.asarray(crop.resize((width, round(width / ratio)), Image.Resampling.LANCZOS))


def fig_dataset(out):
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    fig = plt.figure(figsize=(7.16, 3.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.72], width_ratios=[1, 1, 1, 1.05], hspace=0.42, wspace=0.28,
                          left=0.045, right=0.99, top=0.93, bottom=0.11)
    # (a) the same prompt S01 by three left- and three right-handed writers
    sub = gs[0, :3].subgridspec(2, 3, hspace=0.12, wspace=0.05)
    for row, (ws, col, lab) in enumerate([(SAMPLES_L, LEFT_C, "left-handed"), (SAMPLES_R, RIGHT_C, "right-handed")]):
        for j, w in enumerate(ws):
            ax = fig.add_subplot(sub[row, j])
            r = meta[(meta.writer_id == w) & (meta.paragraph_number == 1)].iloc[0]
            bb = ax.get_position(); ratio = (bb.width * fig.get_figwidth()) / (bb.height * fig.get_figheight())
            ax.imshow(text_block(ROOT / "dataset" / r.relative_path, ratio), cmap="gray", vmin=0, vmax=255,
                      interpolation="none")                     # embedded at full resolution in the PDF
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(col); s.set_linewidth(1.1)
            ax.text(0.015, 0.04, w.replace("_p", " P"), transform=ax.transAxes, fontsize=5.4, color="white",
                    bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none"))
            if j == 0:
                ax.set_ylabel(lab, color=col, fontsize=6.5, labelpad=2)
    fig.text(0.03, 0.955, "(a) The same shared prompt (S01) written by three left- and three right-handed writers", fontsize=7, fontweight="bold")
    # (b) per-writer structure
    ax = fig.add_subplot(gs[0, 3]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0, 9.9, "(b) Per writer", fontsize=7, fontweight="bold", va="top")
    for sheet in range(4):   # one plain page outline per sheet, two paragraphs per page
        x = 0.2 + sheet * 2.45
        ax.add_patch(Rectangle((x, 2.45), 2.15, 4.75, fc="white", ec="#6b7280", lw=0.6))
        ax.plot([x + 0.15, x + 2.0], [4.82, 4.82], color="#d1d5db", lw=0.5)
        for pos in range(2):
            k = 2 * sheet + pos; y = 4.95 - pos * 2.4; shared = k < 6
            ax.text(x + 1.07, y + 1.25, f"{k + 1}", ha="center", va="center", fontsize=8, fontweight="bold", color=INK, zorder=2)
            ax.text(x + 1.07, y + 0.45, f"S0{k + 1}" if shared else "own", ha="center", va="center", fontsize=5.2,
                    color=INK if shared else "#92400e", fontstyle="normal" if shared else "italic", zorder=2)
    for sheet in range(4):
        ax.text(0.2 + sheet * 2.45 + 1.07, 7.6, f"sheet {sheet + 1}", ha="center", fontsize=5, color=MUTED)
    ax.text(0.2, 1.5, "S01-S06: Bangladesh, education,\ntechnology, environment, time,\nhealth; own: free topic. Pens and\npaper from one supply; one phone.",
            fontsize=4.9, color=MUTED, va="top", linespacing=1.3)
    # (c) composition
    ax = fig.add_subplot(gs[1, 0])
    n = meta.groupby(["hand", "condition"]).size().unstack()
    for i, (h, c) in enumerate([("left", LEFT_C), ("right", RIGHT_C)]):
        ax.barh(i, n.loc[h, "shared"], color=c, height=0.55)
        ax.barh(i, n.loc[h, "writer_specific"], left=n.loc[h, "shared"], color=c, alpha=0.45, height=0.55)
        nw = meta[meta.hand == h].writer_id.nunique()
        ax.text(4, i, f"{nw} writers, {n.loc[h].sum()} images", va="center", fontsize=5.4, color="white", fontweight="bold")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["left", "right"]); ax.set_xlim(0, 160); ax.invert_yaxis()
    ax.set_xlabel("images (dark: shared, light: own text)", fontsize=5.6)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=5.6)
    ax.set_title("(c) Composition", fontsize=7, fontweight="bold", loc="left")
    # (d) words per shared prompt / own text topics
    ax = fig.add_subplot(gs[1, 1])
    p = pd.read_csv(ROOT / "dataset/metadata/prompts.csv")
    ax.bar(p.prompt_id, p.n_words, color="#93c5fd", width=0.6)
    for x, v in zip(p.prompt_id, p.n_words):
        ax.text(x, v + 0.8, str(v), ha="center", fontsize=5)
    ax.set_ylim(0, 62); ax.set_ylabel("words", fontsize=5.6); ax.tick_params(labelsize=5.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(d) Shared prompt length", fontsize=7, fontweight="bold", loc="left")
    # (e) capture order
    ax = fig.add_subplot(gs[1, 2])
    t = meta.copy()
    t["h"] = pd.to_datetime(t.capture_datetime, format="%Y:%m:%d %H:%M:%S")
    first = t.groupby("writer_id").h.min().sort_values()
    hrs = (first.dt.hour + first.dt.minute / 60).to_numpy()
    cols = [LEFT_C if w.startswith("L") else RIGHT_C for w in first.index]
    ys = np.where([w.startswith("L") for w in first.index], 0.22, -0.22)   # one row per group: blocks of writers
    ax.scatter(hrs, ys, c=cols, s=9, lw=0)
    ax.set_ylim(-0.6, 0.6); ax.set_yticks([-0.22, 0.22]); ax.set_yticklabels(["right-\nhanded", "left-\nhanded"], fontsize=5)
    ax.set_xlim(np.floor(hrs.min()) - 0.1, np.ceil(hrs.max()) + 0.1)
    ax.set_xlabel(f"time of photography (h), {t.h.dt.strftime('%d %b').iloc[0]}", fontsize=5.6)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="x", labelsize=5.4); ax.tick_params(axis="y", length=0)
    ax.set_title("(e) Photography order", fontsize=7, fontweight="bold", loc="left")
    # (f) duplicate screen
    ax = fig.add_subplot(gs[1, 3])
    A = ROOT / "audit/geometric"
    g = pd.read_csv(A / "geometric_pairs.csv.gz")
    bins = np.logspace(0, np.log10(2500), 32)
    ax.hist(np.clip(g[g.kind == "same_prompt_different_writers"].inliers, 1, None), bins=bins, color="#9ca3af", label="different writers")
    raw = g[(g.kind == "unreleased_vs_released") & g.a.str.startswith("unreleased/P19") & g.b.str.startswith("released/R_p6_")]
    ax.hist(raw.inliers, bins=bins, color="#dc2626", label="unreleased re-photograph")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.tick_params(labelsize=5.4)
    ax.set_xlabel("SIFT-RANSAC inliers per image pair", fontsize=5.6); ax.set_ylabel("pairs", fontsize=5.6)
    ax.legend(frameon=False, fontsize=4.8, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(f) Duplicate audit", fontsize=7, fontweight="bold", loc="left")
    save(fig, out, "fig1_dataset")


def fig_method(out):
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    r = meta[(meta.writer_id == "L_p3") & (meta.paragraph_number == 1)].iloc[0]
    gray = Image.open(ROOT / "dataset" / r.relative_path).convert("L")
    ink, (br0, br1, bc0, bc1) = sf.ink_mask(gray, return_box=True)
    y0, x0, hh, ww = 205, 75, 170, 470                     # same window in every panel (ink coordinates, 1536-px image)
    win = ink[y0:y0 + hh, x0:x0 + ww]
    s = gray.width / sf.WIDTH                                # (a) is cut from the full-resolution photograph
    photo = np.asarray(for_display(gray.crop((round((bc0 + x0) * s), round((br0 + y0) * s), round((bc0 + x0 + ww) * s), round((br0 + y0 + hh) * s)))))
    sk = skeletonize(win); nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk
    ends = np.argwhere(sk & (nb == 1))
    v, _, sw = sf.stroke_features(gray)
    tiles = sf.binary_tiles(sf.ink_mask(gray), sw)

    fig = plt.figure(figsize=(7.16, 2.75))
    top = fig.add_gridspec(1, 5, left=0.01, right=0.995, top=0.88, bottom=0.46, wspace=0.22)
    titles = ["(a) photograph", "(b) binary ink, text block", "(c) oriented outlines, stroke ends",
              "(d) direction histogram", "(e) binary tiles"]
    ax = fig.add_subplot(top[0]); ax.imshow(photo, cmap="gray", vmin=0, vmax=255, aspect="auto", interpolation="none")
    ax = fig.add_subplot(top[1]); ax.imshow(~win, cmap="gray", aspect="auto", interpolation="none")
    ax = fig.add_subplot(top[2]); ax.imshow(np.ones_like(win, float), cmap="gray", vmin=0, vmax=1, aspect="auto")
    for c in find_contours(np.pad(win, 1).astype(float), 0.5, positive_orientation="high"):
        ax.plot(c[:, 1] - 1, c[:, 0] - 1, lw=0.45, color=RIGHT_C)
        if len(c) > 30:
            i = len(c) // 3; d = c[i + 4] - c[i]
            ax.annotate("", xy=(c[i + 4, 1] - 1, c[i + 4, 0] - 1), xytext=(c[i, 1] - 1, c[i, 0] - 1),
                        arrowprops=dict(arrowstyle="-|>", color=RIGHT_C, lw=0.5, mutation_scale=5))
    ax.scatter(ends[:, 1], ends[:, 0], s=6, c="#dc2626", lw=0, zorder=3)
    axd = fig.add_subplot(top[3], projection="polar"); pos = axd.get_position(); axd.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.9])
    th = np.deg2rad(np.arange(16) * 22.5)
    axd.plot(np.append(th, th[0]), np.append(v[:16], v[0]), color=LEFT_C, lw=1.0)
    axd.fill(np.append(th, th[0]), np.append(v[:16], v[0]), color=LEFT_C, alpha=0.18)
    axd.set_xticks(np.deg2rad([0, 90, 180, 270])); axd.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=4.6); axd.tick_params(pad=-2)
    axd.set_yticklabels([]); axd.grid(lw=0.3); axd.set_xticklabels(["", "90°", "", ""], fontsize=4.6)
    rows_ = np.array([(1 - np.asarray(x)).mean(1).max() for x in tiles])
    cols_ = np.array([ndi.maximum_filter1d(1 - np.asarray(x), 7, axis=1).mean(0).max() for x in tiles])   # tolerates slightly slanted rules
    ok = np.flatnonzero((rows_ < 0.8) & (cols_ < 0.6))                      # skip tiles crossed by a page edge or a ruled margin line
    pick = sorted(ok[np.argsort([-(1 - np.asarray(tiles[i])).mean() for i in ok])[:2]])
    ax = fig.add_subplot(top[4]); ax.imshow(np.hstack([tiles[pick[0]], np.ones((224, 8)), tiles[pick[1]]]), cmap="gray", aspect="auto", interpolation="none")
    for i, a_ in enumerate(fig.axes):
        if a_ is not axd:
            a_.set_xticks([]); a_.set_yticks([])
            for s in a_.spines.values():
                s.set_color("#d1d5db"); s.set_linewidth(0.6)
        if a_ is not axd:
            a_.set_title(titles[i], fontsize=6.4, pad=3)
    fig.text(0.70, 0.905, titles[3], ha="center", fontsize=6.4)
    # arrows between panels
    for x in [0.2005, 0.3995, 0.5985, 0.7975]:
        fig.text(x, 0.65, "→", fontsize=8, color="k", ha="center", va="center")
    # bottom left: writer-independent folds (reference split)
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_final_experiments as rf
    wt = meta.groupby("writer_id").label.first().sort_index()
    folds = rf.writer_folds(wt.index.to_numpy(), wt.to_numpy(), 2026)
    order = np.lexsort((wt.index.to_numpy(), -wt.to_numpy()))
    axf = fig.add_axes([0.055, 0.07, 0.43, 0.27])
    for xi, wi in enumerate(order):
        for f_ in range(5):
            test = folds[wi] == f_
            axf.add_patch(Rectangle((xi, 4 - f_), 0.92, 0.84, fc="#374151" if test else "#e5e7eb", ec="none"))
    nl = int(wt.sum())
    axf.plot([nl - 0.04, nl - 0.04], [-0.1, 5], color="k", lw=0.6)
    axf.set_xlim(0, len(order)); axf.set_ylim(-0.1, 5.0)
    axf.set_yticks(np.arange(5) + 0.42); axf.set_yticklabels([f"fold {5 - i}" for i in range(5)], fontsize=5.2)
    axf.set_xticks([nl / 2, nl + (len(order) - nl) / 2]); axf.set_xticklabels([f"{nl} left-handed writers", f"{len(order) - nl} right-handed writers"], fontsize=5.4)
    axf.tick_params(length=0); [s.set_visible(False) for s in axf.spines.values()]
    axf.set_title("(f) Writer folds, one of 20 seeds (dark: test writers, all images held out)", fontsize=6.4, loc="left", pad=3)
    # bottom right: paragraphs used for training and testing in each condition
    axc = fig.add_axes([0.565, 0.07, 0.42, 0.27])
    conds = [("A", "1-6", "1-6"), ("B", "7-8", "7-8"), ("C", "1-6", "7-8"), ("D$_k$", "1-6", "pair k"), ("E$_k$", "pair k", "pair k")]
    for r_, (name, tr, te) in enumerate(conds):
        y = len(conds) - 1 - r_
        for k in range(1, 9):
            def has(spec):
                return (spec == "1-6" and k <= 6) or (spec == "7-8" and k >= 7) or (spec == "pair k" and k in (3, 4))
            trn, tst = has(tr), has(te)
            axc.add_patch(Rectangle((k - 1, y), 0.92, 0.8, fc="white", ec="#9ca3af", lw=0.3))
            if trn:
                axc.add_patch(Rectangle((k - 1, y), 0.46, 0.8, fc="#9ca3af", ec="none"))
            if tst:
                axc.add_patch(Rectangle((k - 0.54, y), 0.46, 0.8, fc="#111827", ec="none"))
        axc.text(-0.3, y + 0.4, name, ha="right", va="center", fontsize=6)
    axc.set_xlim(-1.4, 11.6); axc.set_ylim(-0.6, len(conds) + 0.3); axc.axis("off")
    for k in range(1, 9):
        axc.text(k - 0.54, len(conds) + 0.05, str(k), ha="center", fontsize=5.2)
    axc.plot([5.96, 5.96], [-0.2, len(conds)], color="k", lw=0.6)
    axc.text(2.5, -0.5, "shared prompts", ha="center", fontsize=5.2); axc.text(7.0, -0.5, "own", ha="center", fontsize=5.2)
    lx = 8.6
    axc.add_patch(Rectangle((lx, 3.3), 0.3, 0.6, fc="#9ca3af", ec="none")); axc.text(lx + 0.5, 3.6, "used to train", va="center", fontsize=5.2)
    axc.add_patch(Rectangle((lx, 2.2), 0.3, 0.6, fc="#111827", ec="none")); axc.text(lx + 0.5, 2.5, "used to test", va="center", fontsize=5.2)
    axc.text(lx, 1.2, "C, D reuse\nmodel A", fontsize=5.0, color=MUTED, va="center")
    axc.set_title("(g) Paragraphs per text condition (pair k = 2 shown)", fontsize=6.4, loc="left", pad=3, x=0.0)
    save(fig, out, "fig2_method")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True)
    out = ROOT / ap.parse_args().output; out.mkdir(parents=True, exist_ok=True)
    fig_dataset(out); fig_method(out); print("written", out)


if __name__ == "__main__":
    main()
