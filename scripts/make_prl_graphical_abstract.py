#!/usr/bin/env python3
"""Graphical abstract of the PRL letter (required by the journal: at least 1328 px wide, readable at 13 cm).

Usage: make_prl_graphical_abstract.py --run results/final42_2026-10-19 --output manuscripts/prl/graphical_abstract [--variant 2]
Writes graphical_abstract[_v2].pdf, .tif and .png (13 cm wide, 600 dpi). All numbers are read from the run.
Variant 1: four panels (data, unseen-writer folds, accuracy, darkness cue).
Variant 2: one-line finding on top, samples, horizontal accuracy bars and the darkness cue, protocol line below.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_prl_overview_figures as ov  # noqa: E402
import run_final_experiments as rf  # noqa: E402

LEFT_C, RIGHT_C, INK, MUTED = ov.LEFT_C, ov.RIGHT_C, ov.INK, ov.MUTED
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "pdf.fonttype": 42, "axes.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False})


def sample(fig, ax, meta, w, col, lab, fs=5.0):
    """Top of writer w's first shared paragraph, cropped to the axes' shape (no stretching)."""
    r = meta[(meta.writer_id == w) & (meta.paragraph_number == 1)].iloc[0]
    bb = ax.get_position(); ratio = (bb.width * fig.get_figwidth()) / (bb.height * fig.get_figheight())
    img = ov.text_block(ROOT / "dataset" / r.relative_path, ratio, width=900)
    while (img[-3:].mean() < 40):                         # crop ran past the photo: show fewer lines instead
        ratio *= 1.05; img = ov.text_block(ROOT / "dataset" / r.relative_path, ratio, width=900)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="none")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(col); s.set_linewidth(1.0)
    ax.text(0.02, 0.06, lab, transform=ax.transAxes, fontsize=fs, color="white", bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none"))


def writer_darkness(run):
    pp = pd.read_csv(run / "pseudo_pressure_images.csv")
    pp = pp[pp.image_id.str.extract(r"para(\d+)")[0].astype(int) <= 6]
    wm = pp.groupby("writer_id").agg(hand=("hand", "first"), d=("darkness_at_ref_width", "mean"))
    return wm, pd.read_csv(run / "pseudo_pressure_tests.csv").set_index("measure").loc["darkness_at_ref_width"]


def variant2(run, out, meta, items, hc_p):
    fig = plt.figure(figsize=(13 / 2.54, 6.2 / 2.54))
    fig.text(0.015, 0.965, "Handedness of unseen writers is visible in photographed Bangla handwriting,",
             fontsize=7.2, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.885, "partly through darker stroke centres in right-handed writers",
             fontsize=7.2, fontweight="bold", color=INK, va="top")
    g = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.45, 0.95], wspace=0.22, left=0.015, right=0.97, top=0.72, bottom=0.2)
    head = lambda spec, s: fig.text(spec.get_position(fig).x0, 0.78, s, fontsize=6.0, color=MUTED, va="bottom")

    sub = g[0, 0].subgridspec(2, 1, hspace=0.1)
    for k, (w, col, lab) in enumerate([("L_p11", LEFT_C, "left-handed"), ("R_p16", RIGHT_C, "right-handed")]):
        sample(fig, fig.add_subplot(sub[k]), meta, w, col, lab)
    head(g[0, 0], "same prompt, two hands")

    ax = fig.add_subplot(g[0, 1])
    y = 1.45 * np.arange(len(items))[::-1]
    ax.barh(y + 0.19, [i[1] for i in items], 0.36, color="#374151", label="raw")
    ax.barh(y - 0.19, [i[2] for i in items], 0.36, color="#9ca3af", label="darkness and brightness removed")
    for yi, (_, r_, a_) in zip(y, items):
        ax.text(r_ + 0.6, yi + 0.19, f"{r_:.0f}", va="center", fontsize=5.0)
        ax.text(a_ + 0.6, yi - 0.19, f"{a_:.0f}", va="center", fontsize=5.0, color=MUTED)
    for yi, (name, _, _) in zip(y, items):
        ax.text(40.5, yi + 0.47, name, va="bottom", fontsize=5.2, color=INK, zorder=3,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"))
    ax.axvline(50, color=MUTED, lw=0.6, ls=":"); ax.text(50.4, -0.62, "chance", fontsize=4.6, color=MUTED, va="bottom")
    ax.set_xlim(40, 84); ax.set_ylim(-0.65, y[0] + 1.05); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=5); ax.set_xlabel("writer balanced accuracy (%), unseen writers", fontsize=5.2)
    ax.legend(frameon=False, fontsize=4.8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handlelength=0.9,
              borderaxespad=0.1, columnspacing=0.8)
    head(g[0, 1], f"unseen writers (HC-183 p = {hc_p:.3f})")

    ax = fig.add_subplot(g[0, 2])
    wm, t = writer_darkness(run)
    rng = np.random.default_rng(1)
    for k, (h, col) in enumerate([("right", RIGHT_C), ("left", LEFT_C)]):
        v = wm[wm.hand == h].d.to_numpy()
        ax.scatter(v, k + rng.uniform(-0.15, 0.15, len(v)), s=4, color=col, alpha=0.75, lw=0)
        ax.plot([v.mean()] * 2, [k - 0.28, k + 0.28], color=INK, lw=1.0)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["right", "left"], fontsize=5.2); ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("centre darkness (equal width)", fontsize=5.2); ax.tick_params(labelsize=4.8)
    ax.text(0.98, 0.98, f"g = {t.hedges_g:.2f}\np = {t.p_perm:.3f}".replace("-", "\u2212"), transform=ax.transAxes,
            ha="right", va="top", fontsize=5.0)
    head(g[0, 2], "cue: darker stroke centres")

    fig.text(0.015, 0.025, "42 writers (21 left, 21 right)  \u00b7  shared pens, paper, phone, session  \u00b7  test writers fully "
             "held out  \u00b7  999 permutations  \u00b7  photometric controls",
             fontsize=5.0, color=MUTED, va="bottom")
    return fig


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="results/final42_2026-10-19")
    ap.add_argument("--output", default="manuscripts/prl/graphical_abstract")
    ap.add_argument("--variant", type=int, default=1, choices=[1, 2])
    a = ap.parse_args(); run, out = ROOT / a.run, ROOT / a.output; out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    sm = pd.read_csv(run / "metrics_summary.csv"); ad = pd.read_csv(run / "adjusted.csv")
    gs_ = pd.read_csv(run / "finetune_grey_summary.csv").set_index("readout")
    perm = pd.read_csv(run / "permutation_tests.csv").set_index("rep")
    raw = lambda r: 100 * sm[(sm.rep == r) & (sm.clf == "svm") & (sm.analysis == "A")].ba_mean.iloc[0]
    adj = lambda r: 100 * ad[(ad.rep == r) & (ad.subset == "all")].adjusted_mean.iloc[0]
    if a.variant == 2:
        items = [("HC-183 (gradients)", raw("hc183"), adj("hc183")), ("CNN, frozen", raw("resnet18"), adj("resnet18")),
                 ("CNN, fine-tuned", 100 * gs_.loc["svm_A", "ba_mean"], 100 * gs_.loc["svm_adj_A", "ba_mean"]),
                 ("STROKE (shape only)", raw("stroke"), adj("stroke"))]
        save(variant2(run, out, meta, items, perm.loc["hc183", "p_mean"]), out, "graphical_abstract_v2")
        return

    fig = plt.figure(figsize=(13 / 2.54, 6.0 / 2.54))
    g = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.68, 1.5, 0.8], wspace=0.55, left=0.015, right=0.975, top=0.8, bottom=0.27)
    title = lambda spec, s: fig.text(spec.get_position(fig).x0, 0.95, s, fontsize=6.3, fontweight="bold", color=INK, va="top")
    note = lambda spec, s: fig.text(spec.get_position(fig).x0, 0.02, s, fontsize=5.4, color=MUTED, va="bottom")

    # 1. data: the same shared prompt by a left- and a right-handed writer
    sub = g[0, 0].subgridspec(2, 1, hspace=0.1)
    for k, (w, col, lab) in enumerate([("L_p11", LEFT_C, "left-handed"), ("R_p16", RIGHT_C, "right-handed")]):
        ax = fig.add_subplot(sub[k])
        r = meta[(meta.writer_id == w) & (meta.paragraph_number == 1)].iloc[0]
        bb = ax.get_position(); ratio = (bb.width * fig.get_figwidth()) / (bb.height * fig.get_figheight())
        img = ov.text_block(ROOT / "dataset" / r.relative_path, ratio, width=900)
        while (img[-3:].mean() < 40):                     # crop ran past the photo: show fewer lines instead
            ratio *= 1.05; img = ov.text_block(ROOT / "dataset" / r.relative_path, ratio, width=900)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="none")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(col); s.set_linewidth(1.0)
        ax.text(0.02, 0.06, lab, transform=ax.transAxes, fontsize=5.0, color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none"))
    title(g[0, 0], "BanglaHand\n42 writers")
    note(g[0, 0], "21 left, 21 right; shared\npens, paper, phone, session")

    # 2. writer-independent folds (one repetition)
    ax = fig.add_subplot(g[0, 1])
    wt = meta.groupby("writer_id").label.first().sort_values(ascending=False)
    folds = rf.writer_folds(wt.index.to_numpy(), wt.to_numpy(), 2026)
    for f in range(5):
        for j, fo in enumerate(folds):
            ax.add_patch(plt.Rectangle((j, 4 - f), 0.9, 0.8, color=INK if fo == f else "#d1d5db", lw=0))
    ax.axvline(20.95, color="white", lw=1.2)
    ax.set_xlim(0, 42); ax.set_ylim(-0.9, 5); ax.axis("off")
    ax.text(10.5, -0.7, "left", ha="center", fontsize=5.2, color=LEFT_C); ax.text(31.5, -0.7, "right", ha="center", fontsize=5.2, color=RIGHT_C)
    title(g[0, 1], "Test on\nunseen writers")
    note(g[0, 1], "dark: test writers,\nall images held out")

    # 3. accuracy before and after removing ink darkness and brightness
    ax = fig.add_subplot(g[0, 2])
    items = [("HC-183", raw("hc183"), adj("hc183")), ("frozen\nCNN", raw("resnet18"), adj("resnet18")),
             ("tuned\nCNN", 100 * gs_.loc["svm_A", "ba_mean"], 100 * gs_.loc["svm_adj_A", "ba_mean"]),
             ("STROKE\n(shape)", raw("stroke"), adj("stroke"))]
    x = np.arange(len(items))
    ax.bar(x - 0.19, [i[1] for i in items], 0.36, color="#374151", label="raw")
    ax.bar(x + 0.19, [i[2] for i in items], 0.36, color="#9ca3af", label="darkness, brightness removed")
    for xi, (_, r_, a_) in zip(x, items):
        ax.text(xi - 0.19, r_ + 0.8, f"{r_:.0f}", ha="center", fontsize=5.0); ax.text(xi + 0.19, a_ + 0.8, f"{a_:.0f}", ha="center", fontsize=5.0)
    ax.axhline(50, color=MUTED, lw=0.6, ls=":"); ax.text(3.97, 50.6, "chance", fontsize=4.6, color=MUTED, ha="right")
    ax.set_ylim(40, 92); ax.set_xlim(-0.55, 4.0); ax.set_xticks(x); ax.set_xticklabels([i[0] for i in items], fontsize=4.8)
    ax.tick_params(labelsize=5); ax.set_ylabel("writer balanced accuracy (%)", fontsize=5.2)
    ax.legend(frameon=False, fontsize=4.8, loc="upper left", ncol=2, handlelength=0.9, borderaxespad=0, columnspacing=0.8)
    title(g[0, 2], f"Handedness identified\n(HC-183 77%, p = {perm.loc['hc183', 'p_mean']:.3f})".replace("77%", f"{raw('hc183'):.0f}%"))
    note(g[0, 2], "darkness and brightness removed: HC-183\nkeeps most, CNNs fall near chance")

    # 4. stroke-centre darkness at equal width, writer means
    ax = fig.add_subplot(g[0, 3])
    pp = pd.read_csv(run / "pseudo_pressure_images.csv")
    pp = pp[pp.image_id.str.extract(r"para(\d+)")[0].astype(int) <= 6]
    wm = pp.groupby("writer_id").agg(hand=("hand", "first"), d=("darkness_at_ref_width", "mean"))
    rng = np.random.default_rng(1)
    for k, (h, col) in enumerate([("left", LEFT_C), ("right", RIGHT_C)]):
        v = wm[wm.hand == h].d.to_numpy()
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=4, color=col, alpha=0.75, lw=0)
        ax.plot([k - 0.25, k + 0.25], [v.mean()] * 2, color=INK, lw=1.0)
    t = pd.read_csv(run / "pseudo_pressure_tests.csv").set_index("measure").loc["darkness_at_ref_width"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(["left", "right"], fontsize=5.2); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("relative darkness", fontsize=5.2); ax.tick_params(labelsize=4.8)
    title(g[0, 3], "Cue: darker\nstroke centres")
    note(g[0, 3], "right-handed darker\n" + f"g = {t.hedges_g:.2f}, p = {t.p_perm:.3f}".replace("-", "\u2212"))

    save(fig, out, "graphical_abstract")


def save(fig, out, name):
    for ext, kw in [("pdf", {}), ("png", {"dpi": 600}), ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(out / f"{name}.{ext}", **kw)
    plt.close(fig)
    print("written", out / name)


if __name__ == "__main__":
    main()
