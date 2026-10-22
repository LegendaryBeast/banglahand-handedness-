#!/usr/bin/env python3
"""Figures for the Data in Brief article, generated from the released files and audit outputs."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_prl_overview_figures import for_display  # noqa: E402
LEFT_C, RIGHT_C = "#c2410c", "#1d4ed8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def thumb(rel, width=1600):
    """Whole photograph, display-corrected (lighting flattened, contrast stretched; see make_prl_overview_figures)."""
    with Image.open(ROOT / "dataset" / rel) as im:
        im = for_display(im.convert("L"))
        return im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS)


def fig_samples(meta, out):
    writers = ["L_p3", "L_p11", "R_p16", "R_p2"]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.0), constrained_layout=True)
    for j, w in enumerate(writers):
        g = meta[meta.writer_id == w].set_index("paragraph_number")
        for i, k in enumerate([1, 7]):
            ax = axes[i, j]
            ax.imshow(thumb(g.loc[k, "relative_path"]), cmap="gray", vmin=0, vmax=255, interpolation="none")   # full resolution in the PDF
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(LEFT_C if w.startswith("L") else RIGHT_C); s.set_linewidth(1.2)
            hand = "left-handed" if w.startswith("L") else "right-handed"
            lab = "shared prompt S01" if k == 1 else f"writer-specific:\n{g.loc[k, 'topic_gist']}"
            ax.set_title(f"{w} ({hand})\nparagraph {k}, {lab}", fontsize=6.2)
    fig.savefig(out / "fig_samples.pdf"); fig.savefig(out / "fig_samples.png", dpi=220); plt.close(fig)


def fig_capture(meta, out):
    m = meta.copy()
    m["t"] = pd.to_datetime(m.capture_datetime, format="%Y:%m:%d %H:%M:%S")
    order = m.groupby("writer_id").t.min().sort_values()
    fig, ax = plt.subplots(figsize=(3.5, 3.4), constrained_layout=True)
    for i, (w, t0) in enumerate(order.items()):
        g = m[m.writer_id == w]
        c = LEFT_C if w.startswith("L") else RIGHT_C
        hrs = g.t.dt.hour + g.t.dt.minute / 60 + g.t.dt.second / 3600
        ax.plot([hrs.min(), hrs.max()], [i, i], color=c, lw=2.2, solid_capstyle="round")
        ax.text(hrs.max() + 0.06, i, w, va="center", fontsize=4.8, color=c)
    ax.set_yticks([]); ax.set_ylabel("writers in order of photography")
    day = m.t.dt.strftime("%d %b %Y").iloc[0]
    ax.set_xlabel(f"time of photography on {day} (h)")
    ax.set_xlim(m.t.dt.hour.min() - 0.2, m.t.dt.hour.max() + 1.6)
    ax.plot([], [], color=LEFT_C, lw=2.2, label="left-handed"); ax.plot([], [], color=RIGHT_C, lw=2.2, label="right-handed")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("One session, one phone", fontsize=6.6)
    fig.savefig(out / "fig_capture_timeline.pdf"); fig.savefig(out / "fig_capture_timeline.png", dpi=220); plt.close(fig)


def fig_validation(out):
    A = ROOT / "audit/geometric"
    g = pd.read_csv(A / "geometric_pairs.csv.gz")
    wid = lambda k: k.split("/")[1].rsplit("_para", 1)[0]
    slot = lambda k: int(k.rsplit("para", 1)[1])
    within = g[g.kind == "same_writer"]
    split = within.apply(lambda r: wid(r.a) in ("L_p14", "L_p21") and {slot(r.a), slot(r.b)} == {7, 8}, axis=1)
    raw = g[(g.kind == "unreleased_vs_released") & g.a.str.startswith("unreleased/P19") & g.b.str.startswith("released/R_p6_")]
    groups = {
        "different writers, same prompt": g[g.kind == "same_prompt_different_writers"].inliers,
        "same writer, different paragraphs": within[~split].inliers,
        "unreleased raw folder vs R_p6\n(re-photographed pages)": raw.inliers,
    }
    cols = ["#9ca3af", "#6b7280", "#dc2626"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    bins = np.logspace(0, np.log10(3000), 40)
    for (k, v), c in zip(groups.items(), cols):
        ax.hist(np.clip(v, 1, None), bins=bins, histtype="stepfilled", alpha=0.55, color=c, label=f"{k} (n={len(v)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("SIFT-RANSAC inliers per image pair")
    ax.set_ylabel("image pairs")
    lo = int(groups["different writers, same prompt"].max()); hi = int(raw.inliers.min())
    ax.legend(frameon=False, fontsize=5.4, loc="upper right")
    fig.savefig(out / "fig_duplicate_screen.pdf"); fig.savefig(out / "fig_duplicate_screen.png", dpi=220); plt.close(fig)
    return {k.replace("\n", " "): (int(v.min()), int(np.median(v)), int(v.max()), len(v)) for k, v in groups.items()}


def fig_stats(meta, desc, out):
    d = desc.merge(meta[["image_id", "writer_id", "hand", "condition"]], on="image_id")
    wm = d.groupby(["writer_id", "hand"])[["ink_density", "stroke_width_px1536", "slant_deg"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.0), constrained_layout=True)
    labels = {"ink_density": "(a) ink pixel fraction", "stroke_width_px1536": "(b) mean stroke width (px, 1536-px image)",
              "slant_deg": "(c) slant of near-vertical strokes (deg)"}
    rng = np.random.default_rng(1)
    for ax, c in zip(axes, labels):
        for i, (h, col) in enumerate([("left", LEFT_C), ("right", RIGHT_C)]):
            v = wm[wm.hand == h][c].to_numpy()
            ax.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color="k"))
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=col, alpha=0.8, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"left (n={(wm.hand == 'left').sum()})", f"right (n={(wm.hand == 'right').sum()})"])
        ax.set_title(labels[c], fontsize=7)
    fig.savefig(out / "fig_writer_statistics.pdf"); fig.savefig(out / "fig_writer_statistics.png", dpi=220); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); ap.add_argument("--features", default="")
    a = ap.parse_args()
    out = ROOT / a.output; out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    fig_samples(meta, out)
    fig_capture(meta, out)
    print(fig_validation(out))
    if a.features:
        fig_stats(meta, pd.read_csv(ROOT / a.features), out)


if __name__ == "__main__":
    main()
