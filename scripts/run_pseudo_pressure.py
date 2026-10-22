#!/usr/bin/env python3
"""Pseudo-pressure analysis (protocol/protocol_amendment_4.json; post hoc, descriptive).

Offline proxies of pen pressure from ink deposition, measured on the full-resolution photographs of the shared
paragraphs (1-6) relative to the local paper level (so overall brightness cancels):
  core_darkness          mean (paper - ink)/paper on stroke centreline pixels
  edge_darkness          same on boundary ink pixels (distance to background < 2 px)
  core_edge_contrast     core_darkness - edge_darkness
  darkness_at_ref_width  per-image linear fit of centreline darkness on local stroke width, at the median width of all images
  stroke_width           2 x mean distance transform on centreline pixels (px at 4080-px width)
  dark_core_fraction     fraction of centreline pixels with relative darkness > 0.6
Writes pseudo_pressure_images.csv and pseudo_pressure_tests.csv to the run folder.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MEASURES = ["core_darkness", "edge_darkness", "core_edge_contrast", "darkness_at_ref_width", "stroke_width", "dark_core_fraction"]


def measure(rel):
    g = cv2.imread(str(ROOT / "dataset" / rel), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    s = g.shape[1] / 1536                                            # same ink definition as STROKE, scaled to full resolution
    bg = cv2.GaussianBlur(g, (0, 0), 34 * s)
    ink = ((bg - g) > 17) & (bg > 120) & (g < 205)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 12 * s * s
    ink = keep[lab]
    rel_dark = np.clip((bg - g) / np.maximum(bg, 1), 0, 1)
    edt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(ink) & (edt >= 1.5)                              # centreline of strokes at least 3 px wide
    core, width = rel_dark[sk], 2 * edt[sk]
    edge = rel_dark[ink & (edt < 2)]
    return dict(core=core, width=width, edge_mean=float(edge.mean()))


def summarise(r, ref_w):
    core, width = r["core"], r["width"]
    b = np.polyfit(width, core, 1)
    return dict(core_darkness=float(core.mean()), edge_darkness=r["edge_mean"], core_edge_contrast=float(core.mean() - r["edge_mean"]),
                darkness_at_ref_width=float(np.polyval(b, ref_w)), stroke_width=float(width.mean()),
                dark_core_fraction=float((core > 0.6).mean()))


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); run = ROOT / a.run
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    m = meta[meta.paragraph_number <= 6].reset_index(drop=True)
    with ProcessPoolExecutor(a.workers) as ex:
        raw = list(ex.map(measure, m.relative_path, chunksize=4))
    ref_w = float(np.median(np.concatenate([r["width"] for r in raw])))
    img = pd.concat([m[["image_id", "writer_id", "hand", "label", "capture_datetime"]], pd.DataFrame([summarise(r, ref_w) for r in raw])], axis=1)
    img["ref_width_px"] = ref_w
    img.to_csv(run / "pseudo_pressure_images.csv", index=False)

    t = pd.to_datetime(img.capture_datetime, format="%Y:%m:%d %H:%M:%S")
    img["t"] = (t - t.min()).dt.total_seconds() / 60
    wm = img.groupby("writer_id").agg(label=("label", "first"), t=("t", "mean"), **{k: (k, "mean") for k in MEASURES})
    rng = np.random.default_rng(2029)
    perms = [rng.permutation(wm.label.to_numpy()) for _ in range(10000)]
    rows = []
    for k in MEASURES:
        x, y = wm[k].to_numpy(), wm.label.to_numpy()
        obs = hedges_g(x[y == 1], x[y == 0])
        null = np.array([hedges_g(x[p == 1], x[p == 0]) for p in perms])
        rows.append(dict(measure=k, left_mean=x[y == 1].mean(), right_mean=x[y == 0].mean(), hedges_g=obs,
                         p_perm=(np.sum(np.abs(null) >= abs(obs)) + 1) / 10001,
                         rho_time=spearmanr(wm.t, wm[k]).statistic, p_time=spearmanr(wm.t, wm[k]).pvalue))
    d = pd.DataFrame(rows)
    o = np.argsort(d.p_perm.to_numpy()); q = d.p_perm.to_numpy()[o] * len(d) / (np.arange(len(d)) + 1)
    d.loc[o, "q_bh"] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1)
    d.to_csv(run / "pseudo_pressure_tests.csv", index=False)
    print(f"reference width {ref_w:.2f} px"); print(d.round(4).to_string())


if __name__ == "__main__":
    main()
