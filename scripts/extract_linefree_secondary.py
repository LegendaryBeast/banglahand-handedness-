#!/usr/bin/env python3
"""Line-free inputs for the secondary analyses and figures (IJDAR; printed lines removed, page outside the text block = paper).

1. Mirrored STROKE features of the line-free text blocks (stroke_features(..., mirror=True) on
   run_lineless_embeddings.lineless_gray) -> <dev>/features/stroke_linefree_mirror.npz (key stroke_lf_mirror).
2. Pseudo-pressure measures of run_pseudo_pressure.py, computed on the line-free full-resolution image instead of the raw
   photograph (identical ink definition and summaries) -> <dev>/linefree_secondary/pseudo_pressure_images.csv and
   pseudo_pressure_tests.csv (same columns as the PRL run, so the figure code reads them unchanged).
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def mirror_job(rel):
    import cv2
    cv2.setNumThreads(1)
    import run_lineless_embeddings as le
    import stroke_features as sf
    return sf.stroke_features(le.lineless_gray(rel), mirror=True)[0]


def pp_job(rel):
    """run_pseudo_pressure.measure on the line-free image (same steps, same constants)."""
    import cv2
    cv2.setNumThreads(1)
    from skimage.morphology import skeletonize
    import run_lineless_embeddings as le
    g = np.asarray(le.lineless_gray(rel), np.float32)
    s = g.shape[1] / 1536
    bg = cv2.GaussianBlur(g, (0, 0), 34 * s)
    ink = ((bg - g) > 17) & (bg > 120) & (g < 205)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 12 * s * s
    ink = keep[lab]
    rel_dark = np.clip((bg - g) / np.maximum(bg, 1), 0, 1)
    edt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(ink) & (edt >= 1.5)
    core, width = rel_dark[sk], 2 * edt[sk]
    edge = rel_dark[ink & (edt < 2)]
    return dict(core=core, width=width, edge_mean=float(edge.mean()))


def main():
    import run_pseudo_pressure as pp
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="results/ijdar_dev_2026-11-12")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args(); dev = ROOT / a.dev; out = dev / "linefree_secondary"; out.mkdir(exist_ok=True)
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    with ProcessPoolExecutor(a.workers) as ex:
        Xm = np.array(list(ex.map(mirror_job, meta.relative_path, chunksize=2)), np.float32)
    np.savez_compressed(dev / "features/stroke_linefree_mirror.npz", image_id=meta.image_id.to_numpy(), stroke_lf_mirror=Xm)
    print("mirror", Xm.shape, flush=True)

    m = meta[meta.paragraph_number <= 6].reset_index(drop=True)
    with ProcessPoolExecutor(a.workers) as ex:
        raw = list(ex.map(pp_job, m.relative_path, chunksize=4))
    ref_w = float(np.median(np.concatenate([r["width"] for r in raw])))
    img = pd.concat([m[["image_id", "writer_id", "hand", "label", "capture_datetime"]],
                     pd.DataFrame([pp.summarise(r, ref_w) for r in raw])], axis=1)
    img["ref_width_px"] = ref_w
    img.to_csv(out / "pseudo_pressure_images.csv", index=False)
    t = pd.to_datetime(img.capture_datetime, format="%Y:%m:%d %H:%M:%S")
    img["t"] = (t - t.min()).dt.total_seconds() / 60
    wm = img.groupby("writer_id").agg(label=("label", "first"), t=("t", "mean"), **{k: (k, "mean") for k in pp.MEASURES})
    rng = np.random.default_rng(2029)
    perms = [rng.permutation(wm.label.to_numpy()) for _ in range(10000)]
    rows = []
    for k in pp.MEASURES:
        x, y = wm[k].to_numpy(), wm.label.to_numpy()
        obs = pp.hedges_g(x[y == 1], x[y == 0])
        null = np.array([pp.hedges_g(x[q == 1], x[q == 0]) for q in perms])
        rows.append(dict(measure=k, left_mean=x[y == 1].mean(), right_mean=x[y == 0].mean(), hedges_g=obs,
                         p_perm=(np.sum(np.abs(null) >= abs(obs)) + 1) / 10001,
                         rho_time=spearmanr(wm.t, wm[k]).statistic, p_time=spearmanr(wm.t, wm[k]).pvalue))
    d = pd.DataFrame(rows)
    o = np.argsort(d.p_perm.to_numpy()); q = d.p_perm.to_numpy()[o] * len(d) / (np.arange(len(d)) + 1)
    d.loc[o, "q_bh"] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1)
    d.to_csv(out / "pseudo_pressure_tests.csv", index=False)
    print(f"reference width {ref_w:.2f} px"); print(d.round(4).to_string())


if __name__ == "__main__":
    main()
