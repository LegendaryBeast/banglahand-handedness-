#!/usr/bin/env python3
"""Feature baselines from earlier handwriting soft-biometric and writer-identification work (IJDAR extension).

Re-implementations, not the original code. The parameter choices we had to make are given here and in the paper;
check them against the original papers before submission.

  edge_direction  f1 of Bulacu & Schomaker (2007): contour-fragment direction histogram, fragment length n = 5 px,
                  12 bins over [0, 180) degrees.
  edge_hinge      f2 of Bulacu & Schomaker (2007): joint histogram of the two legs (n = 5 px) at each contour
                  point, 24 bins over 360 degrees, phi2 > phi1 (276 cells).
  chain_code      Freeman 8-direction chain-code histograms of contours, 1st order (8) and 2nd order (64), and a
                  12-bin curvature (turning-angle) histogram: the contour features used in the soft-biometric work
                  of Al-Maadeed and colleagues (direction, curvature, chain codes).
  hog_glbp        HOG of the grey text block (9 orientations, 32-px cells, 2x2 blocks, text block resized to
                  512 x 256) and a gradient-weighted uniform LBP histogram (P = 8, R = 1), in the spirit of the
                  HOG and GLBP features of Bouadjenek et al.
  lbp             uniform LBP histograms of the grey text block, (P, R) = (8, 1) and (16, 2).
All are computed on the text block at 1,536 px width (the STROKE ink mask and crop), so margins do not enter.
Fine-tuned CNN baselines (Morera et al. 2018; Rahmanian et al. 2021) are represented by the fine-tuned ResNet-18
already in results/final42_2026-10-19.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.measure import find_contours

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stroke_features as sf  # noqa: E402

TWO_PI = 2 * np.pi
N_LEG = 5


def contours_resampled(ink, step=1.0):
    out = []
    for c in find_contours(np.pad(ink, 1).astype(float), 0.5, positive_orientation="high"):
        seg = np.diff(c, axis=0); s = np.concatenate([[0], np.cumsum(np.hypot(*seg.T))])
        if s[-1] < 4 * N_LEG:
            continue
        t = np.arange(0, s[-1], step)
        out.append(np.column_stack([np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])]))
    return out


def norm(h):
    h = np.asarray(h, float).ravel(); return h / (h.sum() + 1e-12)


def contour_baselines(ink):
    f1, a1, a2, codes2, turn = [], [], [], [], []
    for p in contours_resampled(ink):
        n = len(p); idx = np.arange(n)
        fwd = p[(idx + N_LEG) % n] - p; bwd = p[(idx - N_LEG) % n] - p
        ang = lambda v: np.mod(np.arctan2(-v[:, 0], v[:, 1]), TWO_PI)
        af, ab = ang(fwd), ang(bwd)
        f1.append(np.mod(af, np.pi))
        a1.append(af); a2.append(ab)
        # chain code: 8 directions of unit steps along the contour
        st = np.diff(p, axis=0, append=p[:1])
        cc = np.mod(np.round(np.arctan2(-st[:, 0], st[:, 1]) / (np.pi / 4)).astype(int), 8)
        codes2.append(cc)
        turn.append(np.angle(np.exp(1j * (af - np.mod(ab + np.pi, TWO_PI)))))
    f1 = np.concatenate(f1); a1 = np.concatenate(a1); a2 = np.concatenate(a2); turn = np.concatenate(turn)
    edge_direction = norm(np.histogram(f1, 12, (0, np.pi))[0])
    b1 = np.minimum((a1 / TWO_PI * 24).astype(int), 23); b2 = np.minimum((a2 / TWO_PI * 24).astype(int), 23)
    lo, hi = np.minimum(b1, b2), np.maximum(b1, b2)
    H = np.zeros((24, 24)); np.add.at(H, (lo, hi), 1)
    iu = np.triu_indices(24, 1)
    edge_hinge = norm(H[iu])
    c1 = np.zeros(8); c2 = np.zeros((8, 8))
    for cc in codes2:
        np.add.at(c1, cc, 1); np.add.at(c2, (cc, np.roll(cc, -1)), 1)
    chain = np.concatenate([norm(c1), norm(c2), norm(np.histogram(turn, 12, (-np.pi, np.pi))[0])])
    return edge_direction, edge_hinge, chain


def grey_block(gray, box):
    w, h = gray.size
    a = np.asarray(gray.resize((sf.WIDTH, round(sf.WIDTH * h / w)), Image.Resampling.LANCZOS), np.float32)
    return a[box[0]:box[1], box[2]:box[3]]


def texture_baselines(block):
    b = (block - block.mean()) / (block.std() + 1e-6)                   # per-image intensity normalisation
    img = Image.fromarray(np.clip(128 + 40 * b, 0, 255).astype(np.uint8))
    small = np.asarray(img.resize((512, 256), Image.Resampling.BILINEAR), float) / 255
    h = hog(small, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), feature_vector=True)
    g8 = np.asarray(img)
    gy, gx = np.gradient(g8.astype(float)); mag = np.hypot(gx, gy)
    l81 = local_binary_pattern(g8, 8, 1, "uniform")
    glbp = norm(np.histogram(l81, 10, (0, 10), weights=mag)[0])
    l162 = local_binary_pattern(g8, 16, 2, "uniform")
    lbp = np.concatenate([norm(np.histogram(l81, 10, (0, 10))[0]), norm(np.histogram(l162, 18, (0, 18))[0])])
    return np.concatenate([h, glbp]), lbp


def job(rel):
    with Image.open(ROOT / "dataset" / rel) as im:
        gray = im.convert("L").copy()
    ink, box = sf.ink_mask(gray, return_box=True)
    ed, eh, cc = contour_baselines(ink)
    hg, lbp = texture_baselines(grey_block(gray, box))
    return dict(edge_direction=ed, edge_hinge=eh, chain_code=cc, hog_glbp=hg, lbp=lbp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/ijdar_dev_2026-11-12/features")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); out = ROOT / a.output; out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(job, meta.relative_path, chunksize=2))
    arrays = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
    np.savez_compressed(out / "baselines.npz", image_id=meta.image_id.to_numpy(), **arrays)
    (out / "baselines_provenance.json").write_text(json.dumps({k: v.shape[1] for k, v in arrays.items()}, indent=2))
    print({k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    main()
