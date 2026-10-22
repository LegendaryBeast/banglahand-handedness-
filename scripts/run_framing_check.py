#!/usr/bin/env python3
"""Amendment 4 (post hoc): separate writing density from photo framing and exposure."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_experiments as rf  # noqa: E402

W = 1536


def measures(path):
    with Image.open(path) as im:
        g = im.convert("L")
    a = np.asarray(g.resize((W, round(W * g.height / g.width)), Image.Resampling.LANCZOS), np.float32)
    shade = ndi.gaussian_filter(a, 17 * W / 768)
    ink = (shade - a > 17) & (shade > 120) & (a < 205)
    b = 8; ink[:b] = ink[-b:] = False; ink[:, :b] = ink[:, -b:] = False
    ink = ndi.binary_opening(ink, np.ones((3, 3)))
    rows = np.flatnonzero(ink.mean(1) > 0.01)
    cols = np.flatnonzero(ink.mean(0) > 0.005)
    r0, r1 = np.percentile(rows, [1, 99]).astype(int)
    c0, c1 = np.percentile(cols, [1, 99]).astype(int)
    block = ink[r0:r1 + 1, c0:c1 + 1]
    paper = ~ndi.binary_dilation(ink, iterations=3)
    return dict(text_block_fraction=block.size / ink.size, ink_density_in_block=float(block.mean()),
                ink_contrast=float(np.median((shade - a)[ink])), paper_level=float(np.median(a[paper])))


def g_and_p(v, y, rng, n=10000):
    d = v[y == 1].mean() - v[y == 0].mean()
    sp = np.sqrt(((y == 1).sum() - 1) * v[y == 1].var(ddof=1) + ((y == 0).sum() - 1) * v[y == 0].var(ddof=1)) / np.sqrt(len(y) - 2)
    null = np.array([v[q == 1].mean() - v[q == 0].mean() for q in (rng.permutation(y) for _ in range(n))])
    return d / sp * (1 - 3 / (4 * len(y) - 9)), (np.sum(np.abs(null) >= abs(d)) + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = ROOT / a.run
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    sh = meta[meta.paragraph_number <= 6]
    rows = [dict(image_id=r.image_id, writer_id=r.writer_id, label=r.label, **measures(ROOT / "dataset" / r.relative_path))
            for r in sh.itertuples()]
    img = pd.DataFrame(rows)
    desc = pd.read_csv(run / "features/descriptors.csv")[["image_id", "ink_density", "stroke_width_px1536"]]
    img = img.merge(desc, on="image_id")
    img.to_csv(run / "framing_check_images.csv", index=False)
    wm = img.groupby("writer_id").mean(numeric_only=True)
    out = []
    for subset, ws in [(f"all{len(wm)}", wm.index)]:
        sub = wm.loc[list(ws)]
        y = sub.label.to_numpy().astype(int)
        for c in ["ink_density", "ink_density_in_block", "text_block_fraction", "stroke_width_px1536", "ink_contrast", "paper_level"]:
            g, p = g_and_p(sub[c].to_numpy(), y, np.random.default_rng(2032))
            out.append(dict(subset=subset, measure=c, left_mean=sub[c][y == 1].mean(), right_mean=sub[c][y == 0].mean(),
                            hedges_g=g, p_perm=p, n_left=int(y.sum()), n_right=int((y == 0).sum())))
    t = pd.DataFrame(out)
    t.to_csv(run / "framing_check_tests.csv", index=False)
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
