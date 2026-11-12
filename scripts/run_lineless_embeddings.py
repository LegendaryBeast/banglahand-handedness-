#!/usr/bin/env python3
"""Frozen ImageNet embeddings on line-free text blocks (IJDAR, printed-border control).

Same tiles and backbones as extract_final_features.py (224-px tiles of the grey photograph at 1,568 px width, tiles
with >= 1% ink, averaged), but the printed border lines found by run_layout_check.ruled_lines are painted out with the
local paper level and everything outside the text block is set to paper. Evaluated like the other representations
(SVM, condition A, 20 x 5-fold writer CV; ink + photo-intensity adjustment; writer-ID top-1).
Writes <dev>/features/lineless_embeddings.npz and <dev>/lineless_embeddings_summary.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import extract_final_features as ef  # noqa: E402
import run_layout_check as lc  # noqa: E402

NETS = ["resnet18", "efficientnet_b0", "vit_b_16"]


def lineless_gray(rel):
    with Image.open(ROOT / "dataset" / rel) as im:
        g = im.convert("L").copy()
    a = lc.load(rel); ink, shade = lc.ink_of(a)
    lines, _ = lc.ruled_lines(a)
    box = lc.text_box(ink & ~lines)
    keep = np.zeros_like(lines); keep[box[0]:box[1], box[2]:box[3]] = True
    paint = lines | ~keep                                                   # at 1,536 px
    full = np.asarray(g, np.float32)
    up = lambda m, mode: np.asarray(Image.fromarray(m).resize(g.size, mode))
    pm = up(paint.astype(np.uint8) * 255, Image.Resampling.BILINEAR) > 0
    sh = up(shade.astype(np.float32), Image.Resampling.BILINEAR)
    full = np.where(pm, sh, full)
    return Image.fromarray(np.clip(full, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/final42_2026-10-19")
    ap.add_argument("--dev", default="results/ijdar_dev_2026-11-12")
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args(); dev = ROOT / a.dev
    import run_final_experiments as rf
    import run_pitfall_analyses as pf
    import run_posthoc_confound_analyses as pc
    D = rf.Data(ROOT / a.base)
    out = dev / "features/lineless_embeddings.npz"
    if a.reuse and out.exists():
        z = np.load(out); E = {k: z[k] for k in NETS}
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        nets = ef.backbones(device); E = {k: [] for k in NETS}
        for i, rel in enumerate(D.meta.relative_path):
            x, kept, total = ef.tiles(lineless_gray(rel))
            with torch.no_grad():
                for k, net in nets.items():
                    E[k].append(torch.cat([net(x[j:j + 32].to(device)).float().cpu() for j in range(0, len(x), 32)]).mean(0).numpy())
            if i % 50 == 0:
                print(i, rel, kept, "of", total, "tiles", flush=True)
        E = {k: np.array(v, np.float32) for k, v in E.items()}
        np.savez_compressed(out, image_id=D.meta.image_id.to_numpy(), **E)
    for k in NETS:
        D.X[k + "_lineless"] = E[k]
    reps = NETS + [k + "_lineless" for k in NETS]
    P = Parallel(8)
    par = pd.concat(P(delayed(rf.run_cv)(D, r, "svm", s, ("A",)) for r in reps for s in rf.SEEDS), ignore_index=True)
    S = rf.summarize(rf.metrics(rf.writer_level(par)))[["rep", "ba_mean", "ba_sd", "auc_mean"]]
    cov = np.column_stack([D.desc.ink_density.to_numpy(), D.X["global_intensity"]])
    S["adjusted_ba"] = [np.mean(P(delayed(pc.cv_A)(D.X[r], D.y, D.w, D.para, s, None, cov) for s in rf.SEEDS)) for r in S.rep]
    S["writer_id_top1"] = [pf.writer_id_top1(D.X[r], D.w, D.para) for r in S.rep]
    S.to_csv(dev / "lineless_embeddings_summary.csv", index=False)
    print(S.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
