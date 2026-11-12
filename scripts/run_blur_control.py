#!/usr/bin/env python3
"""Blur control for the MATRA-INK cross-section result (IJDAR extension; exploratory).

Question. Right-handed writers' headstrokes have a flatter darkness profile across the stroke (more ink towards
the edges relative to the centre). Optical blur (focus, camera distance) flattens profiles too. Is the group
difference an acquisition effect?

Per image (full resolution, same ink rule as matra_features.py):
  stem_cross_00..12   cross-section profile of near-vertical stems, horizontal cut, same positions U as the
                      headstroke profile (in half-widths), divided by the centre value
  edge_width_px       optical edge width: mean distance (px) between 25% and 75% of the peak darkness on both
                      flanks of stem cross-sections sampled every 0.25 px (absolute pixels, not width-normalised)
  paper_sharpness     robust high-frequency energy of paper pixels (Laplacian of the grey image outside dilated
                      ink, median absolute value / local paper level): an ink-free measure of focus
  stem_half_width_px  median half-width of stems
Then, with the headstroke profile from matra_features.npz:
  ratio_cross_k = matra cross_k / stem_cross_k  (blur affects both directions alike; it cancels in the ratio)

Analyses (writer means over shared paragraphs 1-6, 10,000 label permutations, BH within families):
  1. group differences of blur measures and their correlation with capture time
  2. group differences of the headstroke profile, the stem profile and the ratio profile
  3. headstroke profile differences after regressing blur measures and capture time out (writer level)
  4. classification (20 x 5-fold writer CV, SVM): deposition block, raw and with fold-wise residualisation on
     blur + capture time + ink density + photo intensity; 999-permutation test of the adjusted model
Writes blur_measures.npz, blur_tests.csv, blur_classification.csv into the dev folder.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import matra_features as mf  # noqa: E402

def stem_points(ink, w, max_points=4000):
    L = max(15, 2 * int(round(2 * w)) + 1)
    vert = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(vert, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_HEIGHT] <= 12 * w
    v = keep[lab]
    sk = skeletonize(v)
    ys, xs = np.nonzero(sk)
    if len(ys) > max_points:
        sel = np.random.default_rng(0).choice(len(ys), max_points, replace=False)
        ys, xs = ys[sel], xs[sel]
    return ys, xs


def measure(rel):
    cv2.setNumThreads(1)
    g = cv2.imread(str(ROOT / "dataset" / rel), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    ink, r, w, _ = mf.ink_and_darkness(g)
    edt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = stem_points(ink, w)
    half = np.maximum(edt[ys, xs], 1.0)                                   # local half-width (px)
    # cross-section at the headstroke positions U (in half-widths), horizontal cut
    prof = np.stack([ndi.map_coordinates(r, [ys, xs + u * half], order=1, mode="nearest") for u in mf.U])
    centre = np.max([ndi.map_coordinates(r, [ys, xs + d], order=1, mode="nearest") for d in (-1, 0, 1)], axis=0)
    ok = centre > 0.1                                                     # points that lie on dark ink
    sc = prof[:, ok].mean(1); sc = sc / max(sc[len(mf.U) // 2], 1e-6)     # same normalisation as the headstroke profile
    # optical edge width in absolute px: fine profile from -(half+6) to +(half+6) px in 0.25 px steps
    offs = np.arange(-1.0, 1.0001, 0.02)                                  # fraction of (half + 6 px)
    span = half + 6.0
    fp = np.stack([ndi.map_coordinates(r, [ys, xs + o * span], order=1, mode="nearest") for o in offs])  # (n_off, n)
    peak = fp.max(0)
    widths = []
    for i in np.flatnonzero(ok & (peak > 0.15))[:1500]:
        p = fp[:, i] / peak[i]; x = offs * span[i]
        c = np.argmax(p)
        for side in (slice(0, c + 1), slice(c, len(p))):
            pp, xx = p[side], x[side]
            if side.start == 0:
                pp, xx = pp[::-1], xx[::-1]                               # from peak outwards
            try:
                x75 = xx[np.flatnonzero(pp < 0.75)[0]]; x25 = xx[np.flatnonzero(pp < 0.25)[0]]
                widths.append(abs(x25 - x75))
            except IndexError:
                pass
    # paper sharpness: Laplacian energy of paper pixels, relative to local paper level
    s = g.shape[1] / 1536
    bg = cv2.GaussianBlur(g, (0, 0), 34 * s)
    paper = ~cv2.dilate(ink.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    paper[:int(40 * s)] = paper[-int(40 * s):] = False; paper[:, :int(40 * s)] = paper[:, -int(40 * s):] = False
    lap = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.7), cv2.CV_32F, ksize=3)
    ps = float(np.median(np.abs(lap[paper]) / np.maximum(bg[paper], 1)))
    return np.concatenate([sc, [float(np.median(widths)) if widths else np.nan, ps, float(np.median(half))]])


NAMES = [f"stem_cross_{i:02d}" for i in range(len(mf.U))] + ["edge_width_px", "paper_sharpness", "stem_half_width_px"]


def hedges(v, y):
    a, b = v[y == 1], v[y == 0]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9))


def perm_tests(wm, cols, family, n_perm=10000, seed=2031):
    y = wm.y.to_numpy().astype(int)
    rng = np.random.default_rng(seed); perms = np.array([rng.permutation(y) for _ in range(n_perm)])
    rows = []
    for c in cols:
        v = wm[c].to_numpy(float)
        obs = v[y == 1].mean() - v[y == 0].mean()
        null = (perms * v).sum(1) / y.sum() - ((1 - perms) * v).sum(1) / (1 - y).sum()
        rows.append(dict(family=family, measure=c, left_mean=v[y == 1].mean(), right_mean=v[y == 0].mean(),
                         hedges_g=hedges(v, y), p_perm=(np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1),
                         rho_time=spearmanr(wm.t, v).statistic, p_time=spearmanr(wm.t, v).pvalue))
    t = pd.DataFrame(rows)
    p = t.p_perm.to_numpy(); o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1)
    qq = np.empty(len(p)); qq[o] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1); t["q_bh_family"] = qq
    return t


def residualise(wm, cols, covs):
    out = wm.copy()
    C = np.column_stack([np.ones(len(wm))] + [wm[c].to_numpy(float) for c in covs])
    for c in cols:
        b, *_ = np.linalg.lstsq(C, wm[c].to_numpy(float), rcond=None)
        out[c] = wm[c].to_numpy(float) - C[:, 1:] @ b[1:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="results/ijdar_dev_2026-11-12")
    ap.add_argument("--base", default="results/final42_2026-10-19")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--skip-extract", action="store_true")
    a = ap.parse_args()
    dev = ROOT / a.dev
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    out_npz = dev / "features/blur_measures.npz"
    if not a.skip_extract:
        with ProcessPoolExecutor(a.workers) as ex:
            B = np.array(list(ex.map(measure, meta.relative_path, chunksize=2)))
        np.savez_compressed(out_npz, image_id=meta.image_id.to_numpy(), names=np.array(NAMES), blur=B)
        print("extracted", B.shape, "nan:", int(np.isnan(B).sum()), flush=True)
    z = np.load(out_npz, allow_pickle=True); assert list(z["image_id"]) == list(meta.image_id)
    B = pd.DataFrame(z["blur"], columns=NAMES)
    M = pd.DataFrame(np.load(dev / "features/matra_features.npz", allow_pickle=True)["matra"], columns=mf.NAMES)
    cross = [f"cross_{i:02d}" for i in range(len(mf.U))]
    df = pd.concat([meta[["image_id", "writer_id", "label", "paragraph_number", "capture_datetime"]], M[cross], B], axis=1)
    for i in range(len(mf.U)):
        df[f"ratio_cross_{i:02d}"] = df[f"cross_{i:02d}"] / df[f"stem_cross_{i:02d}"]
    t = pd.to_datetime(df.capture_datetime, format="%Y:%m:%d %H:%M:%S"); df["t"] = (t - t.min()).dt.total_seconds() / 3600
    desc = pd.read_csv(ROOT / a.base / "features/descriptors.csv").set_index("image_id").loc[meta.image_id]
    df["ink_density"] = desc.ink_density.to_numpy()
    sh = df[df.paragraph_number <= 6]
    num = [c for c in sh.columns if c not in ("image_id", "writer_id", "capture_datetime")]
    wm = sh.groupby("writer_id")[num].mean().rename(columns={"label": "y"})
    # outer bins only (the centre is 1 by construction; positions |u| >= 0.75 carry the shape)
    shape_idx = [i for i, u in enumerate(mf.U) if abs(u) >= 0.75]
    blur_cols = ["edge_width_px", "paper_sharpness", "stem_half_width_px"]
    tests = [perm_tests(wm, blur_cols, "blur measures"),
             perm_tests(wm, [f"cross_{i:02d}" for i in shape_idx], "headstroke profile (raw)"),
             perm_tests(wm, [f"stem_cross_{i:02d}" for i in shape_idx], "stem profile"),
             perm_tests(wm, [f"ratio_cross_{i:02d}" for i in shape_idx], "headstroke / stem ratio"),
             perm_tests(residualise(wm, [f"cross_{i:02d}" for i in shape_idx], blur_cols + ["t"]),
                        [f"cross_{i:02d}" for i in shape_idx], "headstroke profile | blur + time")]
    T = pd.concat(tests, ignore_index=True)
    T.to_csv(dev / "blur_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(4).to_string(), flush=True)
    rho = {c: spearmanr(wm[c], wm[f"cross_{i:02d}"]).statistic for c in blur_cols for i in (10, 11)}
    print("Spearman (writer level) blur vs lower-edge headstroke bins:", {k: round(v, 3) for k, v in rho.items()})

    # 4. classification with blur covariates
    import run_final_experiments as rf
    import run_posthoc_confound_analyses as pc
    import run_matra_evaluation as me
    from joblib import Parallel, delayed
    D, _ = me.load(ROOT / a.base, dev)
    ratio = df[[f"ratio_cross_{i:02d}" for i in range(len(mf.U))]].to_numpy(float)
    D.X["ratio_profile"] = np.where(np.isfinite(ratio), ratio, np.nanmedian(ratio, 0))
    Bx = B[blur_cols].to_numpy(float); Bx = np.where(np.isfinite(Bx), Bx, np.nanmedian(Bx, 0))
    cov_acq = np.column_stack([D.desc.ink_density.to_numpy(), D.X["global_intensity"]])
    cov_all = np.column_stack([cov_acq, Bx, df.t.to_numpy(), df.t.to_numpy() ** 2])
    P = Parallel(a.workers)
    rows = []
    for rep in ["matra_deposition", "ratio_profile", "hc183"]:
        for name, cov in [("raw", None), ("ink+photo", cov_acq), ("ink+photo+blur+time", cov_all)]:
            b = P(delayed(pc.cv_A)(D.X[rep], D.y, D.w, D.para, s, None, cov) for s in rf.SEEDS)
            rows.append(dict(rep=rep, adjustment=name, ba_mean=np.mean(b), ba_sd=np.std(b, ddof=1)))
            print(rows[-1], flush=True)
    rng = np.random.default_rng(2029); lab = pd.Series(D.wlabels, D.writers)
    perms = [pd.Series(rng.permutation(lab.to_numpy()), lab.index) for _ in range(a.n_perm)]
    for rep in ["matra_deposition", "ratio_profile"]:
        obs = [r for r in rows if r["rep"] == rep and r["adjustment"] == "ink+photo+blur+time"][0]["ba_mean"]
        def one(pl, rep=rep):
            yp = pl.reindex(D.w).to_numpy()
            return float(np.mean([pc.cv_A(D.X[rep], yp, D.w, D.para, s, None, cov_all) for s in rf.SEEDS]))
        null = np.array(P(delayed(one)(pl) for pl in perms))
        rows.append(dict(rep=rep, adjustment="ink+photo+blur+time (perm)", ba_mean=obs,
                         p=(np.sum(null >= obs) + 1) / (a.n_perm + 1), null_q95=np.quantile(null, .95)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(dev / "blur_classification.csv", index=False)


if __name__ == "__main__":
    main()
