#!/usr/bin/env python3
"""Layout check for the HOG + GLBP baseline (IJDAR extension, exploratory).

Question: is the spatial-HOG advantage (71.7% vs 61.1% position-free) a behavioural layout cue (line slope, margins,
spacing) or page/photo framing (ruled lines or drawn frames, page rotation, text-block position)?

Per image (grey at 1,536 px width, the STROKE ink rule):
  ruled lines   long straight ink components (>= 30% of width, or >= 30% of height): counts and angle
  skew          dominant text-line angle (projection-profile variance, ruled lines removed)
  lines         spacing, per-line baseline slope, left starts and right ends, line-length variation
  framing       text-block centre and extent in the photograph
HOG variants: ruled lines painted out (hog_noline); lines painted out + deskewed + re-cropped (hog_registered).
Analyses: writer-level group tests (10,000 permutations, BH), CV of the HOG variants and the layout measures,
HOG with layout covariates regressed out fold-wise, 999-permutation test of the registered HOG.
Writes <dev>/features/layout.npz and <dev>/layout_*.csv.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from joblib import Parallel, delayed
from PIL import Image
from scipy.signal import find_peaks
from skimage.feature import hog
from skimage.morphology import remove_small_objects
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_crop_evaluation import image_to_ink, ink_features  # noqa: E402
W = 1536
ANGLES = np.arange(-8, 8.001, 0.1)

NAMES = ["ruled_h", "ruled_v", "has_ruled", "line_left", "line_right", "line_top", "line_bottom", "line_dist",
         "line_ink_share", "page_rot_deg", "skew_deg", "skew_rel_page", "n_lines",
         "spacing_rel", "spacing_cv", "line_slope_mean", "line_slope_sd", "line_slope_trend",
         "left_start_sd", "left_start_trend", "right_end_sd", "line_len_cv",
         "block_cx", "block_cy", "block_w", "block_h"]
FRAMING = ["ruled_h", "ruled_v", "has_ruled", "line_left", "line_right", "line_top", "line_bottom", "line_dist",
           "line_ink_share", "block_cx", "block_cy", "block_w", "block_h"]
BEHAVIOUR = ["skew_deg", "n_lines", "spacing_rel", "spacing_cv", "line_slope_mean", "line_slope_sd", "line_slope_trend",
             "left_start_sd", "left_start_trend", "right_end_sd", "line_len_cv"]
LAYOUT = FRAMING + BEHAVIOUR


def load(rel):
    with Image.open(ROOT / "dataset" / rel) as im:
        g = im.convert("L")
        g = g.resize((W, round(W * g.height / g.width)), Image.Resampling.LANCZOS)
    return np.asarray(g, np.float32)


def ink_of(a):
    shade = ndi.gaussian_filter(a, 17 * W / 768)
    ink = (shade - a > 17) & (shade > 120) & (a < 205)
    b = 8; ink[:b] = ink[-b:] = False; ink[:, :b] = ink[:, -b:] = False
    ink = ndi.binary_opening(ink, np.ones((2, 2)))
    return remove_small_objects(ink, 12), shade


def ruled_lines(a):
    """Printed margin / frame lines of the sheet: long straight dark ridges (Hough on a DoG ridge map), which are
    often too faint for the ink rule. Horizontal: >= 30% of width; vertical: >= 30% of height; within 10 degrees."""
    H, Wd = a.shape
    R = ndi.gaussian_filter(a, 5) - ndi.gaussian_filter(a, 1)
    segs = cv2.HoughLinesP((R > 3).astype(np.uint8) * 255, 1, np.pi / 720, threshold=200,
                           minLineLength=int(0.30 * Wd), maxLineGap=12)
    mask = np.zeros((H, Wd), np.uint8); info = {"h": [], "v": []}
    for x1, y1, x2, y2 in ([] if segs is None else segs.reshape(-1, 4)):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) < 10:
            info["h"].append((ang, (y1 + y2) / 2))
        elif abs(abs(ang) - 90) < 10 and abs(y2 - y1) >= 0.30 * H:
            info["v"].append((np.sign(ang) * (abs(ang) - 90) if ang > 0 else ang + 90, (x1 + x2) / 2))
        else:
            continue
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, 9)
    return mask.astype(bool), info


def skew_of(ys, xs):
    best, bv = 0.0, -1
    for t in ANGLES:
        r = np.radians(t); yr = ys * np.cos(r) - xs * np.sin(r)
        h = np.bincount((yr - yr.min()).astype(int)); v = h.var()
        if v > bv:
            best, bv = t, v
    return best


def line_measures(ys, xs, skew):
    r = np.radians(skew); yr = ys * np.cos(r) - xs * np.sin(r); xr = ys * np.sin(r) + xs * np.cos(r)
    y0 = yr.min(); prof = np.bincount((yr - y0).astype(int)).astype(float)
    prof = ndi.gaussian_filter1d(prof, 8)
    pk, _ = find_peaks(prof, distance=35, prominence=0.15 * prof.max())
    if len(pk) < 3:
        return dict.fromkeys(["n_lines", "spacing_rel", "spacing_cv", "line_slope_mean", "line_slope_sd",
                              "line_slope_trend", "left_start_sd", "left_start_trend", "right_end_sd",
                              "line_len_cv"], np.nan)
    cen = pk + y0; sp = np.diff(cen); half = np.median(sp) / 2
    idx = np.argmin(np.abs(yr[:, None] - cen[None]), 1)
    ok = np.abs(yr - cen[idx]) < half
    starts, ends, slopes, keep = [], [], [], []
    for j in range(len(cen)):
        m = ok & (idx == j)
        if m.sum() < 200:
            continue
        x = xr[m]; y = yr[m]; s, e = np.percentile(x, [1, 99])
        starts.append(s); ends.append(e); keep.append(j)
        slopes.append(np.degrees(np.arctan(np.polyfit(x, y, 1)[0])))
    starts, ends, slopes = map(np.array, (starts, ends, slopes))
    full = (ends - starts) > 0.5 * np.median(ends - starts)                     # drop short last lines for margins
    bw = np.percentile(xr, 99) - np.percentile(xr, 1); li = np.arange(len(starts))
    trend = lambda v, i: np.polyfit(i, v, 1)[0] if len(v) >= 3 else np.nan
    return dict(n_lines=len(cen), spacing_rel=np.median(sp) / bw, spacing_cv=sp.std() / sp.mean(),
                line_slope_mean=slopes.mean(), line_slope_sd=slopes.std(), line_slope_trend=trend(slopes, li),
                left_start_sd=starts[full].std() / bw, left_start_trend=trend(starts[full] / bw, li[full]),
                right_end_sd=ends[full[:len(ends)]].std() / bw if full.sum() > 1 else np.nan,
                line_len_cv=(ends - starts)[full].std() / (ends - starts)[full].mean())


def hog_of(a, box):
    """HOG + GLBP-HOG part exactly as extract_prior_baselines.texture_baselines (HOG only)."""
    blk = a[box[0]:box[1], box[2]:box[3]]
    b = (blk - blk.mean()) / (blk.std() + 1e-6)
    img = Image.fromarray(np.clip(128 + 40 * b, 0, 255).astype(np.uint8))
    small = np.asarray(img.resize((512, 256), Image.Resampling.BILINEAR), float) / 255
    return hog(small, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), feature_vector=True)


def text_box(ink):
    rows = np.flatnonzero(ink.mean(1) > 0.01); cols = np.flatnonzero(ink.mean(0) > 0.005)
    r0, r1 = np.percentile(rows, [0.5, 99.5]).astype(int); c0, c1 = np.percentile(cols, [0.5, 99.5]).astype(int)
    p = 10
    return max(0, r0 - p), min(ink.shape[0], r1 + p), max(0, c0 - p), min(ink.shape[1], c1 + p)


def job(rel):
    a = load(rel); ink, shade = ink_of(a); H, Wd = a.shape
    lines, info = ruled_lines(a)
    a_nl = np.where(lines, shade, a)                                             # paint ruled lines out
    text = ink & ~lines
    ys, xs = np.nonzero(text); ys = ys.astype(float); xs = xs.astype(float)
    skew = skew_of(ys, xs)
    bx = text_box(text)
    hv = [x for _, x in info["v"]]; hh = [y for _, y in info["h"]]
    rot = [g for g, _ in info["h"]] + [g for g, _ in info["v"]]
    f = dict(ruled_h=len(info["h"]), ruled_v=len(info["v"]), has_ruled=float(len(rot) > 0),
             line_left=float(any(x < bx[2] + 0.02 * Wd for x in hv)), line_right=float(any(x > bx[3] - 0.02 * Wd for x in hv)),
             line_top=float(any(y < bx[0] + 0.02 * H for y in hh)), line_bottom=float(any(y > bx[1] - 0.02 * H for y in hh)),
             line_dist=min([abs(x - bx[2]) for x in hv] + [abs(x - bx[3]) for x in hv], default=Wd) / Wd,
             page_rot_deg=np.median(rot) if rot else np.nan, skew_deg=skew)
    f["skew_rel_page"] = skew - f["page_rot_deg"] if rot else np.nan
    f.update(line_measures(ys, xs, skew))
    f.update(block_cx=(bx[2] + bx[3]) / 2 / Wd, block_cy=(bx[0] + bx[1]) / 2 / H,
             block_w=(bx[3] - bx[2]) / Wd, block_h=(bx[1] - bx[0]) / H)
    hog_noline = hog_of(a_nl, bx)
    M = cv2.getRotationMatrix2D(((bx[2] + bx[3]) / 2, (bx[0] + bx[1]) / 2), skew, 1.0)  # +skew undoes the tilt
    a_r = cv2.warpAffine(a_nl, M, (Wd, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ink_r, _ = ink_of(a_r)
    hog_reg = hog_of(a_r, text_box(ink_r))
    # HC-183 (run_crop_evaluation.ink_features on the 768-px ink of the whole photo) and two controls
    with Image.open(ROOT / "dataset" / rel) as im:
        g = im.convert("L").copy()
    ink768 = image_to_ink(g) > 0
    lm = np.asarray(Image.fromarray(lines.astype(np.uint8) * 255).resize(ink768.shape[::-1], Image.Resampling.BILINEAR)) > 0
    f["line_ink_share"] = float((ink768 & lm).sum() / max(1, ink768.sum()))     # share of HC-183 ink on ruled lines
    hc_nl = ink768 & ~lm
    s = ink768.shape[1] / Wd
    r0, r1, c0, c1 = (int(round(v * s)) for v in bx)
    hc = dict(hc183_check=ink_features(ink768.astype(np.float32)),
              hc183_noline=ink_features(hc_nl.astype(np.float32)),
              hc183_textblock=ink_features(hc_nl[r0:r1, c0:c1].astype(np.float32)))
    return np.array([f[n] for n in NAMES], float), hog_noline, hog_reg, hc


def gtest(v, y, rng, n=10000):
    def g(v, y):
        a, b = v[y == 1], v[y == 0]
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(v) - 2))
        return (a.mean() - b.mean()) / sp
    obs = g(v, y); null = np.array([g(v, rng.permutation(y)) for _ in range(n)])
    J = 1 - 3 / (4 * len(v) - 9)
    return obs * J, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1), v[y == 1].mean(), v[y == 0].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/final42_2026-10-19")
    ap.add_argument("--dev", default="results/ijdar_dev_2026-11-12")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args(); dev = ROOT / a.dev
    import run_final_experiments as rf
    import run_pitfall_analyses as pf
    import run_posthoc_confound_analyses as pc
    D = rf.Data(ROOT / a.base)
    fpath = dev / "features/layout.npz"
    HC = ["hc183_check", "hc183_noline", "hc183_textblock"]
    if a.reuse and fpath.exists():
        z = np.load(fpath, allow_pickle=True)
        L, Hn, Hr = z["layout"], z["hog_noline"], z["hog_registered"]; hcs = {k: z[k] for k in HC}
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            res = list(ex.map(job, D.meta.relative_path, chunksize=2))
        L = np.array([r[0] for r in res]); Hn = np.array([r[1] for r in res]); Hr = np.array([r[2] for r in res])
        hcs = {k: np.array([r[3][k] for r in res]) for k in HC}
        np.savez_compressed(fpath, image_id=D.meta.image_id.to_numpy(), layout=L, names=np.array(NAMES),
                            hog_noline=Hn, hog_registered=Hr, **hcs)
    dmax = np.abs(hcs["hc183_check"] - D.X["hc183"]).max()
    print("HC-183 recomputation matches stored features: max abs diff", dmax, flush=True)
    df = pd.DataFrame(L, columns=NAMES).assign(writer=D.w, label=D.y, para=D.para, image_id=D.meta.image_id.values)
    df.to_csv(dev / "layout_images.csv", index=False)

    print("image-level means by group (1 = left):", flush=True)
    print(df.groupby("label")[FRAMING[:9]].mean().round(3).to_string(), flush=True)

    # writer-level group tests (paragraphs 1-6), BH within framing and within behaviour
    rng = np.random.default_rng(2031); rows = []
    wm = df[df.para <= 6].groupby("writer")[NAMES + ["label"]].mean()
    for fam, names in (("framing", FRAMING + ["page_rot_deg"]), ("behaviour", BEHAVIOUR + ["skew_rel_page"])):
        for n in names:
            v = wm[n].to_numpy(); m = np.isfinite(v)
            if m.sum() < 10 or np.nanstd(v) == 0:
                continue
            g, pv, ml, mr = gtest(v[m], wm.label.to_numpy()[m].astype(int), rng)
            rows.append(dict(family=fam, measure=n, n_writers=int(m.sum()), left_mean=ml, right_mean=mr, hedges_g=g, p_perm=pv))
    T = pd.DataFrame(rows)
    T["q_bh"] = T.groupby("family").p_perm.transform(lambda p: multipletests(p, method="fdr_bh")[1])
    T.to_csv(dev / "layout_tests.csv", index=False); print(T.round(3).to_string(), flush=True)

    # classification
    imp = lambda X: np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    fr = imp(df[FRAMING].to_numpy()); be = imp(df[BEHAVIOUR].to_numpy())
    z = np.load(dev / "features/baselines.npz", allow_pickle=True)
    D.X["hog_spatial"] = z["hog_glbp"][:, :3780]
    D.X["hog_noline"] = Hn; D.X["hog_registered"] = Hr
    D.X["hog_registered_pooled"] = Hr.reshape(len(Hr), 7, 15, 2, 2, 9).mean(axis=(1, 2)).reshape(len(Hr), -1)
    D.X["hc183_noline"] = hcs["hc183_noline"]; D.X["hc183_textblock"] = hcs["hc183_textblock"]
    D.X["framing"] = fr; D.X["behaviour"] = be
    reps = ["hc183", "hc183_noline", "hc183_textblock", "hog_spatial", "hog_noline", "hog_registered",
            "hog_registered_pooled", "framing", "behaviour"]
    P = Parallel(8)
    par = pd.concat(P(delayed(rf.run_cv)(D, r, "svm", s, ("A",)) for r in reps for s in rf.SEEDS), ignore_index=True)
    S = rf.summarize(rf.metrics(rf.writer_level(par)))
    S.to_csv(dev / "layout_cv_summary.csv", index=False)
    print(S[["rep", "ba_mean", "ba_sd", "auc_mean"]].round(3).to_string(), flush=True)
    cov0 = np.column_stack([D.desc.ink_density.to_numpy(), D.X["global_intensity"]])
    extra = []
    for r in ["hc183", "hc183_noline", "hc183_textblock", "hog_spatial", "hog_registered"]:
        for nm, cov in (("ink+photo", cov0), ("framing", fr), ("behaviour", be), ("ink+photo+framing", np.column_stack([cov0, fr]))):
            b = P(delayed(pc.cv_A)(D.X[r], D.y, D.w, D.para, s, None, cov) for s in rf.SEEDS)
            extra.append(dict(rep=r, covariates=nm, adjusted_ba=float(np.mean(b))))
            print(extra[-1], flush=True)
    for r in reps:
        extra.append(dict(rep=r, covariates="writer_id_top1", adjusted_ba=pf.writer_id_top1(D.X[r], D.w, D.para)))
    E = pd.DataFrame(extra); E.to_csv(dev / "layout_cv_adjusted.csv", index=False)
    print(E.round(3).to_string(), flush=True)

    lab = pd.Series(D.wlabels, D.writers); rng = np.random.default_rng(2032)
    perms = [pd.Series(rng.permutation(lab.to_numpy()), lab.index) for _ in range(a.n_perm)]
    prow = []
    for r in ("hc183_textblock", "hog_registered", "framing", "behaviour"):
        obs = rf.perm_stat(D, r, lab)[0]
        null = np.array(P(delayed(rf.perm_stat)(D, r, pl) for pl in perms))[:, 0]
        prow.append(dict(rep=r, observed=obs, p=(np.sum(null >= obs) + 1) / (a.n_perm + 1)))
        print(prow[-1], flush=True)
        pd.DataFrame(prow).to_csv(dev / "layout_permutation.csv", index=False)


if __name__ == "__main__":
    main()
