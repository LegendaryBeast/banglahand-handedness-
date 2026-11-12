#!/usr/bin/env python3
"""MATRA-INK: ink-deposition features of the Bangla headstroke (matra), normalised within each image.

Idea. Most Bangla words carry a long horizontal headstroke (matra), usually drawn left to right. A right-handed
writer pulls the pen along it, a left-handed writer typically pushes it. Ballpoint ink is laid down differently
under pushing and pulling, and the start and end of a stroke differ (pen-down blob, pen-up taper). Each paragraph
has dozens of matras, so they form a script-specific probe for handedness.

Design rules (to keep photographic shortcuts out):
  * darkness is relative to the local paper level, r = (B - I) / B, so overall brightness cancels;
  * every feature is a ratio or a shape measured *within* the image: along-stroke and cross-section profiles are
    divided by the stroke's own mean, matra darkness and width are divided by those of vertical stems of the same
    image, lengths are in units of the writer's stroke width. No absolute grey level, ink amount or framing enters;
  * mirroring the image left-right reverses the along-stroke bins, flips the sign of the left-right asymmetries and
    the matra angle, and leaves the cross-section profile unchanged (tested in tests/test_matra_features.py).

Detection. Ink mask as in run_pseudo_pressure.py (full resolution). Horizontal runs are kept by a morphological
opening with a horizontal line of length 5w (w = stroke width). A run is a matra if it is long (>= 6w), thin,
and has clearly more ink hanging below it than above it (letters hang from the matra; vowel signs above are small).
Columns where a stem or vowel sign joins the matra (junctions) are excluded from the darkness and width profiles.

Output per image: a fixed-length vector (NAMES), the length-weighted mean over the image's matras.
"""
import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", message="Mean of empty slice")
N_ALONG = 8
U = np.linspace(-1.5, 1.5, 13)           # cross-section sample positions, in half-widths from the centreline
NAMES = (["ratio_dark_matra_stem", "ratio_width_matra_stem"]
         + [f"along_dark_{i}" for i in range(N_ALONG)] + [f"along_width_{i}" for i in range(N_ALONG)]
         + ["lr_dark_asym", "lr_width_asym", "overhang_left", "overhang_right", "overhang_asym",
            "texture_cv", "skip_fraction", "angle_mean_deg", "angle_sd_deg"]
         + [f"cross_{i:02d}" for i in range(len(U))]
         + ["cross_asym", "length_mean_w", "matra_per_kpx"])
MIRROR_FLIP_SIGN = {"lr_dark_asym", "lr_width_asym", "overhang_asym", "angle_mean_deg"}


def ink_and_darkness(g):
    """g: float32 grey image (full resolution). Returns ink mask, relative darkness, stroke width, skeleton."""
    s = g.shape[1] / 1536
    bg = cv2.GaussianBlur(g, (0, 0), 34 * s)
    ink = ((bg - g) > 17) & (bg > 120) & (g < 205)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 12 * s * s
    ink = keep[lab]
    b = int(8 * s); ink[:b] = ink[-b:] = False; ink[:, :b] = ink[:, -b:] = False
    r = np.clip((bg - g) / np.maximum(bg, 1), 0, 1).astype(np.float32)
    edt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(ink)
    w = float(2 * edt[sk].mean())
    return ink, r, w, sk


def detect_matras(ink, w):
    """Return a list of (x0, x1, yc[x0:x1+1], thick[x0:x1+1]) for accepted matra runs."""
    L = max(15, 2 * int(round(2.5 * w)) + 1)                   # odd length: centred anchor, mirror-symmetric
    horiz = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, L), np.uint8)).astype(bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats(horiz.astype(np.uint8), connectivity=8)
    out = []
    H = ink.shape[0]
    band = int(round(4 * w))
    for k in range(1, n):
        x, y, bw, bh = st[k, :4]
        if bw < 6 * w or bw > 40 * w or bh > 2.5 * w + 0.09 * bw:   # long and thin (~5 deg tilt); not a ruled line
            continue
        m = lab[y:y + bh, x:x + bw] == k
        cols = m.any(0)
        if cols.mean() < 0.9:
            continue
        yy = np.arange(bh)[:, None]
        cnt = m.sum(0).astype(float)
        yc = (m * yy).sum(0) / np.maximum(cnt, 1) + y
        top, bot = int(np.floor(yc.min() - cnt.max() / 2)), int(np.ceil(yc.max() + cnt.max() / 2))
        above = ink[max(0, top - band):max(0, top - 1), x:x + bw].mean() if top - 1 > 0 else 0.0
        below = ink[min(H, bot + 1):min(H, bot + band), x:x + bw].mean() if bot + 1 < H else 0.0
        if not (below >= 0.06 and below >= 1.5 * above):       # letters hang below the headstroke
            continue
        out.append((x, x + bw - 1, yc, cnt))
    return out


def _sample(img, ys, xs):
    return ndi.map_coordinates(img, [ys, xs], order=1, mode="nearest")


def matra_measures(ink, r, w, m):
    """Measures of one matra, or None if too few free (non-junction) columns."""
    x0, x1, yc, cnt = m
    xs = np.arange(x0, x1 + 1)
    fit = np.polyfit(xs, yc, 1)
    yl = np.polyval(fit, xs)                                   # straight centreline (robust to junction bulges)
    H = ink.shape[0]
    half = np.median(cnt) / 2
    # junctions: ink continues beyond the stroke edge by more than 1.2 w above or below the centreline
    reach = int(round(half + 1.2 * w))
    junc = np.zeros(len(xs), bool)
    for side in (-1, 1):
        yy = np.clip(np.round(yl + side * reach).astype(int), 0, H - 1)
        junc |= ink[yy, xs]
    jd = ndi.binary_dilation(junc, np.ones(2 * int(round(w)) + 1, bool))
    free = ~jd
    if free.sum() < 3 * w:
        return None
    # width: vertical ink run through the centreline; darkness: centreline (max over +-1 px)
    width = np.array([_run_length(ink[:, x], int(round(y))) for x, y in zip(xs, yl)], float)
    dark = np.max([_sample(r, yl + d, xs) for d in (-1, 0, 1)], axis=0)
    cross = np.stack([_sample(r, yl + u * half, xs) for u in U])            # (len(U), n_cols)
    f = free
    pos = (xs - x0) / max(1, x1 - x0)
    bins = np.minimum((pos * N_ALONG).astype(int), N_ALONG - 1)
    dm, wm = dark[f].mean(), width[f].mean()
    ad = np.array([dark[f & (bins == b)].mean() / dm if (f & (bins == b)).any() else np.nan for b in range(N_ALONG)])
    aw = np.array([width[f & (bins == b)].mean() / wm if (f & (bins == b)).any() else np.nan for b in range(N_ALONG)])
    lr = lambda a: (np.nanmean(a[:2]) - np.nanmean(a[-2:])) / (np.nanmean(a[:2]) + np.nanmean(a[-2:]))
    jx = np.flatnonzero(junc)
    if len(jx):
        ol, orr = jx[0] / w, (len(xs) - 1 - jx[-1]) / w
    else:
        ol = orr = np.nan
    d_f = dark[f]; t = xs[f]
    resid = d_f - np.polyval(np.polyfit(t, d_f, 1), t) if len(t) > 2 else d_f - d_f.mean()
    cp = cross[:, f].mean(1)
    cp = cp / max(cp[len(U) // 2], 1e-6)
    upper, lower = cp[U < 0].mean(), cp[U > 0].mean()
    return dict(length=len(xs), dark=dm, width=wm, along_dark=ad, along_width=aw,
                lr_dark=lr(ad), lr_width=lr(aw), overhang_left=ol, overhang_right=orr,
                overhang_asym=(orr - ol) / (orr + ol) if np.isfinite(ol) and (orr + ol) > 0 else np.nan,
                texture_cv=float(resid.std() / max(dm, 1e-6)), skip=float((d_f < 0.6 * np.median(d_f)).mean()),
                angle=float(-np.degrees(np.arctan(fit[0]))), cross=cp,
                cross_asym=float((upper - lower) / (upper + lower)))


def _run_length(col, y):
    if y < 0 or y >= len(col) or not col[y]:
        return 0
    a = y
    while a > 0 and col[a - 1]:
        a -= 1
    b = y
    while b < len(col) - 1 and col[b + 1]:
        b += 1
    return b - a + 1


def stem_reference(ink, r, w):
    """Median centreline darkness and horizontal width of near-vertical stems of the same image."""
    L = max(15, 2 * int(round(2 * w)) + 1)
    vert = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(vert, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = st[1:, cv2.CC_STAT_HEIGHT] <= 12 * w     # drop ruled margin lines
    sk = skeletonize(keep[lab])
    ys, xs = np.nonzero(sk)
    if len(ys) < 50:
        return np.nan, np.nan
    sel = np.random.default_rng(0).choice(len(ys), min(4000, len(ys)), replace=False)
    ys, xs = ys[sel], xs[sel]
    dark = np.max([r[ys, np.clip(xs + d, 0, r.shape[1] - 1)] for d in (-1, 0, 1)], axis=0)
    width = np.array([_run_length(ink[y], x) for y, x in zip(ys, xs)], float)
    return float(np.median(dark)), float(np.median(width))


def features_from_gray(g, mirror=False, return_debug=False):
    g = np.asarray(g, np.float32)
    if mirror:
        g = np.ascontiguousarray(g[:, ::-1])
    ink, r, w, sk = ink_and_darkness(g)
    matras = detect_matras(ink, w)
    ms = [x for x in (matra_measures(ink, r, w, m) for m in matras) if x is not None]
    if not ms:
        raise ValueError("no matra found")
    L = np.array([m["length"] for m in ms], float)
    wavg = lambda v: float(np.nansum(np.asarray(v, float) * L) / np.sum(L[np.isfinite(np.asarray(v, float))]))
    wavg_vec = lambda key: np.array([wavg([m[key][i] for m in ms]) for i in range(len(ms[0][key]))])
    sd, sw = stem_reference(ink, r, w)
    ang = np.array([m["angle"] for m in ms])
    v = np.concatenate([
        [wavg([m["dark"] for m in ms]) / sd, wavg([m["width"] for m in ms]) / sw],
        wavg_vec("along_dark"), wavg_vec("along_width"),
        [wavg([m["lr_dark"] for m in ms]), wavg([m["lr_width"] for m in ms]),
         wavg([m["overhang_left"] for m in ms]), wavg([m["overhang_right"] for m in ms]),
         wavg([m["overhang_asym"] for m in ms]), wavg([m["texture_cv"] for m in ms]),
         wavg([m["skip"] for m in ms]), wavg(ang),
         float(np.sqrt(np.sum(L * (ang - wavg(ang)) ** 2) / L.sum()))],
        wavg_vec("cross"),
        [wavg([m["cross_asym"] for m in ms]), float(np.mean(L) / w), 1000.0 * len(ms) / max(1, sk.sum())],
    ]).astype(np.float64)
    assert v.shape == (len(NAMES),)
    if return_debug:
        return v, dict(ink=ink, w=w, matras=matras, n_used=len(ms))
    return v


def features_from_path(path, mirror=False):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return features_from_gray(g, mirror=mirror)


def _job(args):
    rel, mirror = args
    cv2.setNumThreads(1)                                         # avoid thread oversubscription across workers
    try:
        return features_from_path(ROOT / "dataset" / rel, mirror), ""
    except Exception as e:                                       # recorded, not hidden
        return np.full(len(NAMES), np.nan), repr(e)


def main():
    import json
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="results folder; writes matra_features.npz there")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = ROOT / a.output; out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for mirror in (False, True):
            rows = list(ex.map(_job, [(p, mirror) for p in meta.relative_path], chunksize=2))
            res["matra_mirror" if mirror else "matra"] = np.array([x for x, _ in rows])
            errs = [(i, e) for i, (_, e) in zip(meta.image_id, rows) if e]
            print(("mirror" if mirror else "original"), "errors:", errs[:5], len(errs), flush=True)
    np.savez_compressed(out / "matra_features.npz", image_id=meta.image_id.to_numpy(), names=np.array(NAMES), **res)
    (out / "matra_provenance.json").write_text(json.dumps(dict(script="scripts/matra_features.py", names=NAMES,
                                                               n_images=len(meta)), indent=2))
    print("wrote", out / "matra_features.npz")


if __name__ == "__main__":
    sys.exit(main())
