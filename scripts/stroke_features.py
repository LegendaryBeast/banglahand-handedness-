#!/usr/bin/env python3
"""Writing-only stroke features: direction, hinge, curvature, stroke endings and slant.

Design rules (to keep photographic shortcuts out):
  * only a binary ink mask is used - no grey levels, no paper pixels;
  * the mask is cropped to the text block - framing and margins do not enter;
  * every histogram is normalised to sum 1 - ink amount does not enter;
  * scale-dependent steps are set from the writer's own stroke width.
Contours are traced with a fixed orientation and directions are kept over the full
circle, so mirror-image (left/right) asymmetries of strokes are preserved, unlike
the unsigned gradients of HC-183.  Static images still cannot show the order or
direction in which a stroke was drawn.
"""
from collections import deque

import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from skimage.measure import find_contours
from skimage.morphology import remove_small_objects, skeletonize

WIDTH = 1536
TWO_PI = 2 * np.pi
NAMES = (["dir16_%02d" % i for i in range(16)] + ["hinge_%02d_%02d" % (i, j) for i in range(12) for j in range(12)]
         + ["turn12_%02d" % i for i in range(12)] + ["end12_%02d" % i for i in range(12)]
         + ["orient12_%02d" % i for i in range(12)] + ["slant9_%02d" % i for i in range(9)]
         + ["endpoints_per_len", "junctions_per_len", "holes_per_len"])


def ink_mask(gray, return_box=False):
    """Binary ink of the text block at WIDTH px (local-contrast rule of HC-183, scaled)."""
    w, h = gray.size
    a = np.asarray(gray.resize((WIDTH, round(WIDTH * h / w)), Image.Resampling.LANCZOS), np.float32)
    shade = ndi.gaussian_filter(a, 17 * WIDTH / 768)
    ink = (shade - a > 17) & (shade > 120) & (a < 205)
    b = 8
    ink[:b] = ink[-b:] = False; ink[:, :b] = ink[:, -b:] = False
    ink = ndi.binary_opening(ink, np.ones((2, 2)))
    ink = remove_small_objects(ink, 12)
    rows = np.flatnonzero(ink.mean(1) > 0.01); cols = np.flatnonzero(ink.mean(0) > 0.005)
    r0, r1 = np.percentile(rows, [0.5, 99.5]).astype(int); c0, c1 = np.percentile(cols, [0.5, 99.5]).astype(int)
    pad = 10
    box = (max(0, r0 - pad), r1 + pad, max(0, c0 - pad), c1 + pad)     # rows, cols at WIDTH px
    crop = ink[box[0]:box[1], box[2]:box[3]]
    return (crop, box) if return_box else crop


def stroke_width(ink):
    sk = skeletonize(ink)
    return float(2 * ndi.distance_transform_edt(ink)[sk].mean()), sk


def _hist(v, bins, lo, hi, w=None):
    h = np.histogram(v, bins=bins, range=(lo, hi), weights=w)[0].astype(float)
    return h / (h.sum() + 1e-12)


def _circ(v, bins, period=TWO_PI):
    """Circular histogram with bins centred on 0 (so horizontal/vertical are bin centres, not edges)."""
    return _hist(np.mod(np.asarray(v) + period / (2 * bins), period), bins, 0, period)


def contour_features(ink, L):
    d1, hg, tn = [], [], []
    for c in find_contours(np.pad(ink, 1).astype(float), 0.5, positive_orientation="high"):
        seg = np.diff(c, axis=0); s = np.concatenate([[0], np.cumsum(np.hypot(*seg.T))])
        if s[-1] < 4 * L:
            continue
        t = np.arange(0, s[-1], 2.0)
        p = np.column_stack([np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])])
        k = max(1, int(round(L / 2.0)))
        n = len(p)
        idx = np.arange(n); fwd = p[(idx + k) % n] - p; bwd = p[(idx - k) % n] - p
        ang = lambda v: np.mod(np.arctan2(-v[:, 0], v[:, 1]), TWO_PI)   # x right, y up
        a_f, a_b = ang(fwd), ang(bwd)
        d1.append(a_f)
        hg.append(np.column_stack([a_f, a_b]))
        tn.append(np.angle(np.exp(1j * (a_f - np.mod(a_b + np.pi, TWO_PI)))))
    d1 = np.concatenate(d1); hg = np.concatenate(hg); tn = np.concatenate(tn)
    off = np.pi / 12
    H = np.histogram2d(np.mod(hg[:, 0] + off, TWO_PI), np.mod(hg[:, 1] + off, TWO_PI), bins=12,
                       range=[[0, TWO_PI], [0, TWO_PI]])[0].ravel()
    return _circ(d1, 16), H / H.sum(), _circ(tn, 12)


def skeleton_features(ink, sk, L):
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - sk
    ends = np.argwhere(sk & (nb == 1)); junc = sk & (nb >= 3)
    length = float(sk.sum())
    dirs = []
    for r, c in ends:
        q, seen, last = deque([(r, c, 0)]), {(r, c)}, (r, c)
        while q:
            y, x, d = q.popleft(); last = (y, x)
            if d >= L or (junc[y, x] and d > 0):
                break
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if (dy or dx) and 0 <= yy < sk.shape[0] and 0 <= xx < sk.shape[1] and sk[yy, xx] and (yy, xx) not in seen:
                        seen.add((yy, xx)); q.append((yy, xx, d + 1))
        if last != (r, c):
            dirs.append(np.mod(np.arctan2(-(r - last[0]), c - last[1]), TWO_PI))   # interior -> endpoint
    f = ndi.gaussian_filter(ink.astype(np.float32), 1.0)
    gy, gx = np.gradient(f)
    theta = np.mod(np.arctan2(-gy, gx) + np.pi / 2, np.pi)[sk]                    # stroke direction, unsigned
    vert = theta[np.abs(theta - np.pi / 2) < np.pi / 4] - np.pi / 2              # signed tilt of near-vertical strokes
    holes = ndi.label(ndi.binary_fill_holes(ink) & ~ink)[1]
    per = 1000.0 / max(length, 1.0)
    return (_circ(np.array(dirs), 12), _circ(theta, 12, np.pi), _hist(vert, 9, -np.pi / 4, np.pi / 4),
            np.array([len(ends) * per, junc.sum() * per, holes * per]))


def stroke_features(gray, mirror=False):
    ink = ink_mask(gray)
    if mirror:
        ink = np.ascontiguousarray(ink[:, ::-1])
    sw, sk = stroke_width(ink)
    L = max(4, int(round(2 * sw)))
    d1, hg, tn = contour_features(ink, L)
    e, o, sl, dens = skeleton_features(ink, sk, L)
    v = np.concatenate([d1, hg, tn, e, o, sl, dens]).astype(np.float32)
    assert v.shape == (len(NAMES),) and np.isfinite(v).all()
    return v, ink, sw


def binary_tiles(ink, sw, target_sw=3.0, tile=224, min_ink=0.02):
    """Black-on-white, stroke-width-normalised tiles for a CNN (no grey levels)."""
    s = target_sw / max(sw, 1e-6)
    im = Image.fromarray(np.uint8(~ink) * 255).resize((max(tile, round(ink.shape[1] * s)), max(tile, round(ink.shape[0] * s))),
                                                      Image.Resampling.BILINEAR)
    a = np.asarray(im, np.float32) / 255.0
    H, W = a.shape
    a = np.pad(a, ((0, (-H) % tile), (0, (-W) % tile)), constant_values=1.0)
    out = [a[r:r + tile, c:c + tile] for r in range(0, a.shape[0], tile) for c in range(0, a.shape[1], tile)]
    out = [t for t in out if (t < 0.5).mean() >= min_ink] or out
    return np.stack(out)
