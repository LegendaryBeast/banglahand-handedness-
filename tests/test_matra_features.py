"""Invariance tests of MATRA-INK (scripts/matra_features.py) on a synthetic page of 'words'."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import matra_features as mf  # noqa: E402


def synthetic_page(seed=0):
    """Paper at ~235 with lighting gradient; words = horizontal headstroke + hanging stems; ink darker at the
    left end of each headstroke (an asymmetric 'pen-down' profile) so direction features are non-trivial."""
    rng = np.random.default_rng(seed)
    H, W = 1100, 2048
    yy, xx = np.mgrid[:H, :W]
    paper = 225 + 15 * xx / W
    ink = np.zeros((H, W), np.float32)                          # darkness to subtract
    for line in range(5):
        y = 150 + 190 * line
        x = 120
        while x < W - 300:
            L = int(rng.integers(140, 260))
            prof = np.linspace(150, 110, L)                     # darker at the left end
            for t in range(-4, 5):
                ink[y + t, x:x + L] = np.maximum(ink[y + t, x:x + L], prof * (1 - abs(t) / 6))
            for sx in np.linspace(x + 25, x + L - 25, 3).astype(int):
                cv2.line(ink, (int(sx), y), (int(sx), y + 85), 130, 9)
            x += L + int(rng.integers(50, 90))
    g = np.clip(paper - ink + rng.normal(0, 2, (H, W)), 0, 255).astype(np.float32)
    return g


@pytest.fixture(scope="module")
def page():
    return synthetic_page()


def test_detects_headstrokes(page):
    v, dbg = mf.features_from_gray(page, return_debug=True)
    assert dbg["n_used"] >= 20
    assert np.isfinite(v).all() and v.shape == (len(mf.NAMES),)
    idx = {n: i for i, n in enumerate(mf.NAMES)}
    assert v[idx["lr_dark_asym"]] > 0                            # left end darker, as drawn


def test_mirror_reverses_direction_and_keeps_cross_section(page):
    v = mf.features_from_gray(page)
    vm = mf.features_from_gray(page, mirror=True)
    idx = {n: i for i, n in enumerate(mf.NAMES)}
    for k in range(mf.N_ALONG):
        assert vm[idx[f"along_dark_{k}"]] == pytest.approx(v[idx[f"along_dark_{mf.N_ALONG - 1 - k}"]], abs=0.02)
    for n in mf.MIRROR_FLIP_SIGN - {"overhang_asym"}:
        assert vm[idx[n]] == pytest.approx(-v[idx[n]], abs=0.02)
    assert vm[idx["overhang_left"]] == pytest.approx(v[idx["overhang_right"]], rel=0.1)
    for k in range(len(mf.U)):
        assert vm[idx[f"cross_{k:02d}"]] == pytest.approx(v[idx[f"cross_{k:02d}"]], abs=0.02)


def test_brightness_gain_barely_changes_features(page):
    v = mf.features_from_gray(page)
    v2 = mf.features_from_gray(page * 0.92)                     # darker photograph, same writing
    ang = [mf.NAMES.index(n) for n in ("angle_mean_deg", "angle_sd_deg")]
    other = [i for i in range(len(mf.NAMES)) if i not in ang]
    rel = np.abs(v2 - v)[other] / (np.abs(v)[other] + 0.05)
    assert np.median(rel) < 0.02 and rel.max() < 0.15
    assert np.abs(v2 - v)[ang].max() < 0.1                       # degrees
