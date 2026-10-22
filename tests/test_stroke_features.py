import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stroke_features as sf  # noqa: E402


def synthetic_page(offset=0):
    """Synthetic handwriting-like strokes (SYNTHETIC test fixture, not research data)."""
    im = Image.new("L", (1600, 900), 170 + offset)
    d = ImageDraw.Draw(im)
    rng = np.random.default_rng(0)
    for line in range(5):
        y = 140 + line * 140
        d.line([(120, y), (1480, y)], fill=60 + offset, width=6)               # headstroke-like line
        for x in range(160, 1450, 70):
            dx = int(rng.integers(-15, 25))
            d.line([(x, y), (x + dx, y + 70)], fill=60 + offset, width=6)       # slanted stems
            d.arc([x - 20, y + 20, x + 25, y + 75], start=30, end=300, fill=60 + offset, width=5)
    return im


def test_shapes_and_normalisation():
    v, ink, sw = sf.stroke_features(synthetic_page())
    assert v.shape == (len(sf.NAMES),) and np.isfinite(v).all()
    for sl in [slice(0, 16), slice(16, 160), slice(160, 172), slice(172, 184), slice(184, 196), slice(196, 205)]:
        assert abs(v[sl].sum() - 1) < 1e-4
    assert 3 < sw < 9


def test_mirror_maps_directions():
    g = synthetic_page()
    v, _, _ = sf.stroke_features(g)
    m, _, _ = sf.stroke_features(g, mirror=True)
    # mirroring a fixed-orientation contour maps direction a -> -a, i.e. bin k -> bin (-k) mod 16
    mapped = v[:16][[(-k) % 16 for k in range(16)]]
    assert np.abs(mapped - m[:16]).sum() < 0.15
    assert np.abs(v[160:172] - m[160:172]).sum() < 0.1        # turning is mirror invariant


def test_brightness_shift_invariance():
    a, _, _ = sf.stroke_features(synthetic_page(0))
    b, _, _ = sf.stroke_features(synthetic_page(20))
    assert np.abs(a - b)[:-3].max() < 1e-6
