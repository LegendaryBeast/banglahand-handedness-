#!/usr/bin/env python3
"""Post hoc (amendment 3): text-line tilt of each photograph (projection-profile sharpness over -6..6 deg).

Checks whether page rotation differs between groups, which would be a chiral acquisition cue that the
mirror test could not separate from stroke direction.
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stroke_features as sf  # noqa: E402
import extract_final_features as ef  # noqa: E402
import run_framing_check as fc  # noqa: E402
import run_final_experiments as rf  # noqa: E402


def tilt(ink):
    small = ndi.zoom(ink.astype(float), 0.5, order=1)
    best, ang = -1, 0.0
    for a in np.arange(-6, 6.01, 0.25):
        r = ndi.rotate(small, a, reshape=False, order=1)
        v = r.sum(1).var()
        if v > best:
            best, ang = v, a
    return ang          # degrees that make text lines horizontal (counter-clockwise positive)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    run = ROOT / ap.parse_args().run
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    rows = [dict(image_id=r.image_id, writer_id=r.writer_id, label=r.label, para=r.paragraph_number,
                 tilt_deg=tilt(sf.ink_mask(ef.load_gray(ROOT / "dataset" / r.relative_path)))) for r in meta.itertuples()]
    t = pd.DataFrame(rows); t.to_csv(run / "tilt_by_image.csv", index=False)
    wm = t[t.para <= 6].groupby("writer_id").agg(label=("label", "first"), tilt=("tilt_deg", "mean"), abs_tilt=("tilt_deg", lambda s: s.abs().mean()))
    out = []
    for sub, ws in [("all", wm.index)]:
        s = wm.loc[list(ws)]; y = s.label.to_numpy().astype(int)
        for c in ["tilt", "abs_tilt"]:
            g, p = fc.g_and_p(s[c].to_numpy(), y, np.random.default_rng(2033))
            out.append(dict(subset=sub, measure=c, left_mean=s[c][y == 1].mean(), right_mean=s[c][y == 0].mean(), hedges_g=g, p_perm=p))
    pd.DataFrame(out).to_csv(run / "tilt_tests.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string())


if __name__ == "__main__":
    main()
