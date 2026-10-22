#!/usr/bin/env python3
"""Sensitivity analysis S3 (protocol_declared_before_fitting.json): re-include the unreleased raw RH/P3 folder.

Primary SVM / condition A, writer balanced accuracy over the 20 repetitions, 42 released writers versus 42 + raw P3.
Writes s3_sensitivity.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_experiments as rf  # noqa: E402
import run_pitfall_analyses as pf  # noqa: E402

CTRL = ["global_intensity", "background"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    run = ROOT / a.run
    D = rf.Data(run)
    P = Parallel(a.jobs)
    reps = rf.MAIN_REPS + (rf.STROKE_REPS if "stroke" in D.X else [])
    ex = pd.read_csv(run / "features/excluded_images.csv")
    XE = dict(np.load(run / "features/features_excluded.npz"))
    if "stroke" in XE:
        XE["combined"] = np.hstack([XE["hc183"], XE["style6"], XE["stroke"]])
    keep = (ex.writer_id == "R_rawP3").to_numpy()
    rows = []
    for r in reps + CTRL:
        if r not in XE:
            continue
        X = np.vstack([D.X[r], XE[r][keep]]); y = np.concatenate([D.y, ex.label[keep]])
        w = np.concatenate([D.w, ex.writer_id[keep]]); p = np.concatenate([D.para, ex.paragraph_number[keep]])
        r42 = P(delayed(pf.svm_A_writer_ba)(D.X[r], D.y, D.w, D.para, s) for s in rf.SEEDS)
        r43 = P(delayed(pf.svm_A_writer_ba)(X, y, w, p, s) for s in rf.SEEDS)
        rows.append(dict(rep=r, writers_released=len(set(D.w)), writer_ba=np.mean([x[0] for x in r42]),
                         writers_S3=len(set(w)), writer_ba_S3=np.mean([x[0] for x in r43]),
                         writer_ba_S3_sd=np.std([x[0] for x in r43], ddof=1)))
        print("s3", r, rows[-1]["writer_ba"], rows[-1]["writer_ba_S3"], flush=True)
    pd.DataFrame(rows).to_csv(run / "s3_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
