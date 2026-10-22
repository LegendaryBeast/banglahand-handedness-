#!/usr/bin/env python3
"""Secondary analyses of the 37-writer run (declared in protocol_declared_before_fitting.json).

Writes: pitfalls.csv, adjusted.csv, controls_metrics_summary.csv, controls_contrasts_by_split.csv,
controls_sensitivity.csv, single_measures.csv, mirror_test.csv (+ *_by_split.csv where useful).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_experiments as rf  # noqa: E402
import run_pitfall_analyses as pf  # noqa: E402
import run_posthoc_confound_analyses as pc  # noqa: E402

CTRL = ["global_intensity", "background"]


def mirror_cv(D, seed):
    """STROKE SVM fitted on original A images, applied to original and mirrored test images."""
    X, Xm = D.X["stroke"], D.X["stroke_mirror"]
    sh = D.para <= 6
    fold = dict(zip(D.writers, rf.writer_folds(D.writers, D.wlabels, seed)))
    fw = np.array([fold[u] for u in D.w])
    so, sm = np.full(len(D.w), np.nan), np.full(len(D.w), np.nan)
    for f in range(5):
        tr, te = sh & (fw != f), sh & (fw == f)
        m = rf.estimator("svm"); m.fit(X[tr], D.y[tr])
        so[te] = m.decision_function(X[te]); sm[te] = m.decision_function(Xm[te])
    d = pd.DataFrame(dict(w=D.w[sh], y=D.y[sh], so=so[sh], sm=sm[sh])).groupby("w").mean()
    po, pm = (d.so >= 0).astype(int), (d.sm >= 0).astype(int)
    return dict(seed=seed, ba_original=balanced_accuracy_score(d.y, po), ba_mirrored=balanced_accuracy_score(d.y, pm),
                flip_fraction=float((po != pm).mean()), mean_margin_shift_left=float((d.sm - d.so)[d.y == 1].mean()),
                mean_margin_shift_right=float((d.sm - d.so)[d.y == 0].mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    run = ROOT / a.run
    D = rf.Data(run)
    P = Parallel(a.jobs)
    reps = rf.MAIN_REPS + (rf.STROKE_REPS if "stroke" in D.X else [])
    seeds = rf.SEEDS
    ex = pd.read_csv(run / "features/excluded_images.csv")
    XE = dict(np.load(run / "features/features_excluded.npz"))
    if "stroke" in XE:
        XE["combined"] = np.hstack([XE["hc183"], XE["style6"], XE["stroke"]])

    # Pitfalls
    rows = []
    for r in reps + CTRL:
        X40 = np.vstack([D.X[r], XE[r]]); y40 = np.concatenate([D.y, ex.label]); w40 = np.concatenate([D.w, ex.writer_id])
        p40 = np.concatenate([D.para, ex.paragraph_number])
        r37 = P(delayed(pf.svm_A_writer_ba)(D.X[r], D.y, D.w, D.para, s) for s in seeds)
        r40 = P(delayed(pf.svm_A_writer_ba)(X40, y40, w40, p40, s) for s in seeds)
        lk = P(delayed(pf.leaky_ba)(D.X[r], D.y, D.para, s) for s in seeds)
        rows.append(dict(rep=r, writer_ba=np.mean([x[0] for x in r37]), writer_ba_40folders=np.mean([x[0] for x in r40]),
                         para_ba_writer_split=np.mean([x[1] for x in r37]), para_ba_leaky=np.mean(lk),
                         writer_id_top1=pf.writer_id_top1(D.X[r], D.w, D.para)))
        print("pitfall", r, flush=True)
    pd.DataFrame(rows).to_csv(run / "pitfalls.csv", index=False)

    # Adjustment for ink density + photo intensity (fold-wise residualisation)
    cov = np.column_stack([D.desc.ink_density.to_numpy(), D.X["global_intensity"]])
    rows = []
    for r in reps:
        b = P(delayed(pc.cv_A)(D.X[r], D.y, D.w, D.para, s, None, cov) for s in seeds)
        rows.append(dict(rep=r, subset="all", adjusted_mean=np.mean(b), adjusted_sd=np.std(b, ddof=1)))
    pd.DataFrame(rows).to_csv(run / "adjusted.csv", index=False)
    print("adjusted done", flush=True)

    # Controls on every condition and their sensitivity
    par = pd.concat(P(delayed(rf.run_cv)(D, r, c, s) for r in CTRL for c in rf.CLFS for s in seeds), ignore_index=True)
    m = rf.metrics(rf.writer_level(par))
    rf.summarize(m).to_csv(run / "controls_metrics_summary.csv", index=False)
    piv = m.pivot_table(index=["rep", "clf", "seed"], columns="analysis", values="ba")
    pd.DataFrame(dict(transfer_C_minus_D=piv["C"] - piv[["D_S1", "D_S2", "D_S3"]].mean(axis=1),
                      varied_B_minus_E=piv["B"] - piv[["E_S1", "E_S2", "E_S3"]].mean(axis=1))).reset_index() \
        .to_csv(run / "controls_contrasts_by_split.csv", index=False)
    rows = []
    for r in CTRL:
        o2 = pd.concat(P(delayed(rf.run_cv)(D, r, "svm", s, analyses=("A",), image_ok=~D.trunc) for s in seeds))
        b2 = rf.metrics(rf.writer_level(o2)).ba
        rows.append(dict(rep=r, S2=b2.mean()))
    pd.DataFrame(rows).to_csv(run / "controls_sensitivity.csv", index=False)
    print("controls done", flush=True)

    # Single interpretable measures
    rows = []
    for j, c in enumerate(rf.STYLE6):
        b = P(delayed(pc.cv_A)(D.X["style6"][:, [j]], D.y, D.w, D.para, s) for s in seeds)
        rows.append(dict(measure=c, mean=np.mean(b), sd=np.std(b, ddof=1)))
    pd.DataFrame(rows).to_csv(run / "single_measures.csv", index=False)

    # Mirror test
    if "stroke" in D.X:
        mt = pd.DataFrame(P(delayed(mirror_cv)(D, s) for s in seeds))
        mt.to_csv(run / "mirror_test_by_split.csv", index=False)
        mt.drop(columns="seed").agg(["mean", "std"]).to_csv(run / "mirror_test.csv")
        print(mt.drop(columns="seed").mean().round(3).to_string())
    print("secondary analyses done")


if __name__ == "__main__":
    main()
