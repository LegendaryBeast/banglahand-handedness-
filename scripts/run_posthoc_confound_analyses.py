#!/usr/bin/env python3
"""Post hoc confound analyses declared in protocol_amendment_2.json (exploratory)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_experiments as rf  # noqa: E402


def cv_A(X, y, w, para, seed, writers=None, covariates=None):
    """Fixed linear SVM, analysis A, writer-level BA; optional fold-wise residualisation on covariates."""
    keep = para <= 6
    if writers is not None:
        keep &= np.isin(w, writers)
    wr = np.array(sorted(set(w[keep])))
    wl = np.array([y[w == u][0] for u in wr])
    fold = dict(zip(wr, rf.writer_folds(wr, wl, seed)))
    sc = np.full(len(w), np.nan)
    fw = np.array([fold.get(u, -1) for u in w])
    for f in range(5):
        tr, te = keep & (fw != f), keep & (fw == f)
        Xtr, Xte = X[tr], X[te]
        if covariates is not None:
            reg = LinearRegression().fit(covariates[tr], Xtr)
            Xtr, Xte = Xtr - reg.predict(covariates[tr]), Xte - reg.predict(covariates[te])
        m = rf.estimator("svm"); m.fit(Xtr, y[tr]); sc[te] = m.decision_function(Xte)
    d = pd.DataFrame(dict(w=w[keep], s=sc[keep], y=y[keep])).groupby("w").agg(s=("s", "mean"), y=("y", "first"))
    return balanced_accuracy_score(d.y, (d.s >= 0).astype(int))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    run = ROOT / a.run
    D = rf.Data(run)
    y, w, para = D.y, D.w, D.para
    s1 = [u for u in D.writers if u not in getattr(rf, "NONSTANDARD_CAPTURE", [])]   # no S1 list in the 42-writer release
    cov = np.column_stack([D.desc.ink_density.to_numpy(), D.X["global_intensity"]])
    jobs = []
    for c in rf.STYLE6:
        jobs.append((f"Q1_single:{c}", D.X["style6"][:, rf.STYLE6.index(c)][:, None], None, None))
    for c in ["global_intensity", "background"]:
        jobs.append((f"Q2_S1_control:{c}", D.X[c], s1, None))
    jobs.append(("Q3_hc183_adjusted:all38", D.X["hc183"], None, cov))
    jobs.append(("Q3_hc183_adjusted:S1", D.X["hc183"], s1, cov))
    jobs.append(("Q3_hc183_unadjusted:S1", D.X["hc183"], s1, None))
    rows = []
    for name, X, wsub, cv in jobs:
        bas = Parallel(a.jobs)(delayed(cv_A)(X, y, w, para, s, wsub, cv) for s in rf.SEEDS)
        rows += [dict(analysis=name, seed=s, ba=b) for s, b in zip(rf.SEEDS, bas)]
        print(name, round(float(np.mean(bas)), 3), round(float(np.std(bas, ddof=1)), 3), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(run / "posthoc_confound_by_split.csv", index=False)
    df.groupby("analysis").ba.agg(["mean", "std", "min", "max"]).to_csv(run / "posthoc_confound_summary.csv")


if __name__ == "__main__":
    main()
