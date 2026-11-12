#!/usr/bin/env python3
"""Evaluate re-implemented earlier-work baselines under the writer-independent protocol (IJDAR extension).

Same protocol as the PRL study: 20 x 5-fold writer CV (SVM C = 0.1 fixed; LR / RF tuned on training writers),
999 whole-writer permutations (A, SVM), fold-wise adjustment for ink density + photo intensity, and the evaluation
pitfalls: per-image BA with writer-level vs image-level (leaky) folds, and writer-identification top-1 accuracy.
Reads <dev>/features/baselines.npz (extract_prior_baselines.py). Writes baselines_*.csv into <dev>.
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
import run_posthoc_confound_analyses as pc  # noqa: E402

BASE = ["edge_direction", "edge_hinge", "chain_code", "hog_glbp", "lbp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/final42_2026-10-19")
    ap.add_argument("--dev", default="results/ijdar_dev_2026-11-12")
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(); dev = ROOT / a.dev
    D = rf.Data(ROOT / a.base)
    z = np.load(dev / "features/baselines.npz", allow_pickle=True)
    assert list(z["image_id"]) == list(D.meta.image_id)
    for k in BASE:
        D.X[k] = z[k].astype(float)
    P = Parallel(a.jobs)

    jobs = [(r, "svm", s, ("A", "C", "B")) for r in BASE for s in rf.SEEDS]
    jobs += [(r, c, s, ("A",)) for r in BASE for c in ("logistic", "random_forest") for s in rf.SEEDS]
    par = pd.concat(P(delayed(rf.run_cv)(D, r, c, s, an) for r, c, s, an in jobs), ignore_index=True)
    summ = rf.summarize(rf.metrics(rf.writer_level(par)))
    summ.to_csv(dev / "baselines_metrics_summary.csv", index=False)
    print(summ[["rep", "clf", "analysis", "ba_mean", "ba_sd", "auc_mean"]].round(3).to_string(), flush=True)

    cov = np.column_stack([D.desc.ink_density.to_numpy(), D.X["global_intensity"]])
    rows = []
    for r in BASE:
        b = P(delayed(pc.cv_A)(D.X[r], D.y, D.w, D.para, s, None, cov) for s in rf.SEEDS)
        w = P(delayed(pf.svm_A_writer_ba)(D.X[r], D.y, D.w, D.para, s) for s in rf.SEEDS)
        lk = P(delayed(pf.leaky_ba)(D.X[r], D.y, D.para, s) for s in rf.SEEDS)
        rows.append(dict(rep=r, dims=D.X[r].shape[1], adjusted_ba=np.mean(b), writer_ba=np.mean([x[0] for x in w]),
                         para_ba_writer_split=np.mean([x[1] for x in w]), para_ba_leaky=np.mean(lk),
                         writer_id_top1=pf.writer_id_top1(D.X[r], D.w, D.para)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(dev / "baselines_pitfalls_adjusted.csv", index=False)

    rng = np.random.default_rng(2029); lab = pd.Series(D.wlabels, D.writers)
    perms = [pd.Series(rng.permutation(lab.to_numpy()), lab.index) for _ in range(a.n_perm)]
    prow = []
    for r in BASE:
        obs = rf.perm_stat(D, r, lab)[0]
        null = np.array(P(delayed(rf.perm_stat)(D, r, pl) for pl in perms))[:, 0]
        prow.append(dict(rep=r, observed=obs, p=(np.sum(null >= obs) + 1) / (a.n_perm + 1), null_q95=np.quantile(null, .95)))
        print(prow[-1], flush=True)
        pd.DataFrame(prow).to_csv(dev / "baselines_permutation.csv", index=False)


if __name__ == "__main__":
    main()
