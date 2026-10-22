#!/usr/bin/env python3
"""Amendment 2 (post hoc): permutation test of acquisition-adjusted representations."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_experiments as rf  # noqa: E402
import run_posthoc_confound_analyses as pc  # noqa: E402


def stat(X, y_img, w, para, writers, cov, seeds):
    return float(np.mean([pc.cv_A(X, y_img, w, para, s, writers, cov) for s in seeds]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--reps", default="stroke,hc183,combined")
    ap.add_argument("--output", default="adjusted_permutation.csv")
    a = ap.parse_args(); run = ROOT / a.run; D = rf.Data(run)
    cov = np.column_stack([D.desc.ink_density.to_numpy(), D.X["global_intensity"]])
    rows = []
    plan = [("all", list(D.writers), r, a.n_perm) for r in a.reps.split(",")]
    for subset, ws, rep, n_perm in plan:
        rng = np.random.default_rng(2029)
        lab = pd.Series(D.wlabels, D.writers).loc[ws]
        perms = [pd.Series(rng.permutation(lab.to_numpy()), lab.index) for _ in range(n_perm)]
        if True:
            obs = stat(D.X[rep], D.y, D.w, D.para, ws, cov, rf.SEEDS)
            def one(pl):
                y = pd.Series(D.y, index=D.w).copy()
                yp = np.array([pl.get(u, -1) for u in D.w])
                return stat(D.X[rep], yp, D.w, D.para, ws, cov, rf.SEEDS)
            null = np.array(Parallel(a.jobs)(delayed(one)(pl) for pl in perms))
            rows.append(dict(subset=subset, rep=rep, n_writers=len(ws), n_perm=n_perm, observed=obs, p=(np.sum(null >= obs) + 1) / (n_perm + 1),
                             null_mean=null.mean(), null_q95=np.quantile(null, 0.95)))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(run / a.output, index=False)
    pd.DataFrame(rows).to_csv(run / a.output, index=False)


if __name__ == "__main__":
    main()
