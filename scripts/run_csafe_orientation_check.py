#!/usr/bin/env python3
"""CSAFE post hoc check: is the handedness signal line orientation? (IJDAR, exploratory, added after the primary run.)

The writers' own line geometry differed between the groups on CSAFE (skew, line slope, line spacing; csafe_layout_tests.csv).
Here those measures are regressed out of every feature inside each training fold (run_csafe_evaluation.cv_ba, unchanged),
alone, together with ink density and scan intensity, and as all 16 layout measures. HC-183 (text block) also gets 999
whole-writer permutations of the orientation-adjusted estimate (same rng 2029 permutations as the primary run).
Writes results/csafe_ijdar_2026-11-12/csafe_orientation_check.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_csafe_evaluation as ce  # noqa: E402

OUT = ROOT / "results/csafe_ijdar_2026-11-12"


def main(jobs=8, n_perm=999):
    D = ce.CsafeData(OUT); P = Parallel(jobs)
    lay = pd.DataFrame(D.X["layout"], columns=D.layout_names)
    orient = lay[["skew_deg", "line_slope_mean", "spacing_rel"]].to_numpy()
    acq = np.column_stack([D.ink[:, 0], D.X["global_intensity"]])
    covs = {"none": None, "orientation": orient, "orientation+ink+scan": np.column_stack([orient, acq]),
            "all layout": D.X["layout"], "all layout+ink+scan": np.column_stack([D.X["layout"], acq])}
    lab = pd.Series(D.wlabels, D.writers); rows = []
    for r in ("hc183_tb", "stroke", "hog_glbp"):
        for name, cov in covs.items():
            res = np.array(P(delayed(ce.cv_ba)(D, D.X[r], lab, s, cov) for s in ce.rf.SEEDS)).mean(0)
            rows.append(dict(rep=r, covariates=name, A=res[0], X=res[1])); print(rows[-1], flush=True)
    rng = np.random.default_rng(2029)
    perms = [pd.Series(rng.permutation(lab.to_numpy()), lab.index) for _ in range(n_perm)]
    obs = [x for x in rows if x["rep"] == "hc183_tb" and x["covariates"] == "orientation"][0]
    null = np.array(P(delayed(ce.mean_ba)(D, D.X["hc183_tb"], pl, orient) for pl in perms))
    for j, an in enumerate(("A", "X")):
        obs[f"{an}_p"] = (np.sum(null[:, j] >= obs[an]) + 1) / (n_perm + 1)
    pd.DataFrame(rows).to_csv(OUT / "csafe_orientation_check.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
