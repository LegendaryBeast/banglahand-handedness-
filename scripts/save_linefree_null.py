#!/usr/bin/env python3
"""Recompute and save the line-free HC-183 permutation null (same rng 2029 permutations as run_linefree_primary.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_experiments as rf  # noqa: E402

D = rf.Data(ROOT / "results/final42_2026-10-19")
z = np.load(ROOT / "results/ijdar_dev_2026-11-12/features/layout.npz", allow_pickle=True)
assert list(z["image_id"]) == list(D.meta.image_id)
D.X["hc183_lf"] = z["hc183_textblock"].astype(float)
lab = pd.Series(D.wlabels, D.writers); rng = np.random.default_rng(2029)
perms = [pd.Series(rng.permutation(lab.to_numpy()), lab.index) for _ in range(999)]
obs = rf.perm_stat(D, "hc183_lf", lab)[0]
null = np.array(Parallel(int(sys.argv[1]) if len(sys.argv) > 1 else 2)(delayed(rf.perm_stat)(D, "hc183_lf", p) for p in perms))[:, 0]
out = ROOT / "results/ijdar_dev_2026-11-12/permutation_null_hc183_lf.csv"
pd.DataFrame(dict(null_mean_ba=null)).to_csv(out, index=False)
print("observed", obs, "p", (np.sum(null >= obs) + 1) / 1000, "->", out)
