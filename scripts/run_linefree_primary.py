#!/usr/bin/env python3
"""Full Table 1 / Table 2 rows for the line-free representations (IJDAR, printed-border control).

hc183_lf: HC-183 on the HC-183 ink with printed border lines removed, cropped to the text block
(run_layout_check.py, features/layout.npz: hc183_textblock). resnet18_lf, efficientnet_b0_lf, vit_b_16_lf: frozen
embeddings of line-free text blocks (run_lineless_embeddings.py). Same functions as the PRL tables:
run_final_experiments.run_cv over all text conditions (SVM) and condition A (LR, RF), writer-bootstrap interval
(run_revision_analyses.boot_ci), adjustment for ink + photo intensity, capture time, and both
(run_posthoc_confound_analyses.cv_A), pitfalls (run_pitfall_analyses), text contrasts, 999 whole-writer permutations.
Writes <dev>/linefree_metrics_summary.csv, linefree_table_rows.csv.
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
import run_revision_analyses as rv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/final42_2026-10-19")
    ap.add_argument("--dev", default="results/ijdar_dev_2026-11-12")
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--reps", nargs="+", default=["hc183", "hc183_lf", "resnet18_lf", "efficientnet_b0_lf", "vit_b_16_lf"])
    ap.add_argument("--out", default="linefree_table_rows.csv")
    a = ap.parse_args(); dev = ROOT / a.dev
    D = rf.Data(ROOT / a.base); P = Parallel(a.jobs)
    z = np.load(dev / "features/layout.npz", allow_pickle=True)
    assert list(z["image_id"]) == list(D.meta.image_id)
    D.X["hc183_lf"] = z["hc183_textblock"].astype(float)
    e = np.load(dev / "features/lineless_embeddings.npz", allow_pickle=True)
    for k in ("resnet18", "efficientnet_b0", "vit_b_16"):
        D.X[k + "_lf"] = e[k].astype(float)
    s = dev / "features/stroke_linefree.npz"
    if s.exists():
        D.X["stroke_lf"] = np.load(s)["stroke_lf"].astype(float)
    reps = a.reps                                      # default includes hc183 as a reproduction check

    jobs = [(r, "svm", s, tuple(rf.TESTS)) for r in reps for s in rf.SEEDS]
    jobs += [(r, c, s, ("A",)) for r in reps for c in ("logistic", "random_forest") for s in rf.SEEDS]
    par = pd.concat(P(delayed(rf.run_cv)(D, r, c, s, an) for r, c, s, an in jobs), ignore_index=True)
    wl = rf.writer_level(par); m = rf.metrics(wl); S = rf.summarize(m)
    S.to_csv(dev / a.out.replace("table_rows", "metrics_summary"), index=False)
    piv = m[m.clf == "svm"].pivot_table(index=["rep", "seed"], columns="analysis", values="ba")
    con = pd.DataFrame(dict(C_minus_D=piv["C"] - piv[["D_S1", "D_S2", "D_S3"]].mean(1),
                            B_minus_E=piv["B"] - piv[["E_S1", "E_S2", "E_S3"]].mean(1))).groupby("rep").mean()

    t = rv.capture_minutes(D.meta); ts = np.column_stack([t / 60, (t / 60) ** 2])
    acq = np.column_stack([D.desc.ink_density.to_numpy(), D.X["global_intensity"]])
    lab = pd.Series(D.wlabels, D.writers); rng = np.random.default_rng(2029)
    perms = [pd.Series(rng.permutation(lab.to_numpy()), lab.index) for _ in range(a.n_perm)]
    g = lambda r, an, clf="svm", col="ba_mean": S[(S.rep == r) & (S.clf == clf) & (S.analysis == an)][col].iloc[0]
    rows = []
    for r in reps:
        wa = wl[(wl.rep == r) & (wl.clf == "svm") & (wl.analysis == "A")].rename(columns={"writer_id": "writer_id"})
        est, lo, hi = rv.boot_ci(wa)
        cv = lambda cov: float(np.mean(P(delayed(pc.cv_A)(D.X[r], D.y, D.w, D.para, s, None, cov) for s in rf.SEEDS)))
        w = P(delayed(pf.svm_A_writer_ba)(D.X[r], D.y, D.w, D.para, s) for s in rf.SEEDS)
        lk = P(delayed(pf.leaky_ba)(D.X[r], D.y, D.para, s) for s in rf.SEEDS)
        obs = rf.perm_stat(D, r, lab)[0]
        null = np.array(P(delayed(rf.perm_stat)(D, r, pl) for pl in perms))[:, 0]
        rows.append(dict(rep=r, A=g(r, "A"), A_sd=g(r, "A", col="ba_sd"), ci_low=lo, ci_high=hi,
                         LR=g(r, "A", "logistic"), RF=g(r, "A", "random_forest"),
                         C=g(r, "C"), C_sd=g(r, "C", col="ba_sd"), B=g(r, "B"), B_sd=g(r, "B", col="ba_sd"),
                         ALL8=g(r, "ALL8"), ALL8_sd=g(r, "ALL8", col="ba_sd"), auc=g(r, "A", col="auc_mean"),
                         D_S1=g(r, "D_S1"), D_S2=g(r, "D_S2"), D_S3=g(r, "D_S3"),
                         E_S1=g(r, "E_S1"), E_S2=g(r, "E_S2"), E_S3=g(r, "E_S3"),
                         C_minus_D=con.loc[r, "C_minus_D"], B_minus_E=con.loc[r, "B_minus_E"],
                         perm_observed=obs, p=(np.sum(null >= obs) + 1) / (a.n_perm + 1),
                         adjusted=cv(acq), time_adj=cv(ts), acq_time=cv(np.column_stack([acq, ts])),
                         para_ba_writer=np.mean([x[1] for x in w]), para_ba_leaky=np.mean(lk),
                         writer_id_top1=pf.writer_id_top1(D.X[r], D.w, D.para)))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
        pd.DataFrame(rows).to_csv(dev / a.out, index=False)


if __name__ == "__main__":
    main()
