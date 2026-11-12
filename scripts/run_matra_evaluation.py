#!/usr/bin/env python3
"""Development evaluation of MATRA-INK on BanglaHand v1 (IJDAR extension; exploratory, not confirmatory).

Uses the protocol of the PRL study unchanged: 20 x 5-fold writer cross-validation, every image of a test writer
held out, fixed linear SVM (C = 0.1), writer score = mean paragraph margin, whole-writer label permutations that
regenerate the folds, and fold-wise residualisation on ink density + the seven photo-intensity statistics.

Reads   results/final42_2026-10-19/features (existing representations) and <dev>/features/matra_features.npz
Writes  <dev>/matra_metrics_summary.csv, matra_permutation.csv, matra_adjusted.csv, matra_mirror_test.csv,
        matra_feature_tests.csv
The descriptor (scripts/matra_features.py) was fixed before any of these label-dependent analyses were run.
Confirmatory testing is reserved for the held-out v2 cohort (manuscripts/ijdar/IJDAR_EXTENSION_PLAN.md).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import matra_features as mf  # noqa: E402
import run_final_experiments as rf  # noqa: E402
import run_posthoc_confound_analyses as pc  # noqa: E402

DIRECTION = [n for n in mf.NAMES if n.startswith(("along_", "lr_", "overhang_")) or n == "angle_mean_deg"]
DEPOSITION = [n for n in mf.NAMES if n.startswith(("ratio_", "cross_")) or n in ("texture_cv", "skip_fraction")]
GEOMETRY = ["length_mean_w", "matra_per_kpx", "angle_sd_deg"]
assert sorted(DIRECTION + DEPOSITION + GEOMETRY) == sorted(mf.NAMES)


def load(base, dev):
    D = rf.Data(base)
    z = np.load(dev / "features/matra_features.npz", allow_pickle=True)
    assert list(z["image_id"]) == list(D.meta.image_id)
    nan_report = {}
    for key in ("matra", "matra_mirror"):
        X = z[key].astype(float)
        nan_report[key] = int((~np.isfinite(X)).sum())
        med = np.nanmedian(z["matra"].astype(float), axis=0)        # label-free imputation (same medians for both)
        X = np.where(np.isfinite(X), X, med)
        D.X[key] = X
    idx = {n: i for i, n in enumerate(mf.NAMES)}
    for name, cols in (("matra_direction", DIRECTION), ("matra_deposition", DEPOSITION), ("matra_geometry", GEOMETRY)):
        D.X[name] = D.X["matra"][:, [idx[c] for c in cols]]
        D.X[name + "_mirror"] = D.X["matra_mirror"][:, [idx[c] for c in cols]]
    D.X["hc183+matra"] = np.hstack([D.X["hc183"], D.X["matra"]])
    D.X["stroke+matra"] = np.hstack([D.X["stroke"], D.X["matra"]])
    D.X["combined+matra"] = np.hstack([D.X["combined"], D.X["matra"]])
    return D, nan_report


def mirror_cv(D, rep, seed):
    X, Xm = D.X[rep], D.X[rep + "_mirror"]
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
    return dict(rep=rep, seed=seed, ba_original=balanced_accuracy_score(d.y, po),
                ba_mirrored=balanced_accuracy_score(d.y, pm), flip_fraction=float((po != pm).mean()))


def hedges(v, y):
    a, b = v[y == 1], v[y == 0]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9))


def feature_tests(D, n_perm=10000, seed=2031):
    sh = D.para <= 6
    wm = pd.DataFrame(D.X["matra"][sh], columns=mf.NAMES).assign(w=D.w[sh], y=D.y[sh]).groupby("w").mean()
    y = wm.y.to_numpy().astype(int)
    rng = np.random.default_rng(seed)
    perms = np.array([rng.permutation(y) for _ in range(n_perm)])
    rows = []
    for n in mf.NAMES:
        v = wm[n].to_numpy()
        obs = v[y == 1].mean() - v[y == 0].mean()
        null = (perms * v).sum(1) / y.sum() - ((1 - perms) * v).sum(1) / (1 - y).sum()
        rows.append(dict(feature=n, block="direction" if n in DIRECTION else "deposition" if n in DEPOSITION else "geometry",
                         left_mean=v[y == 1].mean(), right_mean=v[y == 0].mean(), hedges_g=hedges(v, y),
                         p_perm=(np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)))
    t = pd.DataFrame(rows)
    for blk, g in t.groupby("block"):                          # Benjamini-Hochberg within block
        p = g.p_perm.to_numpy(); o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1)
        qq = np.empty(len(p)); qq[o] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1)
        t.loc[g.index, "q_bh_block"] = qq
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/final42_2026-10-19")
    ap.add_argument("--dev", default="results/ijdar_dev_2026-11-12")
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    base, dev = ROOT / a.base, ROOT / a.dev
    D, nan_report = load(base, dev)
    P = Parallel(a.jobs)
    reps = ["matra", "matra_direction", "matra_deposition", "matra_geometry", "hc183", "stroke", "combined",
            "hc183+matra", "stroke+matra", "combined+matra"]

    # 1. Repeated writer CV, conditions A, C, B (SVM) and A with LR / RF
    jobs = [(r, "svm", s, ("A", "C", "B")) for r in reps for s in rf.SEEDS]
    jobs += [(r, c, s, ("A",)) for r in ["matra", "hc183+matra"] for c in ("logistic", "random_forest") for s in rf.SEEDS]
    par = pd.concat(P(delayed(rf.run_cv)(D, r, c, s, an) for r, c, s, an in jobs), ignore_index=True)
    summ = rf.summarize(rf.metrics(rf.writer_level(par)))
    summ.to_csv(dev / "matra_metrics_summary.csv", index=False)
    print(summ[["rep", "clf", "analysis", "ba_mean", "ba_sd", "auc_mean"]].round(3).to_string(), flush=True)

    # 2. Adjustment for ink density + photo intensity (fold-wise)
    cov = np.column_stack([D.desc.ink_density.to_numpy(), D.X["global_intensity"]])
    adj = []
    for r in ["matra", "matra_direction", "matra_deposition", "hc183+matra"]:
        b = P(delayed(pc.cv_A)(D.X[r], D.y, D.w, D.para, s, None, cov) for s in rf.SEEDS)
        adj.append(dict(rep=r, adjusted_mean=np.mean(b), adjusted_sd=np.std(b, ddof=1)))
    adj = pd.DataFrame(adj); adj.to_csv(dev / "matra_adjusted.csv", index=False)
    print(adj.round(3).to_string(), flush=True)

    # 3. Whole-writer permutation tests (A, SVM), raw and adjusted
    rng = np.random.default_rng(2029)
    lab = pd.Series(D.wlabels, D.writers)
    perms = [pd.Series(rng.permutation(lab.to_numpy()), lab.index) for _ in range(a.n_perm)]
    rows = []
    for r in ["matra", "matra_direction", "matra_deposition"]:
        obs = rf.perm_stat(D, r, lab)[0]
        null = np.array(P(delayed(rf.perm_stat)(D, r, pl) for pl in perms))[:, 0]
        rows.append(dict(rep=r, adjusted=False, observed=obs, p=(np.sum(null >= obs) + 1) / (a.n_perm + 1),
                         null_mean=null.mean(), null_q95=np.quantile(null, .95)))
        print(rows[-1], flush=True)
    for r in ["matra", "matra_deposition"]:
        obs = float(np.mean([pc.cv_A(D.X[r], D.y, D.w, D.para, s, None, cov) for s in rf.SEEDS]))
        def one(pl, r=r):
            yp = pl.reindex(D.w).to_numpy()
            return float(np.mean([pc.cv_A(D.X[r], yp, D.w, D.para, s, None, cov) for s in rf.SEEDS]))
        null = np.array(P(delayed(one)(pl) for pl in perms))
        rows.append(dict(rep=r, adjusted=True, observed=obs, p=(np.sum(null >= obs) + 1) / (a.n_perm + 1),
                         null_mean=null.mean(), null_q95=np.quantile(null, .95)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(dev / "matra_permutation.csv", index=False)

    # 4. Mirror test
    mir = pd.DataFrame(P(delayed(mirror_cv)(D, r, s) for r in ["matra", "matra_direction", "matra_deposition"] for s in rf.SEEDS))
    mir.to_csv(dev / "matra_mirror_test.csv", index=False)
    print(mir.groupby("rep")[["ba_original", "ba_mirrored", "flip_fraction"]].mean().round(3).to_string(), flush=True)

    # 5. Writer-level feature differences (descriptive)
    ft = feature_tests(D)
    ft.to_csv(dev / "matra_feature_tests.csv", index=False)
    print(ft.sort_values("p_perm").head(12).round(4).to_string(), flush=True)
    (dev / "matra_evaluation_provenance.json").write_text(json.dumps(dict(
        base=a.base, n_perm=a.n_perm, seeds=rf.SEEDS, imputed_values=nan_report,
        status="exploratory development on v1; descriptor fixed before label-dependent analyses"), indent=2))


if __name__ == "__main__":
    main()
