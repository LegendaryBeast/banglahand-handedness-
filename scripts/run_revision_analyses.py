#!/usr/bin/env python3
"""Post hoc analyses of protocol/protocol_amendment_3.json (pre-submission review).

Writes to the run folder:
  capture_time_correlations.csv   writer-level Spearman correlation of capture time with photo/ink measures
  capture_time_adjusted.csv       condition-A writer BA after fold-wise regression on capture time (and on ink + photo + time)
  capture_time_permutation.csv    999 whole-writer permutations under the time adjustment (HC-183, STROKE, photo intensity)
  mirror_augmented_stroke.csv     STROKE SVM trained on original + mirrored ink, tested on original and mirrored images
  mirror_effect_permutation.csv   permutation test of the mirror effect (BA original - BA mirrored) of the STROKE SVM
  writer_bootstrap_ci.csv         95% writer-bootstrap intervals for condition-A writer BA (conditional on the fitted models)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_experiments as rf  # noqa: E402
import run_posthoc_confound_analyses as pc  # noqa: E402

REPS = ["hc183", "style6", "stroke", "combined", "resnet18", "efficientnet_b0", "vit_b_16", "stroke_cnn"]
CTRL = ["global_intensity", "background"]


def capture_minutes(meta):
    t = pd.to_datetime(meta.capture_datetime, format="%Y:%m:%d %H:%M:%S")
    return ((t - t.min()).dt.total_seconds() / 60).to_numpy()


def perm_labels(D, n_perm, seed=2029):
    rng = np.random.default_rng(seed)
    lab = pd.Series(D.wlabels, D.writers)
    return [pd.Series(rng.permutation(lab.to_numpy()), lab.index) for _ in range(n_perm)]


def img_labels(D, wl):
    return np.array([wl[u] for u in D.w])


def mean_ba(X, y, D, cov):
    return float(np.mean([pc.cv_A(X, y, D.w, D.para, s, None, cov) for s in rf.SEEDS]))


def mirror_aug(D, seed, augment=True):
    """STROKE SVM on condition A; training on original (+ mirrored) features; test on original and mirrored."""
    X, Xm = D.X["stroke"], D.X["stroke_mirror"]
    sh = D.para <= 6
    fold = dict(zip(D.writers, rf.writer_folds(D.writers, np.array([D.y[D.w == u][0] for u in D.writers]), seed)))
    fw = np.array([fold[u] for u in D.w])
    so, sm = np.full(len(D.w), np.nan), np.full(len(D.w), np.nan)
    for f in range(5):
        tr, te = sh & (fw != f), sh & (fw == f)
        Xtr, ytr = (np.vstack([X[tr], Xm[tr]]), np.concatenate([D.y[tr], D.y[tr]])) if augment else (X[tr], D.y[tr])
        m = rf.estimator("svm"); m.fit(Xtr, ytr)
        so[te] = m.decision_function(X[te]); sm[te] = m.decision_function(Xm[te])
    d = pd.DataFrame(dict(w=D.w[sh], y=D.y[sh], so=so[sh], sm=sm[sh])).groupby("w").mean()
    po, pm = (d.so >= 0).astype(int), (d.sm >= 0).astype(int)
    return balanced_accuracy_score(d.y, po), balanced_accuracy_score(d.y, pm)


def mirror_effect(D, y_img):
    """Mean over the 20 repetitions of BA(original) - BA(mirrored) for the declared (non-augmented) STROKE SVM."""
    keep_y = D.y
    D.y = y_img
    try:
        r = [mirror_aug(D, s, augment=False) for s in rf.SEEDS]
    finally:
        D.y = keep_y
    return float(np.mean([a - b for a, b in r]))


def boot_ci(pred, n_boot=2000, seed=7):
    """pred: DataFrame seed, writer_id, label, pred. Class-stratified writer bootstrap of the mean-over-seeds BA."""
    rng = np.random.default_rng(seed)
    P = pred.pivot_table(index="writer_id", columns="seed", values="pred")
    lab = pred.groupby("writer_id").label.first().loc[P.index].to_numpy()
    M = P.to_numpy()
    L, R = np.flatnonzero(lab == 1), np.flatnonzero(lab == 0)
    def ba(idx):
        y = lab[idx]; p = M[idx]
        tpr = (p[y == 1] == 1).mean(0); tnr = (p[y == 0] == 0).mean(0)
        return float(np.mean((tpr + tnr) / 2))
    b = [ba(np.concatenate([rng.choice(L, len(L)), rng.choice(R, len(R))])) for _ in range(n_boot)]
    return ba(np.arange(len(lab))), *np.percentile(b, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=999)
    a = ap.parse_args(); run = ROOT / a.run
    D = rf.Data(run); P = Parallel(a.jobs)
    t = capture_minutes(D.meta)
    ts = np.column_stack([t / 60, (t / 60) ** 2])
    acq = np.column_stack([D.desc.ink_density.to_numpy(), D.X["global_intensity"]])

    # 1a. correlations at writer level
    fr = pd.read_csv(run / "framing_check_images.csv").set_index("image_id").reindex(D.meta.image_id).reset_index()   # shared paragraphs only
    wm = pd.DataFrame(dict(writer_id=D.w, label=D.y, t=t, paper_level=fr.paper_level, ink_contrast=fr.ink_contrast,
                           photo_mean=D.X["global_intensity"][:, 0]))[D.para <= 6].groupby("writer_id").mean()
    rng = np.random.default_rng(1)
    rows = []
    for m in ["paper_level", "ink_contrast", "photo_mean"]:
        for grp, g in [("all", wm), ("left", wm[wm.label == 1]), ("right", wm[wm.label == 0])]:
            rho = spearmanr(g.t, g[m]).statistic
            null = np.array([spearmanr(rng.permutation(g.t.to_numpy()), g[m]).statistic for _ in range(10000)])
            rows.append(dict(measure=m, group=grp, n=len(g), rho=rho, p_perm=(np.sum(np.abs(null) >= abs(rho)) + 1) / 10001))
    rows.append(dict(measure="class_vs_time", group="all", n=len(wm), rho=spearmanr(wm.t, wm.label).statistic, p_perm=np.nan))
    pd.DataFrame(rows).to_csv(run / "capture_time_correlations.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)

    # 1b. time-adjusted models
    rows = []
    for r in REPS + CTRL:
        for name, cov in [("time", ts), ("acquisition_and_time", np.column_stack([acq, ts]) if r not in CTRL else None)]:
            if cov is None:
                continue
            b = P(delayed(pc.cv_A)(D.X[r], D.y, D.w, D.para, s, None, cov) for s in rf.SEEDS)
            rows.append(dict(rep=r, covariates=name, ba_mean=np.mean(b), ba_sd=np.std(b, ddof=1)))
        print("time-adjusted", r, flush=True)
    pd.DataFrame(rows).to_csv(run / "capture_time_adjusted.csv", index=False)

    # 1c. permutations under the time adjustment
    perms = perm_labels(D, a.n_perm)
    rows = []
    for r in ["hc183", "stroke", "global_intensity"]:
        obs = mean_ba(D.X[r], D.y, D, ts)
        null = np.array(P(delayed(mean_ba)(D.X[r], img_labels(D, pl), D, ts) for pl in perms))
        rows.append(dict(rep=r, covariates="time", observed=obs, p=(np.sum(null >= obs) + 1) / (a.n_perm + 1), null_mean=null.mean(), n_perm=a.n_perm))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(run / "capture_time_permutation.csv", index=False)

    # 2. mirror-augmented STROKE and permutation test of the mirror effect
    r = P(delayed(mirror_aug)(D, s, True) for s in rf.SEEDS)
    pd.DataFrame([dict(seed=s, ba_original=o, ba_mirrored=m) for s, (o, m) in zip(rf.SEEDS, r)]).to_csv(run / "mirror_augmented_stroke.csv", index=False)
    print("mirror-augmented STROKE", np.mean([x[0] for x in r]), np.mean([x[1] for x in r]), flush=True)
    obs = mirror_effect(D, D.y.copy())
    null = np.array(P(delayed(mirror_effect)(D, img_labels(D, pl)) for pl in perms))
    pd.DataFrame([dict(statistic="BA_original_minus_BA_mirrored", observed=obs, p=(np.sum(null >= obs) + 1) / (a.n_perm + 1),
                       null_mean=null.mean(), null_q95=np.quantile(null, 0.95), n_perm=a.n_perm)]).to_csv(run / "mirror_effect_permutation.csv", index=False)
    print("mirror effect", obs, (np.sum(null >= obs) + 1) / (a.n_perm + 1), flush=True)

    # 3. writer-bootstrap intervals
    wp = pd.read_csv(run / "writer_predictions.csv.gz")
    wp = wp[(wp.clf == "svm") & (wp.analysis == "A")]
    rows = []
    for r in REPS:
        est, lo, hi = boot_ci(wp[wp.rep == r]); rows.append(dict(rep=r, ba=est, ci_low=lo, ci_high=hi))
    for r in CTRL:
        par = pd.concat(P(delayed(rf.run_cv)(D, r, "svm", s, analyses=("A",)) for s in rf.SEEDS))
        est, lo, hi = boot_ci(rf.writer_level(par)); rows.append(dict(rep=r, ba=est, ci_low=lo, ci_high=hi))
    ft = pd.read_csv(run / "finetune_writer_predictions.csv").assign(pred=lambda d: (d.A >= 0.5).astype(int))
    est, lo, hi = boot_ci(ft); rows.append(dict(rep="finetuned", ba=est, ci_low=lo, ci_high=hi))
    pd.DataFrame(rows).to_csv(run / "writer_bootstrap_ci.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
