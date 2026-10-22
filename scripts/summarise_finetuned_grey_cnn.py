#!/usr/bin/env python3
"""Summarise the grey-image fine-tuned ResNet-18 baseline (run_finetuned_grey_cnn.py, run on a Kaggle T4).

Reads --run/finetune_grey_writer_predictions.csv and --run/finetune_grey_embeddings.npz. The fine-tuned embedding of
each outer fold (network trained on the training writers' shared paragraphs) is evaluated with the same classifiers,
folds and text conditions as the other representations (run_final_experiments.run_cv on that fold only): linear SVM in
A, C, B and All 8, logistic regression and random forest in A. Writes
  finetune_grey_paragraph_predictions.csv.gz   paragraph scores of every classifier, condition, seed and fold
  finetune_grey_summary.csv                    per readout: writer BA (mean, s.d. over the 20 repetitions),
                                               within-fold AUC and the 95% writer-bootstrap interval
Readouts: <clf>_<condition> from the embeddings; e2e_A, e2e_C end to end (mean tile probability >= 0.5);
svm_adj_A, the SVM after ink density and photo intensity are regressed out within the training folds.
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
from run_revision_analyses import boot_ci  # noqa: E402

REP = "finetuned_grey"
JOBS = [("svm", ("A", "C", "B", "ALL8")), ("logistic", ("A",)), ("random_forest", ("A",))]


def one_fold(D, emb, seed, f, clf, analyses):
    D.X[REP] = emb.astype(np.float64)
    return rf.run_cv(D, REP, clf, seed, analyses=analyses, only_fold=f)


def row(name, w):
    s = rf.summarize(rf.metrics(w)).iloc[0]
    est, lo, hi = boot_ci(w)
    assert abs(est - s.ba_mean) < 1e-9
    return dict(readout=name, ba_mean=s.ba_mean, ba_sd=s.ba_sd, auc_mean=s.auc_mean, ci_low=lo, ci_high=hi, n_splits=s.n_splits)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(); run = ROOT / a.run
    D = rf.Data(run)
    z = np.load(run / "finetune_grey_embeddings.npz", allow_pickle=True)
    assert list(z["image_id"]) == list(D.meta.image_id)
    seeds, E = list(z["seeds"]), z["emb"]
    par = pd.concat(Parallel(a.jobs)(delayed(one_fold)(D, E[i, f], s, f, clf, an)
                                     for i, s in enumerate(seeds) for f in range(5) for clf, an in JOBS), ignore_index=True)
    par.drop(columns="params").to_csv(run / "finetune_grey_paragraph_predictions.csv.gz", index=False)
    wl = rf.writer_level(par)
    rows = [row(f"{clf}_{an}", wl[(wl.clf == clf) & (wl.analysis == an)]) for clf, ans in JOBS for an in ans]
    wp = pd.read_csv(run / "finetune_grey_writer_predictions.csv")
    for name, col, thr in [("e2e_A", "A", 0.5), ("e2e_C", "C", 0.5), ("svm_adj_A", "svm_adj", 0.0)]:
        rows.append(row(name, wp.assign(rep=REP, clf="svm", analysis=name, score=wp[col], pred=(wp[col] >= thr).astype(int))))
    chk = wl[(wl.clf == "svm") & (wl.analysis == "A")].merge(wp, on=["seed", "writer_id"])
    print("SVM A from saved embeddings vs training run: max score difference", float(np.abs(chk.score - chk.svm).max()))
    out = pd.DataFrame(rows); out.to_csv(run / "finetune_grey_summary.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
