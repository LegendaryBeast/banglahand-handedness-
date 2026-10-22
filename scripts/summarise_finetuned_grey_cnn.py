#!/usr/bin/env python3
"""Summarise the grey-image fine-tuned ResNet-18 baseline (run_finetuned_grey_cnn.py, run on a Kaggle T4).

Reads --run/finetune_grey_writer_predictions.csv and writes --run/finetune_grey_summary.csv with, for each readout,
writer BA (mean and s.d. over the 20 repetitions), within-fold AUC and the 95% writer-bootstrap interval, computed
exactly as for the other representations (run_final_experiments.metrics, run_revision_analyses.boot_ci).
  e2e_A, e2e_C   end to end (mean tile probability >= 0.5), conditions A and C
  svm_A          linear SVM on the fine-tuned features, condition A
  svm_adj_A      the same after ink density and photo intensity are regressed out within the training folds
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_experiments as rf  # noqa: E402
from run_revision_analyses import boot_ci  # noqa: E402

READOUTS = {"e2e_A": ("A", 0.5), "e2e_C": ("C", 0.5), "svm_A": ("svm", 0.0), "svm_adj_A": ("svm_adj", 0.0)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    run = ROOT / ap.parse_args().run
    wp = pd.read_csv(run / "finetune_grey_writer_predictions.csv")
    rows = []
    for name, (col, thr) in READOUTS.items():
        w = wp.assign(rep="finetuned_grey", clf="svm", analysis=name, score=wp[col], pred=(wp[col] >= thr).astype(int))
        s = rf.summarize(rf.metrics(w)).iloc[0]
        est, lo, hi = boot_ci(w)
        rows.append(dict(readout=name, ba_mean=s.ba_mean, ba_sd=s.ba_sd, auc_mean=s.auc_mean, ci_low=lo, ci_high=hi, n_splits=s.n_splits))
        assert abs(est - s.ba_mean) < 1e-9
    out = pd.DataFrame(rows); out.to_csv(run / "finetune_grey_summary.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
