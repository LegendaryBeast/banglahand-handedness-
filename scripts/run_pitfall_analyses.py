#!/usr/bin/env python3
"""Helpers for the evaluation-pitfall analyses (used by run_secondary_analyses.py and run_s3_sensitivity.py).

svm_A_writer_ba  primary SVM, condition A, writer and paragraph BA for one repetition
leaky_ba         image-level (writer-leaky) CV, paragraph-level BA
writer_id_top1   closed-set writer identification (leave-one-paragraph-out 1-NN)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_experiments as rf  # noqa: E402



def svm_A_writer_ba(X, y, w, para, seed):
    sh = para <= 6
    wr = np.array(sorted(set(w)))
    wl = np.array([y[w == u][0] for u in wr])
    fold = dict(zip(wr, rf.writer_folds(wr, wl, seed)))
    fw = np.array([fold[u] for u in w])
    sc = np.full(len(w), np.nan)
    for f in range(5):
        tr, te = sh & (fw != f), sh & (fw == f)
        m = rf.estimator("svm"); m.fit(X[tr], y[tr]); sc[te] = m.decision_function(X[te])
    d = pd.DataFrame(dict(w=w[sh], s=sc[sh], y=y[sh]))
    g = d.groupby("w").agg(s=("s", "mean"), y=("y", "first"))
    return balanced_accuracy_score(g.y, (g.s >= 0).astype(int)), balanced_accuracy_score(d.y, (d.s >= 0).astype(int))


def leaky_ba(X, y, para, seed):
    sh = np.flatnonzero(para <= 6)
    sc = np.empty(len(sh))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(sh, y[sh]):
        m = rf.estimator("svm"); m.fit(X[sh[tr]], y[sh[tr]]); sc[te] = m.decision_function(X[sh[te]])
    return balanced_accuracy_score(y[sh], (sc >= 0).astype(int))


def writer_id_top1(X, w, para):
    sh = para <= 6
    Z = StandardScaler().fit_transform(X[sh])
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    ws = w[sh]
    return float(np.mean(ws[S.argmax(1)] == ws))
