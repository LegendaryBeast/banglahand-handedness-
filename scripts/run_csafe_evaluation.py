#!/usr/bin/env python3
"""CSAFE v2: the BanglaHand writer-independent protocol applied to an external dataset (IJDAR).

Follows results/csafe_ijdar_2026-11-12/protocol.json. Reuses run_final_experiments (run_cv, writer_level, metrics,
summarize, perm_stat, estimator, writer_folds, SEEDS), run_pitfall_analyses, run_revision_analyses.boot_ci unchanged.
CSAFE pages are mapped to paragraph numbers: session 1 -> 1-6 (condition A), session 2 -> 7-8, session 3 -> 9;
condition X applies model A to paragraphs 7-9 (C2: 7-8, C3: 9).
Writes <out>/csafe_metrics_summary.csv, csafe_table.csv, csafe_layout_tests.csv, csafe_evaluation.log via stdout.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_experiments as rf  # noqa: E402
import run_pitfall_analyses as pf  # noqa: E402
import run_revision_analyses as rv  # noqa: E402

EXTRA_TESTS = {"X": ("A", [7, 8, 9]), "C2": ("A", [7, 8]), "C3": ("A", [9])}
rf.TESTS.update(EXTRA_TESTS)


def run_cv_x(*args):
    """rf.run_cv with the CSAFE conditions registered (joblib workers import run_final_experiments afresh)."""
    rf.TESTS.update(EXTRA_TESTS)
    return rf.run_cv(*args)
X_PARAS = [7, 8, 9]
REPS = ["hc183_tb", "hc183_page", "stroke", "resnet18", "efficientnet_b0", "vit_b_16", "edge_direction", "edge_hinge",
        "chain_code", "hog_glbp", "lbp", "global_intensity", "background", "layout"]
CTRL = {"global_intensity", "background", "layout"}


class CsafeData:
    def __init__(self, out):
        f = out / "features"
        self.meta = pd.read_csv(f / "csafe_meta.csv").assign(text_truncated_at_edge="")
        h = np.load(f / "csafe_hand.npz", allow_pickle=True); d = np.load(f / "csafe_deep.npz", allow_pickle=True)
        assert list(h["image_id"]) == list(self.meta.image_id) == list(d["image_id"])
        self.layout_names = list(h["layout_names"])
        self.X = {k: h[k].astype(float) for k in h.files if k not in ("image_id", "layout_names", "ink_density")}
        self.X.update({k: d[k].astype(float) for k in d.files if k != "image_id"})
        lay = self.X["layout"]; self.X["layout"] = np.where(np.isfinite(lay), lay, np.nanmedian(lay, 0))
        self.ink = h["ink_density"]
        self.desc = pd.DataFrame(dict(ink_density=self.ink[:, 0]))
        self.y = self.meta.label.to_numpy(); self.w = self.meta.writer_id.to_numpy(); self.para = self.meta.paragraph_number.to_numpy()
        self.trunc = np.zeros(len(self.y), bool)
        wt = self.meta.groupby("writer_id").label.first().sort_index()
        self.writers, self.wlabels = wt.index.to_numpy(), wt.to_numpy()


def cv_ba(D, X, lab_w, seed, cov=None):
    """Fixed SVM trained on condition-A pages of training writers; writer BA on the test writers' condition-A pages and
    on their later-session pages (condition X), from the same fitted models."""
    lab_img = lab_w.reindex(D.w).to_numpy(); wl = lab_w.loc[D.writers].to_numpy()
    fold = dict(zip(D.writers, rf.writer_folds(D.writers, wl, seed))); fw = np.array([fold[u] for u in D.w])
    trp = np.isin(D.para, rf.FITS["A"])
    sc = np.full(len(D.w), np.nan)
    for f in range(5):
        tr, te = trp & (fw != f), fw == f
        Xtr, Xte = X[tr], X[te]
        if cov is not None:
            reg = LinearRegression().fit(cov[tr], Xtr); Xtr, Xte = Xtr - reg.predict(cov[tr]), Xte - reg.predict(cov[te])
        m = rf.estimator("svm"); m.fit(Xtr, lab_img[tr]); sc[te] = m.decision_function(Xte)
    out = []
    for tp in (rf.FITS["A"], X_PARAS):
        k = np.isin(D.para, tp)
        g = pd.DataFrame(dict(w=D.w[k], s=sc[k], y=lab_img[k])).groupby("w").agg(s=("s", "mean"), y=("y", "first"))
        out.append(balanced_accuracy_score(g.y, (g.s >= 0).astype(int)))
    return out


def mean_ba(D, X, lab_w, cov=None):
    return np.mean([cv_ba(D, X, lab_w, s, cov) for s in rf.SEEDS], axis=0)          # (BA_A, BA_X)


def hedges(v, y):
    a, b = v[y == 1], v[y == 0]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(v) - 2))
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * len(v) - 9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/csafe_ijdar_2026-11-12")
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(); out = ROOT / a.out
    proto = json.loads((out / "protocol.json").read_text()); print("protocol:", proto["title"], flush=True)
    D = CsafeData(out); P = Parallel(a.jobs)
    print("writers", dict(zip(*np.unique(D.wlabels, return_counts=True))), "pages", len(D.y), flush=True)

    jobs = [(r, "svm", s, ("A", "X", "C2", "C3")) for r in REPS for s in rf.SEEDS]
    jobs += [(r, c, s, ("A", "X")) for r in REPS for c in ("logistic", "random_forest") for s in rf.SEEDS]
    par = pd.concat(P(delayed(run_cv_x)(D, r, c, s, an) for r, c, s, an in jobs), ignore_index=True)
    wl_ = rf.writer_level(par); S = rf.summarize(rf.metrics(wl_)); S.to_csv(out / "csafe_metrics_summary.csv", index=False)
    g = lambda r, an, clf="svm", col="ba_mean": S[(S.rep == r) & (S.clf == clf) & (S.analysis == an)][col].iloc[0]

    lab = pd.Series(D.wlabels, D.writers); rng = np.random.default_rng(2029)
    perms = [pd.Series(rng.permutation(lab.to_numpy()), lab.index) for _ in range(a.n_perm)]
    cov = np.column_stack([D.ink[:, 0], D.X["global_intensity"]])
    rows = []
    for r in REPS:
        X = D.X[r]; row = dict(rep=r, dims=X.shape[1])
        obs = mean_ba(D, X, lab)
        null = np.array(P(delayed(mean_ba)(D, X, pl) for pl in perms))
        adj = mean_ba(D, X, lab, cov) if r not in CTRL else (np.nan, np.nan)
        for j, an in enumerate(("A", "X")):
            wa = wl_[(wl_.rep == r) & (wl_.clf == "svm") & (wl_.analysis == an)]
            est, lo, hi = rv.boot_ci(wa)
            row.update({f"{an}": g(r, an), f"{an}_sd": g(r, an, col="ba_sd"), f"{an}_lo": lo, f"{an}_hi": hi,
                        f"{an}_auc": g(r, an, col="auc_mean"), f"{an}_LR": g(r, an, "logistic"), f"{an}_RF": g(r, an, "random_forest"),
                        f"{an}_perm_obs": obs[j], f"{an}_p": (np.sum(null[:, j] >= obs[j]) + 1) / (a.n_perm + 1),
                        f"{an}_adjusted": adj[j]})
        row.update(C2=g(r, "C2"), C3=g(r, "C3"))
        w = P(delayed(pf.svm_A_writer_ba)(X, D.y, D.w, D.para, s) for s in rf.SEEDS)
        lk = P(delayed(pf.leaky_ba)(X, D.y, D.para, s) for s in rf.SEEDS)
        row.update(para_ba_writer=np.mean([x[1] for x in w]), para_ba_leaky=np.mean(lk), writer_id_top1=pf.writer_id_top1(X, D.w, D.para))
        rows.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        T = pd.DataFrame(rows)
        if {"hc183_tb"} <= set(T.rep):
            pr = T.set_index("rep").loc["hc183_tb"]
            holm = multipletests([pr.A_p, pr.X_p], method="holm")[1]
            T["H1_holm_p"], T["H2_holm_p"] = holm[0], holm[1]
        T.to_csv(out / "csafe_table.csv", index=False)

    # layout / acquisition measures between groups (writer means, session-1 pages), 10,000 permutations, BH
    rng = np.random.default_rng(2031); trows = []
    df = pd.DataFrame(D.X["layout"], columns=D.layout_names).assign(ink_page=D.ink[:, 0], ink_textblock=D.ink[:, 1],
                                                                    w=D.w, y=D.y)[D.para <= 6]
    wm = df.groupby("w").mean(); yv = wm.y.to_numpy().astype(int)
    for n in [c for c in wm.columns if c not in ("y",)]:
        v = wm[n].to_numpy()
        if np.nanstd(v) == 0:
            continue
        o = hedges(v, yv); null = np.array([hedges(v, rng.permutation(yv)) for _ in range(10000)])
        trows.append(dict(measure=n, left_mean=v[yv == 1].mean(), right_mean=v[yv == 0].mean(), hedges_g=o,
                          p_perm=(np.sum(np.abs(null) >= abs(o)) + 1) / 10001))
    L = pd.DataFrame(trows); L["q_bh"] = multipletests(L.p_perm, method="fdr_bh")[1]
    L.to_csv(out / "csafe_layout_tests.csv", index=False); print(L.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
