#!/usr/bin/env python3
"""Writer-independent experiments for the 38-writer study (see protocol_declared_before_fitting.json).

Reads   results/<run>/features/features.npz and descriptors.csv (from extract_final_features.py)
Writes  results/<run>/*.csv and summary.json. Models are refitted from cached features; images
        are not re-read. All fitting, scaling and tuning use training writers only.
"""
import argparse
import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import final_study_pipeline as study  # noqa: E402

SEEDS = list(range(2026, 2046))
if os.environ.get("HANDEDNESS_SMOKE_SEEDS"):          # functional tests only; the protocol uses 20
    SEEDS = SEEDS[:int(os.environ["HANDEDNESS_SMOKE_SEEDS"])]
REF_SEED = 2026
INNER_SEED = 2027
FITS = {"A": [1, 2, 3, 4, 5, 6], "B": [7, 8], "E_S1": [1, 2], "E_S2": [3, 4], "E_S3": [5, 6],
        "ALL8": [1, 2, 3, 4, 5, 6, 7, 8]}
TESTS = {"A": ("A", FITS["A"]), "B": ("B", [7, 8]), "C": ("A", [7, 8]),
         "D_S1": ("A", [1, 2]), "D_S2": ("A", [3, 4]), "D_S3": ("A", [5, 6]),
         "E_S1": ("E_S1", [1, 2]), "E_S2": ("E_S2", [3, 4]), "E_S3": ("E_S3", [5, 6]),
         "ALL8": ("ALL8", FITS["ALL8"])}
DEEP = ["resnet18", "efficientnet_b0", "vit_b_16"]
MAIN_REPS = ["hc183"] + DEEP + ["style6"]
STROKE_REPS = ["stroke", "stroke_cnn", "combined"]   # added for the 37-writer run (used when present in features.npz)
STROKE_BLOCKS = {"stroke_direction": slice(0, 16), "stroke_hinge": slice(16, 160), "stroke_turning": slice(160, 172),
                 "stroke_endings": slice(172, 184), "stroke_orient_slant": slice(184, 205), "stroke_rates": slice(205, 208)}
CONTROLS = ["global_intensity", "background"]
CLFS = ["svm", "logistic", "random_forest"]
STYLE6 = ["slant_deg", "stroke_width_px1536", "ink_density", "horizontal_energy", "vertical_energy", "row_profile_cv"]


def estimator(name, p=None):
    p = p or {}
    if name == "svm":
        m = SVC(kernel="linear", C=0.1)
    elif name == "logistic":
        m = LogisticRegression(C=p.get("C", 0.1), max_iter=5000, random_state=2026)
    else:
        m = RandomForestClassifier(n_estimators=100, max_depth=p.get("max_depth", 5), random_state=2026, n_jobs=1)
    return make_pipeline(StandardScaler(), m)


def scores(model, name, X):
    assert list(model.classes_) == [0, 1]
    return model.decision_function(X) if name == "svm" else model.predict_proba(X)[:, 1]


THR = {"svm": 0.0, "logistic": 0.5, "random_forest": 0.5}


def writer_folds(writers, labels, seed, n=5):
    """writers sorted; labels aligned; returns array of fold ids (StratifiedKFold over writers)."""
    fold = np.empty(len(writers), int)
    for f, (_, te) in enumerate(StratifiedKFold(n, shuffle=True, random_state=seed).split(writers, labels)):
        fold[te] = f
    return fold


def fit(name, X, y, w):
    """Fit with inner writer-CV selection (logistic / RF); SVM is fixed."""
    if name == "svm":
        m = estimator(name); m.fit(X, y); return m, {}
    grid = [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]] if name == "logistic" else [{"max_depth": d} for d in [3, 5, None]]
    uw = np.unique(w)
    ul = np.array([y[w == u][0] for u in uw])
    inner = writer_folds(uw, ul, INNER_SEED, 4)
    best, best_s = grid[0], -1
    for p in grid:
        s_all, y_all, w_all = [], [], []
        for f in range(4):
            te_w = set(uw[inner == f])
            te = np.array([x in te_w for x in w])
            m = estimator(name, p); m.fit(X[~te], y[~te])
            s_all.append(scores(m, name, X[te])); y_all.append(y[te]); w_all.append(w[te])
        d = pd.DataFrame(dict(s=np.concatenate(s_all), y=np.concatenate(y_all), w=np.concatenate(w_all)))
        g = d.groupby("w").agg(s=("s", "mean"), y=("y", "first"))
        ba = balanced_accuracy_score(g.y, (g.s >= THR[name]).astype(int))
        if ba > best_s:
            best, best_s = p, ba
    m = estimator(name, best); m.fit(X, y)
    return m, best


class Data:
    def __init__(self, run):
        self.meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
        z = np.load(run / "features/features.npz", allow_pickle=True)
        assert list(z["image_id"]) == list(self.meta.image_id)
        self.X = {k: z[k].astype(np.float64) for k in z.files if k != "image_id"}
        if "stroke" in self.X:
            for name, sl in STROKE_BLOCKS.items():
                self.X[name] = self.X["stroke"][:, sl]
        d = pd.read_csv(run / "features/descriptors.csv").set_index("image_id").loc[self.meta.image_id]
        self.desc = d.reset_index()
        self.X["style6"] = d[STYLE6].to_numpy(float)
        if "stroke" in self.X:
            self.X["combined"] = np.hstack([self.X["hc183"], self.X["style6"], self.X["stroke"]])
        hc = self.X["hc183"]
        cells = np.arange(160).reshape(16, 10)
        self.X["hc_orientation"] = hc[:, cells[:, :9].ravel()]
        self.X["hc_density"] = hc[:, cells[:, 9]]
        self.X["hc_projection"] = hc[:, 160:178]
        self.X["hc_components"] = hc[:, 178:183]
        self.y = self.meta.label.to_numpy()
        self.w = self.meta.writer_id.to_numpy()
        self.para = self.meta.paragraph_number.to_numpy()
        self.trunc = self.meta.text_truncated_at_edge.fillna("").ne("").to_numpy()
        wt = self.meta.groupby("writer_id").label.first().sort_index()
        self.writers, self.wlabels = wt.index.to_numpy(), wt.to_numpy()


def run_cv(D, rep, clf, seed, analyses=tuple(TESTS), writers=None, image_ok=None, labels=None, train_frac=1.0):
    """One repetition of 5-fold writer CV; returns paragraph predictions."""
    X = D.X[rep]
    wr = D.writers if writers is None else np.array(sorted(writers))
    wl = (pd.Series(D.wlabels, D.writers) if labels is None else labels).loc[wr].to_numpy()
    lab_img = pd.Series(wl, wr).reindex(D.w).to_numpy()
    in_cohort = np.isin(D.w, wr) & (np.ones(len(D.w), bool) if image_ok is None else image_ok)
    fold_of = dict(zip(wr, writer_folds(wr, wl, seed)))
    fits_needed = sorted({TESTS[a][0] for a in analyses})
    rows = []
    for f in range(5):
        test_w = {u for u, k in fold_of.items() if k == f}
        train_w = np.array([u for u in wr if u not in test_w])
        if train_frac < 1.0:
            tl = pd.Series(wl, wr).loc[train_w].to_numpy()
            rng = np.random.default_rng(seed * 100 + f)
            keep = []
            for c in (0, 1):
                cand = train_w[tl == c]
                keep += list(rng.choice(cand, max(2, int(round(train_frac * len(cand)))), replace=False))
            train_w = np.array(sorted(keep))
        tr_base = in_cohort & np.isin(D.w, train_w)
        te_base = in_cohort & np.isin(D.w, list(test_w))
        for fk in fits_needed:
            tr = tr_base & np.isin(D.para, FITS[fk])
            model, params = fit(clf, X[tr], lab_img[tr], D.w[tr])
            for a in analyses:
                if TESTS[a][0] != fk:
                    continue
                te = te_base & np.isin(D.para, TESTS[a][1])
                s = scores(model, clf, X[te])
                rows.append(pd.DataFrame(dict(rep=rep, clf=clf, seed=seed, analysis=a, fold=f,
                                              image_id=D.meta.image_id.to_numpy()[te], writer_id=D.w[te],
                                              label=lab_img[te], para=D.para[te], score=s,
                                              params=json.dumps(params))))
    return pd.concat(rows, ignore_index=True)


def writer_level(par):
    g = (par.groupby(["rep", "clf", "seed", "analysis", "writer_id"])
         .agg(label=("label", "first"), score=("score", "mean"), fold=("fold", "first"), n=("score", "size"))
         .reset_index())
    g["pred"] = (g.score >= g.clf.map(THR)).astype(int)
    return g


def metrics(wr):
    out = []
    for k, g in wr.groupby(["rep", "clf", "seed", "analysis"]):
        aucs, wts = [], []
        for _, h in g.groupby("fold"):
            if h.label.nunique() == 2:
                aucs.append(roc_auc_score(h.label, h.score)); wts.append((h.label == 0).sum() * (h.label == 1).sum())
        out.append(dict(zip(["rep", "clf", "seed", "analysis"], k), n_writers=len(g), n_correct=int((g.pred == g.label).sum()),
                        ba=balanced_accuracy_score(g.label, g.pred), acc=accuracy_score(g.label, g.pred),
                        f1=f1_score(g.label, g.pred, average="macro"),
                        recall_right=float(((g.pred == 0) & (g.label == 0)).sum() / (g.label == 0).sum()),
                        recall_left=float(((g.pred == 1) & (g.label == 1)).sum() / (g.label == 1).sum()),
                        auc=float(np.average(aucs, weights=wts)) if aucs else np.nan))
    return pd.DataFrame(out)


def summarize(m, keys=("rep", "clf", "analysis")):
    return (m.groupby(list(keys)).agg(ba_mean=("ba", "mean"), ba_sd=("ba", "std"), ba_min=("ba", "min"), ba_max=("ba", "max"),
                                      auc_mean=("auc", "mean"), f1_mean=("f1", "mean"), recall_right=("recall_right", "mean"),
                                      recall_left=("recall_left", "mean"), n_writers=("n_writers", "first"), n_splits=("ba", "size"))
            .reset_index())


def evidence_curve(par, draws=200, rng_seed=7):
    """Expected writer BA when a test writer contributes k random shared paragraphs (A models)."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for (rep, clf, seed), g in par[par.analysis == "A"].groupby(["rep", "clf", "seed"]):
        thr = THR[clf]
        per = {w: (h.score.to_numpy(), h.label.iloc[0]) for w, h in g.groupby("writer_id")}
        ws = list(per)
        y = np.array([per[w][1] for w in ws])
        for k in range(1, 7):
            bas = []
            for _ in range(draws):
                pred = np.array([int(rng.choice(per[w][0], min(k, len(per[w][0])), replace=False).mean() >= thr) for w in ws])
                bas.append(balanced_accuracy_score(y, pred))
            rows.append(dict(rep=rep, clf=clf, seed=seed, k=k, ba=float(np.mean(bas))))
    return pd.DataFrame(rows)


def perm_stat(D, rep, labels):
    """Repeated-CV mean BA of the fixed SVM on analysis A with the given writer labels."""
    X, bas = D.X[rep], []
    sh = np.isin(D.para, FITS["A"])
    wl = labels.loc[D.writers].to_numpy()
    lab_img = labels.reindex(D.w).to_numpy()
    for seed in SEEDS:
        fold = dict(zip(D.writers, writer_folds(D.writers, wl, seed)))
        fw = np.array([fold[u] for u in D.w])
        sc = np.empty(len(D.w))
        for f in range(5):
            tr, te = sh & (fw != f), sh & (fw == f)
            m = estimator("svm"); m.fit(X[tr], lab_img[tr]); sc[te] = m.decision_function(X[te])
        g = pd.DataFrame(dict(w=D.w[sh], s=sc[sh], y=lab_img[sh])).groupby("w").agg(s=("s", "mean"), y=("y", "first"))
        bas.append(balanced_accuracy_score(g.y, (g.s >= 0).astype(int)))
    return float(np.mean(bas)), bas[0]


def descriptor_tests(D, n_perm=10000, seed=2031):
    d = D.desc.copy()
    d["writer_id"], d["label"], d["para"] = D.w, D.y, D.para
    cols = STYLE6 + [c for c in d.columns if c.startswith("orient_")]
    wm = d[d.para <= 6].groupby("writer_id")[cols + ["label"]].mean()
    y = wm.label.to_numpy().astype(int)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(y) for _ in range(n_perm)]
    rows = []
    for c in cols:
        v = wm[c].to_numpy()
        diff = v[y == 1].mean() - v[y == 0].mean()
        sp = np.sqrt(((y == 1).sum() - 1) * v[y == 1].var(ddof=1) + ((y == 0).sum() - 1) * v[y == 0].var(ddof=1)) / np.sqrt(len(y) - 2)
        n = len(y); g = diff / sp * (1 - 3 / (4 * n - 9))
        null = np.array([v[p == 1].mean() - v[p == 0].mean() for p in perms])
        p = (np.sum(np.abs(null) >= abs(diff)) + 1) / (n_perm + 1)
        rows.append(dict(descriptor=c, left_mean=v[y == 1].mean(), right_mean=v[y == 0].mean(), difference=diff, hedges_g=g, p_perm=p))
    t = pd.DataFrame(rows)
    order = np.argsort(t.p_perm.to_numpy()); m = len(t)
    q = np.empty(m); prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, t.p_perm.iloc[i] * m / rank); q[i] = prev
    t["q_bh"] = q
    return t, wm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--n-seeds", type=int, default=20, help="smoke tests only; the declared protocol uses 20")
    a = ap.parse_args()
    global SEEDS
    SEEDS = SEEDS[:a.n_seeds]
    run = ROOT / a.run
    D = Data(run)
    global MAIN_REPS
    if "stroke" in D.X:
        MAIN_REPS = MAIN_REPS + STROKE_REPS
    t0 = time.time()
    P = Parallel(n_jobs=a.jobs, verbose=0)

    # 1. Main grid: representations x classifiers x analyses x 20 repetitions
    tasks = [(r, c, s) for r in MAIN_REPS for c in CLFS for s in SEEDS]
    par = pd.concat(P(delayed(run_cv)(D, r, c, s) for r, c, s in tasks), ignore_index=True)
    par.to_csv(run / "paragraph_predictions.csv.gz", index=False)
    wr = writer_level(par)
    wr.to_csv(run / "writer_predictions.csv.gz", index=False)
    m = metrics(wr)
    m.to_csv(run / "metrics_by_split.csv", index=False)
    summarize(m).to_csv(run / "metrics_summary.csv", index=False)
    print("grid done", round(time.time() - t0), "s", flush=True)

    # 2. Reference split detail and conditional bootstrap (reuse tested pipeline code)
    ref = wr[wr.seed == REF_SEED].rename(columns={"clf": "model", "rep": "feature_set", "analysis": "analysis_id", "pred": "prediction"})
    ref_rows = []
    for fs, g in ref.groupby("feature_set"):
        s = study.summarize_writer_predictions(g[g.analysis_id.isin(list(TESTS))], bootstrap_resamples=10000, seed=2028)
        ref_rows.append(s)
    pd.concat(ref_rows).to_csv(run / "reference_split_metrics.csv", index=False)
    pc = study.paired_comparisons(ref[ref.analysis_id != "ALL8"], bootstrap_resamples=10000, seed=2028)
    pc.to_csv(run / "reference_split_paired_contrasts.csv", index=False)

    # 3. Contrasts across all repetitions
    piv = m.pivot_table(index=["rep", "clf", "seed"], columns="analysis", values="ba")
    con = pd.DataFrame(dict(transfer_C_minus_D=piv["C"] - piv[["D_S1", "D_S2", "D_S3"]].mean(axis=1),
                            varied_B_minus_E=piv["B"] - piv[["E_S1", "E_S2", "E_S3"]].mean(axis=1),
                            B_minus_A=piv["B"] - piv["A"])).reset_index()
    con.to_csv(run / "contrasts_by_split.csv", index=False)

    # 4. Evidence quantity (k shared paragraphs per test writer)
    ev = evidence_curve(par)
    ev.to_csv(run / "evidence_curve.csv", index=False)

    # 5. Controls, ablations, sensitivity, learning curve (analysis A [+C], all repetitions)
    extra = []
    jobs = [("control", r, c, s, None, None, 1.0, ("A",)) for r in CONTROLS for c in CLFS for s in SEEDS]
    jobs += [("ablation", r, "svm", s, None, None, 1.0, ("A",)) for r in ["hc_orientation", "hc_density", "hc_projection", "hc_components"]
             + ([*STROKE_BLOCKS] if "stroke" in D.X else []) for s in SEEDS]
    jobs += [("S2_no_truncated_images", r, "svm", s, None, ~D.trunc, 1.0, ("A", "C")) for r in MAIN_REPS for s in SEEDS]
    jobs += [(f"learning_{fr:.2f}", r, "svm", s, None, None, fr, ("A",)) for fr in [0.25, 0.5, 0.75] for r in MAIN_REPS for s in SEEDS]

    def job(tag, r, c, s, w, ok, fr, an):
        out = run_cv(D, r, c, s, analyses=an, writers=w, image_ok=ok, train_frac=fr)
        out["study"] = tag
        return out
    ex = pd.concat(P(delayed(job)(*j) for j in jobs), ignore_index=True)
    exw = writer_level(ex.drop(columns="study").assign(rep=ex.study + "|" + ex.rep))
    exm = metrics(exw)
    exm[["study", "rep"]] = exm.rep.str.split("|", n=1, expand=True)
    exm.to_csv(run / "secondary_metrics_by_split.csv", index=False)
    summarize(exm, ("study", "rep", "clf", "analysis")).to_csv(run / "secondary_metrics_summary.csv", index=False)
    print("secondary done", round(time.time() - t0), "s", flush=True)

    # 6. Permutation tests (fixed SVM, analysis A, statistic = repeated-CV mean BA)
    obs_labels = pd.Series(D.wlabels, D.writers)
    perm_rows = []
    rng = np.random.default_rng(2029)
    draws = [pd.Series(rng.permutation(D.wlabels), D.writers) for _ in range(a.n_perm)]
    for r in MAIN_REPS + CONTROLS:
        obs, obs_ref = perm_stat(D, r, obs_labels)
        null = P(delayed(perm_stat)(D, r, lab) for lab in draws)
        nm = np.array([x[0] for x in null]); nr = np.array([x[1] for x in null])
        perm_rows.append(dict(rep=r, observed_mean_ba=obs, observed_ref_split_ba=obs_ref, n_perm=a.n_perm,
                              p_mean=(np.sum(nm >= obs) + 1) / (a.n_perm + 1),
                              p_ref_split=(np.sum(nr >= obs_ref) + 1) / (a.n_perm + 1),
                              null_mean=nm.mean(), null_q95=np.quantile(nm, 0.95)))
        pd.DataFrame(dict(null_mean_ba=nm, null_ref_ba=nr)).to_csv(run / f"permutation_null_{r}.csv", index=False)
        print("perm", r, perm_rows[-1], flush=True)
    pd.DataFrame(perm_rows).to_csv(run / "permutation_tests.csv", index=False)

    # 7. Interpretable descriptors
    t, wm = descriptor_tests(D)
    t.to_csv(run / "descriptor_tests.csv", index=False)
    wm.to_csv(run / "descriptor_writer_means.csv")

    (run / "run_provenance.json").write_text(json.dumps(dict(
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
        seeds=SEEDS, ref_seed=REF_SEED, n_perm=a.n_perm, runtime_s=round(time.time() - t0),
        script_sha256=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    print("all done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
