#!/usr/bin/env python3
"""Secondary analyses of the PRL study recomputed on the line-free text blocks (IJDAR; printed border removed).

Same library functions, seeds and settings as the PRL pipeline, applied to the line-free features:
  run_final_experiments.run_cv / writer_level / metrics / evidence_curve / perm_stat, the STROKE block ablation and
  learning curves (run_final_experiments step 5), run_secondary_analyses.mirror_cv, run_revision_analyses.mirror_aug /
  mirror_effect, and the STROKE bin differences of run_stroke_differences.py.
Representations: hc183_lf, stroke_lf, resnet18_lf, efficientnet_b0_lf, vit_b_16_lf (run_linefree_primary.py inputs).
Checks: the condition means and permutation p-values must equal results/ijdar_dev_2026-11-12/linefree_table_rows*.csv.
Writes <dev>/linefree_secondary/: metrics_by_split.csv, writer_predictions.csv.gz, evidence_curve.csv,
secondary_metrics_by_split.csv, mirror_test_by_split.csv, mirror_augmented_stroke.csv, mirror_effect_permutation.csv,
stroke_bin_differences.csv, writer_correctness_spearman.csv, permutation_null_<rep>.csv, permutation_tests.csv.
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
import run_revision_analyses as rv  # noqa: E402
import run_secondary_analyses as rs  # noqa: E402
import stroke_features as sf  # noqa: E402

REPS = ["hc183_lf", "stroke_lf", "resnet18_lf", "efficientnet_b0_lf", "vit_b_16_lf"]
BLOCKS = {"direction": slice(0, 16), "hinge": slice(16, 160), "turning": slice(160, 172), "endings": slice(172, 184),
          "orientation": slice(184, 196), "slant": slice(196, 205), "rates": slice(205, 208)}      # run_stroke_differences


def load(base, dev):
    D = rf.Data(ROOT / base); dev = ROOT / dev
    z = np.load(dev / "features/layout.npz", allow_pickle=True); assert list(z["image_id"]) == list(D.meta.image_id)
    D.X["hc183_lf"] = z["hc183_textblock"].astype(float)
    e = np.load(dev / "features/lineless_embeddings.npz", allow_pickle=True)
    for k in ("resnet18", "efficientnet_b0", "vit_b_16"):
        D.X[k + "_lf"] = e[k].astype(float)
    for f, k in (("stroke_linefree.npz", "stroke_lf"), ("stroke_linefree_mirror.npz", "stroke_lf_mirror")):
        s = np.load(dev / "features" / f, allow_pickle=True); assert list(s["image_id"]) == list(D.meta.image_id)
        D.X[k] = s[k].astype(float)
    for name, sl in rf.STROKE_BLOCKS.items():                                    # line-free STROKE blocks
        D.X[name.replace("stroke_", "stroke_lf_")] = D.X["stroke_lf"][:, sl]
    return D


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty_like(p); prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, p[i] * len(p) / rank); q[i] = prev
    return q


def stroke_differences(D, X):
    """run_stroke_differences.main on the given STROKE matrix (same statistics, seed 2034)."""
    sh = D.para <= 6
    W = pd.DataFrame(X[sh], columns=sf.NAMES).assign(w=D.w[sh], y=D.y[sh]).groupby("w").mean()
    y = W.pop("y").to_numpy().astype(int); V = W.to_numpy(); n1, n0 = (y == 1).sum(), (y == 0).sum()

    def g_of(lab):
        a, b = V[lab == 1], V[lab == 0]
        sp = np.sqrt(((n1 - 1) * a.var(0, ddof=1) + (n0 - 1) * b.var(0, ddof=1)) / (n1 + n0 - 2)) + 1e-12
        return (a.mean(0) - b.mean(0)) / sp * (1 - 3 / (4 * (n1 + n0) - 9))
    g = g_of(y); rng = np.random.default_rng(2034)
    null = np.array([g_of(rng.permutation(y)) for _ in range(10000)])
    p = (np.sum(np.abs(null) >= np.abs(g), 0) + 1) / 10001
    out = pd.DataFrame(dict(feature=sf.NAMES, hedges_g=g, p_perm=p, left_mean=V[y == 1].mean(0), right_mean=V[y == 0].mean(0)))
    out["block"] = ""; out["q_bh_block"] = np.nan
    for b, sl in BLOCKS.items():
        out.loc[out.index[sl], "block"] = b; out.loc[out.index[sl], "q_bh_block"] = bh(out.p_perm.to_numpy()[sl])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/final42_2026-10-19")
    ap.add_argument("--dev", default="results/ijdar_dev_2026-11-12")
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(); out = ROOT / a.dev / "linefree_secondary"; out.mkdir(exist_ok=True)
    D = load(a.base, a.dev); P = Parallel(a.jobs)

    # 1. per-repetition CV (all text conditions, SVM), writer predictions, evidence curve
    par = pd.concat(P(delayed(rf.run_cv)(D, r, "svm", s, tuple(rf.TESTS)) for r in REPS for s in rf.SEEDS), ignore_index=True)
    wl = rf.writer_level(par); m = rf.metrics(wl)
    m.to_csv(out / "metrics_by_split.csv", index=False); wl.to_csv(out / "writer_predictions.csv.gz", index=False)
    rf.evidence_curve(par).to_csv(out / "evidence_curve.csv", index=False)
    S = rf.summarize(m)
    R = pd.concat([pd.read_csv(f) for f in sorted((ROOT / a.dev).glob("linefree_table_rows*.csv"))]).drop_duplicates("rep", keep="last").set_index("rep")
    for r in REPS:
        for an, col in (("A", "A"), ("C", "C"), ("B", "B"), ("ALL8", "ALL8")):
            v = S[(S.rep == r) & (S.analysis == an)].ba_mean.iloc[0]
            assert abs(v - R.loc[r, col]) < 1e-9, (r, an, v, R.loc[r, col])
    print("check 1 passed: condition means equal linefree_table_rows", flush=True)

    # 2. STROKE block ablation and learning curves (as run_final_experiments step 5)
    def job(tag, r, s, fr):
        o = rf.run_cv(D, r, "svm", s, analyses=("A",), train_frac=fr); o["study"] = tag; return o
    jobs = [("ablation", b, s, 1.0) for b in [k.replace("stroke_", "stroke_lf_") for k in rf.STROKE_BLOCKS] for s in rf.SEEDS]
    jobs += [(f"learning_{fr:.2f}", r, s, fr) for fr in (0.25, 0.5, 0.75) for r in REPS for s in rf.SEEDS]
    ex = pd.concat(P(delayed(job)(*j) for j in jobs), ignore_index=True)
    exm = rf.metrics(rf.writer_level(ex.drop(columns="study").assign(rep=ex.study + "|" + ex.rep)))
    exm[["study", "rep"]] = exm.rep.str.split("|", n=1, expand=True)
    exm.to_csv(out / "secondary_metrics_by_split.csv", index=False)
    rf.summarize(exm, ("study", "rep", "clf", "analysis")).to_csv(out / "secondary_metrics_summary.csv", index=False)

    # 3. mirror test, mirror-augmented STROKE, permutation test of the mirror effect (line-free STROKE)
    M = rf.Data(ROOT / a.base); M.X = {"stroke": D.X["stroke_lf"], "stroke_mirror": D.X["stroke_lf_mirror"]}
    pd.DataFrame(P(delayed(rs.mirror_cv)(M, s) for s in rf.SEEDS)).to_csv(out / "mirror_test_by_split.csv", index=False)
    r_ = P(delayed(rv.mirror_aug)(M, s, True) for s in rf.SEEDS)
    pd.DataFrame([dict(seed=s, ba_original=o, ba_mirrored=mm) for s, (o, mm) in zip(rf.SEEDS, r_)]).to_csv(out / "mirror_augmented_stroke.csv", index=False)
    rng = np.random.default_rng(2029)
    perms = [pd.Series(rng.permutation(D.wlabels), D.writers) for _ in range(a.n_perm)]
    obs = rv.mirror_effect(M, M.y.copy())
    null = np.array(P(delayed(rv.mirror_effect)(M, rv.img_labels(M, pl)) for pl in perms))
    pd.DataFrame([dict(statistic="BA_original_minus_BA_mirrored", observed=obs, p=(np.sum(null >= obs) + 1) / (a.n_perm + 1),
                       null_mean=null.mean(), n_perm=a.n_perm)]).to_csv(out / "mirror_effect_permutation.csv", index=False)
    print("mirror effect", round(obs, 4), flush=True)

    # 4. STROKE bin differences and per-writer correctness
    stroke_differences(D, D.X["stroke_lf"]).to_csv(out / "stroke_bin_differences.csv", index=False)
    d = wl[(wl.clf == "svm") & (wl.analysis == "A")]
    acc = d.assign(c=(d.pred == d.label)).groupby(["writer_id", "rep"]).c.mean().unstack()[REPS]
    acc.corr(method="spearman").to_csv(out / "writer_correctness_spearman.csv")

    # 5. permutation nulls of every line-free representation (same rng 2029 relabellings as the primary run)
    lab = pd.Series(D.wlabels, D.writers); rows = []
    for r in REPS:
        o = rf.perm_stat(D, r, lab)[0]
        nul = np.array(P(delayed(rf.perm_stat)(D, r, pl) for pl in perms))[:, 0]
        p = (np.sum(nul >= o) + 1) / (a.n_perm + 1)
        assert abs(p - R.loc[r, "p"]) < 1e-12, (r, p, R.loc[r, "p"])
        pd.DataFrame(dict(null_mean_ba=nul)).to_csv(out / f"permutation_null_{r}.csv", index=False)
        rows.append(dict(rep=r, observed_mean_ba=o, p_mean=p)); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "permutation_tests.csv", index=False)
    print("check 2 passed: permutation p-values equal linefree_table_rows", flush=True)


if __name__ == "__main__":
    main()
