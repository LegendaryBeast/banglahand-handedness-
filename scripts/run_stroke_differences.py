#!/usr/bin/env python3
"""Amendment 4 (post hoc, descriptive): which STROKE bins differ between left- and right-handed writers."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stroke_features as sf  # noqa: E402
import run_final_experiments as rf  # noqa: E402

BLOCKS = {"direction": slice(0, 16), "hinge": slice(16, 160), "turning": slice(160, 172), "endings": slice(172, 184),
          "orientation": slice(184, 196), "slant": slice(196, 205), "rates": slice(205, 208)}


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty_like(p); prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, p[i] * len(p) / rank); q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    run = ROOT / ap.parse_args().run
    D = rf.Data(run)
    X = D.X["stroke"]; sh = D.para <= 6
    W = pd.DataFrame(X[sh], columns=sf.NAMES).assign(w=D.w[sh], y=D.y[sh]).groupby("w").mean()
    y = W.pop("y").to_numpy().astype(int); V = W.to_numpy()
    n1, n0 = (y == 1).sum(), (y == 0).sum()

    def g_of(lab):
        a, b = V[lab == 1], V[lab == 0]
        sp = np.sqrt(((n1 - 1) * a.var(0, ddof=1) + (n0 - 1) * b.var(0, ddof=1)) / (n1 + n0 - 2)) + 1e-12
        return (a.mean(0) - b.mean(0)) / sp * (1 - 3 / (4 * (n1 + n0) - 9))
    g = g_of(y)
    rng = np.random.default_rng(2034)
    null = np.array([g_of(rng.permutation(y)) for _ in range(10000)])
    p = (np.sum(np.abs(null) >= np.abs(g), 0) + 1) / 10001
    out = pd.DataFrame(dict(feature=sf.NAMES, hedges_g=g, p_perm=p, left_mean=V[y == 1].mean(0), right_mean=V[y == 0].mean(0)))
    out["block"] = ""; out["q_bh_block"] = np.nan
    for b, sl in BLOCKS.items():
        out.loc[out.index[sl], "block"] = b
        out.loc[out.index[sl], "q_bh_block"] = bh(out.p_perm.to_numpy()[sl])
    # fold-averaged standardised SVM weights (reference seed)
    fold = dict(zip(D.writers, rf.writer_folds(D.writers, D.wlabels, rf.REF_SEED)))
    fw = np.array([fold[u] for u in D.w]); coefs = []
    for f in range(5):
        tr = sh & (fw != f); m = rf.estimator("svm"); m.fit(X[tr], D.y[tr]); coefs.append(m[-1].coef_.ravel())
    out["svm_weight_mean"] = np.mean(coefs, 0)
    out.to_csv(run / "stroke_bin_differences.csv", index=False)
    s = out.groupby("block").agg(n=("feature", "size"), n_p05=("p_perm", lambda v: int((v < .05).sum())),
                                 n_q10=("q_bh_block", lambda v: int((v < .10).sum())), max_abs_g=("hedges_g", lambda v: float(np.abs(v).max())))
    s.to_csv(run / "stroke_bin_differences_summary.csv"); print(s)
    print(out.sort_values("p_perm").head(12)[["feature", "hedges_g", "p_perm", "q_bh_block"]].round(4).to_string())


if __name__ == "__main__":
    main()
