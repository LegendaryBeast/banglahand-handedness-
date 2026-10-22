#!/usr/bin/env python3
"""Write every number used in the PRL text as a LaTeX macro, read directly from result files.

Usage: make_prl_macros.py --run results/final42_2026-10-19 --output manuscripts/prl/figures/results_macros.tex
The manuscript uses only these macros for quantitative statements, so text and results cannot diverge.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEEP = ["resnet18", "efficientnet_b0", "vit_b_16"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--output", required=True)
    a = ap.parse_args(); R = ROOT / a.run
    M = {}
    pc = lambda v: f"{100 * float(v):.1f}"
    p3 = lambda v: f"{float(v):.3f}"
    rng_ = lambda vals: f"{min(vals)}--{max(vals)}"
    sg = lambda v: ("0.0" if abs(v) < 0.05 else f"+{v:.1f}" if v > 0 else f"$-${abs(v):.1f}")

    img = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    M.update(nWriters=img.writer_id.nunique(), nLeft=img[img.hand == "left"].writer_id.nunique(),
             nRight=img[img.hand == "right"].writer_id.nunique(), nImages=len(img), nShared=int((img.condition == "shared").sum()),
             nSpecific=int((img.condition != "shared").sum()), nTrunc=int(img.text_truncated_at_edge.notna().sum()),
             nTruncLeft=int((img.text_truncated_at_edge.notna() & (img.hand == "left")).sum()),
             nTruncRight=int((img.text_truncated_at_edge.notna() & (img.hand == "right")).sum()))
    M["nTrainWriters"] = f"{img.writer_id.nunique() * 0.8:.0f}"

    sm = pd.read_csv(R / "metrics_summary.csv"); mb = pd.read_csv(R / "metrics_by_split.csv")
    g = lambda rep, an="A", clf="svm", col="ba_mean": sm[(sm.rep == rep) & (sm.clf == clf) & (sm.analysis == an)][col].iloc[0]
    key = {"hc183": "Hc", "style6": "Style", "stroke": "Stroke", "combined": "Comb", "stroke_cnn": "Scnn",
           "resnet18": "Res", "efficientnet_b0": "Eff", "vit_b_16": "Vit"}
    for rep, k in key.items():
        for an, ak in [("A", "A"), ("C", "C"), ("B", "B"), ("ALL8", "All")]:
            M[f"{k}{ak}"] = pc(g(rep, an))
        M[f"{k}Asd"] = pc(g(rep, "A", col="ba_sd")); M[f"{k}Auc"] = f"{g(rep, 'A', col='auc_mean'):.2f}"
        M[f"{k}Alog"] = pc(g(rep, "A", "logistic")); M[f"{k}Arf"] = pc(g(rep, "A", "random_forest"))
    v = mb[(mb.rep == "hc183") & (mb.clf == "svm") & (mb.analysis == "A")].ba
    M.update(HcAmin=pc(v.min()), HcAmax=pc(v.max()), HcRecR=pc(g("hc183", col="recall_right")), HcRecL=pc(g("hc183", col="recall_left")))
    v = mb[(mb.rep == "stroke") & (mb.clf == "svm") & (mb.analysis == "A")].ba
    M.update(StrokeAmin=pc(v.min()), StrokeAmax=pc(v.max()))
    M["DeepArange"] = rng_([pc(g(r)) for r in DEEP])
    M["DeepOtherMax"] = pc(max(g(r, "A", c) for r in DEEP for c in ["logistic", "random_forest"]))
    M["CombBlog"] = pc(g("combined", "B", "logistic"))

    perm = pd.read_csv(R / "permutation_tests.csv").set_index("rep")
    for rep, k in list(key.items()) + [("global_intensity", "Photo"), ("background", "Paper")]:
        M[f"{k}P"] = p3(perm.loc[rep, "p_mean"])
    M["PhotoA"] = pc(perm.loc["global_intensity", "observed_mean_ba"]); M["PaperA"] = pc(perm.loc["background", "observed_mean_ba"])
    M["HcNull"] = pc(perm.loc["hc183", "null_mean"])

    ref = pd.read_csv(R / "reference_split_metrics.csv")
    for rep, k in [("hc183", "Hc"), ("stroke", "Stroke")]:
        r = ref[(ref.model == "svm") & (ref.feature_set == rep) & (ref.analysis_id == "A")].iloc[0]
        M.update({f"{k}RefCorrect": int(r.n_correct), f"{k}RefBA": pc(r.balanced_accuracy), f"{k}RefLo": pc(r.ci95_low), f"{k}RefHi": pc(r.ci95_high)})

    ss = pd.read_csv(R / "secondary_metrics_summary.csv")
    s2 = lambda study, rep, an="A": ss[(ss.study == study) & (ss.rep == rep) & (ss.analysis == an)].ba_mean.iloc[0]
    for rep, k in key.items():
        M[f"{k}NoTrunc"] = pc(s2("S2_no_truncated_images", rep))
    for b, k in [("stroke_hinge", "BlkHinge"), ("stroke_direction", "BlkDir"), ("stroke_orient_slant", "BlkOrient"), ("stroke_endings", "BlkEnd"),
                 ("stroke_rates", "BlkRates"), ("stroke_turning", "BlkTurn"), ("hc_projection", "BlkHcProj"), ("hc_orientation", "BlkHcOri"),
                 ("hc_density", "BlkHcDens")]:
        M[k] = pc(s2("ablation", b))
    for fr, k in [(0.25, "Q"), (0.5, "H")]:
        for rep, rk in [("hc183", "Hc"), ("stroke", "Stroke"), ("style6", "Style")]:
            M[f"Learn{k}{rk}"] = pc(s2(f"learning_{fr:.2f}", rep))

    ev = pd.read_csv(R / "evidence_curve.csv")
    for rep, k in [("hc183", "Hc"), ("stroke", "Stroke"), ("style6", "Style")]:
        M[f"Ev{k}One"] = pc(ev[(ev.rep == rep) & (ev.clf == "svm") & (ev.k == 1)].ba.mean())

    con = pd.read_csv(R / "contrasts_by_split.csv"); cc = pd.read_csv(R / "controls_contrasts_by_split.csv")
    for rep, k in [("hc183", "Hc"), ("stroke", "Stroke"), ("style6", "Style"), ("combined", "Comb")]:
        c = con[(con.rep == rep) & (con.clf == "svm")]
        M[f"{k}CminusD"] = sg(100 * c.transfer_C_minus_D.mean()); M[f"{k}BminusE"] = sg(100 * c.varied_B_minus_E.mean())
    c = cc[(cc.rep == "background") & (cc.clf == "svm")]; M["PaperCminusD"] = sg(100 * c.transfer_C_minus_D.mean())
    cs = pd.read_csv(R / "controls_metrics_summary.csv")
    M["PaperC"] = pc(cs[(cs.rep == "background") & (cs.clf == "svm") & (cs.analysis == "C")].ba_mean.iloc[0])
    M["HcD"] = pc(np.mean([g("hc183", f"D_S{i}") for i in (1, 2, 3)]))
    sens = pd.read_csv(R / "controls_sensitivity.csv").set_index("rep")
    M.update(PhotoNoTrunc=pc(sens.loc["global_intensity", "S2"]), PaperNoTrunc=pc(sens.loc["background", "S2"]))

    ad = pd.read_csv(R / "adjusted.csv")
    for rep, k in list(key.items()):
        M[f"{k}Adj"] = pc(ad[(ad.rep == rep) & (ad.subset == "all")].adjusted_mean.iloc[0])
    M["DeepAdjRange"] = rng_([pc(ad[(ad.rep == r) & (ad.subset == "all")].adjusted_mean.iloc[0]) for r in DEEP])
    if (R / "adjusted_permutation.csv").is_file():
        ap_ = pd.read_csv(R / "adjusted_permutation.csv")
        for rep, k in [("stroke", "Stroke"), ("combined", "Comb"), ("hc183", "Hc")]:
            for sub, sk in [("all", "All")]:
                r = ap_[(ap_.rep == rep) & (ap_.subset == sub)]
                if len(r):
                    M[f"{k}Adj{sk}P"] = p3(r.p.iloc[0])

    pf = pd.read_csv(R / "pitfalls.csv").set_index("rep")
    M.update(StrokeWid=pc(pf.loc["stroke", "writer_id_top1"]), HcWid=pc(pf.loc["hc183", "writer_id_top1"]),
             DeepWidRange=rng_([pc(pf.loc[r, "writer_id_top1"]) for r in DEEP]),
             StrokeLeaky=pc(pf.loc["stroke", "para_ba_leaky"]), StrokeParaSplit=pc(pf.loc["stroke", "para_ba_writer_split"]),
             DeepLeakyRange=rng_([pc(pf.loc[r, "para_ba_leaky"]) for r in DEEP]),
             DeepParaRange=rng_([pc(pf.loc[r, "para_ba_writer_split"]) for r in DEEP]),
             HcLeaky=pc(pf.loc["hc183", "para_ba_leaky"]), HcParaSplit=pc(pf.loc["hc183", "para_ba_writer_split"]),
             CombWid=pc(pf.loc["combined", "writer_id_top1"]), CombLeaky=pc(pf.loc["combined", "para_ba_leaky"]))
    shift = [100 * (pf.loc[r, "writer_ba_40folders"] - pf.loc[r, "writer_ba"]) for r in key]
    M["DupShiftMin"] = sg(min(shift)); M["DupShiftMax"] = sg(max(shift))

    mt = pd.read_csv(R / "mirror_test_by_split.csv")
    M.update(StrokeMirOrig=pc(mt.ba_original.mean()), StrokeMir=pc(mt.ba_mirrored.mean()), StrokeMirFlip=f"{100 * mt.flip_fraction.mean():.0f}")
    ft = pd.read_csv(R / "finetune_metrics_by_split.csv")
    M.update(FtA=pc(ft.ba_A.mean()), FtAsd=pc(ft.ba_A.std()), FtAmin=pc(ft.ba_A.min()), FtAmax=pc(ft.ba_A.max()), FtC=pc(ft.ba_C.mean()),
             FtMir=pc(ft.ba_A_mirror.mean()), FtFlip=f"{100 * ft.flip_fraction.mean():.0f}")
    if (R / "finetune_mirroraug_metrics_by_split.csv").is_file():
        fa = pd.read_csv(R / "finetune_mirroraug_metrics_by_split.csv")
        M["FtMinusFa"] = f"{100 * (ft.ba_A.mean() - fa.ba_A.mean()):.0f}"
        M.update(FaA=pc(fa.ba_A.mean()), FaAsd=pc(fa.ba_A.std()), FaMir=pc(fa.ba_A_mirror.mean()), FaC=pc(fa.ba_C.mean()),
                 FaFlip=f"{100 * fa.flip_fraction.mean():.0f}")

    dt = pd.read_csv(R / "descriptor_tests.csv").set_index("descriptor")
    for d, k in [("ink_density", "Ink"), ("row_profile_cv", "Row"), ("slant_deg", "Slant")]:
        M[f"{k}G"] = f"{dt.loc[d, 'hedges_g']:.2f}".replace("-", "$-$"); M[f"{k}Pv"] = p3(dt.loc[d, "p_perm"]); M[f"{k}Q"] = p3(dt.loc[d, "q_bh"])
    fc = pd.read_csv(R / "framing_check_tests.csv")
    allk = [s for s in fc.subset.unique() if s.startswith("all")][0]
    f = lambda sub, m, col: fc[(fc.subset == sub) & (fc.measure == m)][col].iloc[0]
    M.update(PaperLevG=f"{f(allk, 'paper_level', 'hedges_g'):.2f}", PaperLevP=p3(f(allk, "paper_level", "p_perm")),
             InBlockG=f"{f(allk, 'ink_density_in_block', 'hedges_g'):.2f}", InBlockP=p3(f(allk, "ink_density_in_block", "p_perm")),
             BlockFracL=f"{100 * f(allk, 'text_block_fraction', 'left_mean'):.0f}", BlockFracR=f"{100 * f(allk, 'text_block_fraction', 'right_mean'):.0f}",
             BlockFracP=p3(f(allk, "text_block_fraction", "p_perm")))
    tt = pd.read_csv(R / "tilt_tests.csv")
    t0 = tt[tt.subset.str.startswith("all") & (tt.measure == "tilt")].iloc[0]
    M.update(TiltL=f"{t0.left_mean:.2f}", TiltR=f"{t0.right_mean:.2f}", TiltP=f"{t0.p_perm:.2f}")
    bd = R / "stroke_bin_differences.csv"
    if bd.is_file():
        d = pd.read_csv(bd)
        for b, k in [("hinge", "Hinge"), ("slant", "Slant"), ("orientation", "Orient"), ("turning", "Turn"), ("direction", "Dir"), ("endings", "End")]:
            M[f"Sig{k}"] = int((d[d.block == b].q_bh_block < 0.10).sum()); M[f"N{k}"] = int((d.block == b).sum())
        sl = d[d.block == "slant"].reset_index(drop=True)
        M["SlantBackG"] = f"$-${abs(sl.hedges_g[6:9]).min():.2f} to $-${abs(sl.hedges_g[6:9]).max():.2f}"
        M["SlantBackQ"] = p3(sl.q_bh_block[6:9].max()); M["SlantVertG"] = f"{sl.hedges_g[4]:.2f}"; M["SlantVertP"] = p3(sl.p_perm[4])
        M["SlantBackL"] = pc(sl.left_mean[6:9].sum()); M["SlantBackR"] = pc(sl.right_mean[6:9].sum())
        hg = d[d.block == "hinge"]; M["HingeMaxG"] = f"{abs(hg.hedges_g).max():.2f}"
        M["HingeStraightG"] = f"{hg[hg.feature == 'hinge_03_09'].hedges_g.iloc[0]:.2f}"
        tn = d[d.block == "turning"]; M["TurnStraightG"] = f"{tn[tn.feature == 'turn12_00'].hedges_g.iloc[0]:.2f}"
    M.update(InkConG=f"{f(allk, 'ink_contrast', 'hedges_g'):.2f}".replace("-", "$-$"), InkConP=p3(f(allk, "ink_contrast", "p_perm")),
             InkDensP=p3(f(allk, "ink_density", "p_perm")))
    if (R / "s3_sensitivity.csv").is_file():
        s3 = pd.read_csv(R / "s3_sensitivity.csv").set_index("rep")
        for rep, k in key.items():
            M[f"{k}Outl"] = pc(s3.loc[rep, "writer_ba_S3"])
        M["OutlShiftMin"] = sg(min(100 * (s3.writer_ba_S3 - s3.writer_ba))); M["OutlShiftMax"] = sg(max(100 * (s3.writer_ba_S3 - s3.writer_ba)))
    noT = [100 * (s2("S2_no_truncated_images", r) - g(r)) for r in key]
    M["TruncShiftMin"] = sg(min(noT)); M["TruncShiftMax"] = sg(max(noT))
    csm = pd.read_csv(R / "controls_metrics_summary.csv")
    M["PhotoAuc"] = f"{csm[(csm.rep == 'global_intensity') & (csm.clf == 'svm') & (csm.analysis == 'A')].auc_mean.iloc[0]:.2f}"
    for rep, k in [("hc183", "Hc"), ("style6", "Style"), ("stroke", "Stroke"), ("combined", "Comb")]:
        for i in (1, 2, 3):
            M[f"{k}D{'ABC'[i - 1]}"] = pc(g(rep, f"D_S{i}")); M[f"{k}E{'ABC'[i - 1]}"] = pc(g(rep, f"E_S{i}"))
    sp = pd.read_csv(R / "writer_correctness_spearman.csv", index_col=0)
    M["SpStrokeHc"] = f"{sp.loc['stroke', 'hc183']:.2f}"

    au = json.loads((ROOT / "audit/geometric/summary.json").read_text())      # dataset validation (validate_dataset.py)
    M.update(AuditMaxDiff=au["same_prompt_different_writers_max_inliers"],
             AuditDupLo=f"{au['unreleased_P19_vs_R_p6_inliers'][0]:,}", AuditDupHi=f"{au['unreleased_P19_vs_R_p6_inliers'][1]:,}")

    out = ROOT / a.output; out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% generated by scripts/make_prl_macros.py from " + a.run + "; do not edit by hand"]
    for k, v in M.items():
        assert k.isalpha(), k
        lines.append(f"\\newcommand{{\\v{k}}}{{{v}}}")
    out.write_text("\n".join(lines) + "\n")
    pd.Series(M).to_csv(R / "manuscript_macros.csv", header=["value"])
    print(len(M), "macros written to", out)


if __name__ == "__main__":
    main()
