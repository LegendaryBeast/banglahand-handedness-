#!/usr/bin/env python3
"""LaTeX macros for the IJDAR manuscript from the framing / baseline results in results/ijdar_dev_2026-11-12."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "results/ijdar_dev_2026-11-12"
OUT = ROOT / "manuscripts/ijdar/figures/layout_macros.tex"


def main():
    S = pd.read_csv(DEV / "layout_cv_summary.csv").set_index("rep")
    E = pd.read_csv(DEV / "layout_cv_adjusted.csv").set_index(["rep", "covariates"]).adjusted_ba
    P = pd.read_csv(DEV / "layout_permutation.csv").set_index("rep").p
    T = pd.read_csv(DEV / "layout_tests.csv").set_index("measure")
    img = pd.read_csv(DEV / "layout_images.csv")
    B = pd.read_csv(DEV / "baselines_pitfalls_adjusted.csv").set_index("rep")
    pct = lambda v: f"{100 * v:.1f}"
    m = {
        "vFrameA": pct(S.loc["framing", "ba_mean"]), "vFrameP": f"{P['framing']:.3f}",
        "vLayoutBehA": pct(S.loc["behaviour", "ba_mean"]), "vLayoutBehP": f"{P['behaviour']:.3f}",
        "vHcNoLineA": pct(S.loc["hc183_noline", "ba_mean"]),
        "vHcTbA": pct(S.loc["hc183_textblock", "ba_mean"]), "vHcTbP": f"{P['hc183_textblock']:.3f}",
        "vHcTbAdj": pct(E[("hc183_textblock", "ink+photo")]),
        "vHcTbAdjFrame": pct(E[("hc183_textblock", "ink+photo+framing")]),
        "vHcAdjFrame": pct(E[("hc183", "ink+photo+framing")]),
        "vHogRegA": pct(S.loc["hog_registered", "ba_mean"]), "vHogRegP": f"{P['hog_registered']:.3f}",
        "vBorderRight": f"{100 * img[img.label == 0].has_ruled.mean():.0f}",
        "vBorderLeft": f"{100 * img[img.label == 1].has_ruled.mean():.0f}",
        "vBorderInkG": f"{T.loc['line_ink_share', 'hedges_g']:.2f}",
        "vHingeSplit": pct(B.loc["edge_hinge", "para_ba_writer_split"]),
        "vHingeLeaky": pct(B.loc["edge_hinge", "para_ba_leaky"]),
        "vHingeGain": f"{100 * (B.loc['edge_hinge', 'para_ba_leaky'] - B.loc['edge_hinge', 'para_ba_writer_split']):.0f}",
    }
    # MATRA-INK (run_matra_evaluation.py) and blur control (run_blur_control.py)
    M = pd.read_csv(DEV / "matra_metrics_summary.csv").set_index(["rep", "clf", "analysis"]).ba_mean
    MA = pd.read_csv(DEV / "matra_adjusted.csv").set_index("rep").adjusted_mean
    MP = pd.read_csv(DEV / "matra_permutation.csv").set_index(["rep", "adjusted"]).p
    MM = pd.read_csv(DEV / "matra_mirror_test.csv").groupby("rep")[["ba_original", "ba_mirrored", "flip_fraction"]].mean()
    FT = pd.read_csv(DEV / "matra_feature_tests.csv").set_index("feature")
    BT = pd.read_csv(DEV / "blur_tests.csv"); BC = pd.read_csv(DEV / "blur_classification.csv")
    low = FT.loc[[f"cross_{k:02d}" for k in range(7, 13)]]
    bt = lambda fam: BT[(BT.family == fam) & BT.measure.isin([f"cross_{k:02d}" for k in range(9, 13)])]
    braw, badj = bt("headstroke profile (raw)"), bt("headstroke profile | blur + time")
    m.update({
        "vMatraA": pct(M[("matra", "svm", "A")]), "vMatraP": f"{MP[('matra', False)]:.3f}",
        "vMatraC": pct(M[("matra", "svm", "C")]), "vMatraB": pct(M[("matra", "svm", "B")]),
        "vMatraLR": pct(M[("matra", "logistic", "A")]), "vMatraRF": pct(M[("matra", "random_forest", "A")]),
        "vMatraAdj": pct(MA["matra"]), "vMatraAdjP": f"{MP[('matra', True)]:.3f}",
        "vMatraDepA": pct(M[("matra_deposition", "svm", "A")]), "vMatraDepP": f"{MP[('matra_deposition', False)]:.3f}",
        "vMatraDepAdj": pct(MA["matra_deposition"]), "vMatraDepAdjP": f"{MP[('matra_deposition', True)]:.3f}",
        "vMatraDirA": pct(M[("matra_direction", "svm", "A")]), "vMatraDirP": f"{MP[('matra_direction', False)]:.3f}",
        "vMatraGeoA": pct(M[("matra_geometry", "svm", "A")]),
        "vHcMatraA": pct(M[("hc183+matra", "svm", "A")]), "vHcMatraAdj": pct(MA["hc183+matra"]),
        "vStrokeMatraA": pct(M[("stroke+matra", "svm", "A")]),
        "vMatraDepMir": pct(MM.loc["matra_deposition", "ba_mirrored"]),
        "vMatraDepFlip": f"{100 * MM.loc['matra_deposition', 'flip_fraction']:.1f}",
        "vMatraDirMir": pct(MM.loc["matra_direction", "ba_mirrored"]),
        "vMatraAllMir": pct(MM.loc["matra", "ba_mirrored"]),
        "vLowGmin": f"{low.hedges_g.max():.2f}", "vLowGmax": f"{low.hedges_g.min():.2f}", "vLowQ": f"{low.q_bh_block.max():.3f}",
        "vBlurGmin": f"{badj.hedges_g.max():.2f}", "vBlurGmax": f"{badj.hedges_g.min():.2f}",
        "vBlurPmin": f"{badj.p_perm.min():.3f}", "vBlurPmax": f"{badj.p_perm.max():.3f}", "vBlurQ": f"{badj.q_bh_family.max():.3f}",
        "vEdgeWidthP": f"{BT.set_index('measure').loc['edge_width_px', 'p_perm']:.2f}",
        "vPaperSharpP": f"{BT.set_index('measure').loc['paper_sharpness', 'p_perm']:.2f}",
        "vMatraDepAdjAll": pct(BC[(BC.rep == "matra_deposition") & (BC.adjustment == "ink+photo+blur+time")].ba_mean.iloc[0]),
        "vMatraDepAdjAllP": f"{BC[(BC.rep == 'matra_deposition') & (BC.adjustment == 'ink+photo+blur+time (perm)')].p.iloc[0]:.3f}",
    })
    f = DEV / "lineless_embeddings_summary.csv"
    if f.exists():
        L = pd.read_csv(f).set_index("rep")
        for k, nm in (("resnet18", "Res"), ("efficientnet_b0", "Eff"), ("vit_b_16", "Vit")):
            m[f"v{nm}LinelessA"] = pct(L.loc[k + "_lineless", "ba_mean"])
            m[f"v{nm}LinelessAdj"] = pct(L.loc[k + "_lineless", "adjusted_ba"])
            m[f"v{nm}WholeA"] = pct(L.loc[k, "ba_mean"])
    # line-free primary rows (run_linefree_primary.py); the table run's permutation p is used everywhere
    rows = sorted(DEV.glob("linefree_table_rows*.csv"))
    if rows:
        R = pd.concat([pd.read_csv(f) for f in rows]).drop_duplicates("rep", keep="last").set_index("rep")
        x = R.loc["hc183_lf"]
        m.update({"vHcTbA": pct(x.A), "vHcTbP": f"{x.p:.3f}", "vHcTbAdj": pct(x.adjusted),
                  "vHcLfSd": pct(x.A_sd), "vHcLfLo": f"{100 * x.ci_low:.0f}", "vHcLfHi": f"{100 * x.ci_high:.0f}",
                  "vHcLfAuc": f"{x.auc:.2f}", "vHcLfLR": pct(x.LR), "vHcLfRF": pct(x.RF), "vHcLfC": pct(x.C),
                  "vHcLfB": pct(x.B), "vHcLfAll": pct(x.ALL8), "vHcLfTime": pct(x.time_adj), "vHcLfAcqTime": pct(x.acq_time),
                  "vHcLfParaW": pct(x.para_ba_writer), "vHcLfLeaky": pct(x.para_ba_leaky), "vHcLfWid": pct(x.writer_id_top1),
                  "vHcLfCD": f"{100 * x.C_minus_D:+.1f}".replace("-", "$-$"), "vHcLfBE": f"{100 * x.B_minus_E:+.1f}".replace("-", "$-$")})
        nets = [r for r in ("resnet18_lf", "efficientnet_b0_lf", "vit_b_16_lf") if r in R.index]
        m["vNetLfRange"] = f"{100 * R.loc[nets, 'A'].min():.1f}--{100 * R.loc[nets, 'A'].max():.1f}"
        m["vNetLfPmax"] = f"{R.loc[nets, 'p'].max():.3f}"
        m["vNetLfWidRange"] = f"{100 * R.loc[nets, 'writer_id_top1'].min():.1f}--{100 * R.loc[nets, 'writer_id_top1'].max():.1f}"
        m["vNetLfLeakyRange"] = f"{100 * R.loc[nets, 'para_ba_leaky'].min():.1f}--{100 * R.loc[nets, 'para_ba_leaky'].max():.1f}"
        if "stroke_lf" in R.index:
            s = R.loc["stroke_lf"]
            m.update({"vStrokeLfA": pct(s.A), "vStrokeLfP": f"{s.p:.3f}", "vStrokeLfAdj": pct(s.adjusted),
                      "vStrokeLfLo": f"{100 * s.ci_low:.0f}", "vStrokeLfHi": f"{100 * s.ci_high:.0f}"})
    ftp = DEV / "linefree_finetune/finetune_grey_summary.csv"
    if ftp.exists():
        F = pd.read_csv(ftp).set_index("readout")
        m.update({"vFtLfA": pct(F.loc["svm_A", "ba_mean"]), "vFtLfAdj": pct(F.loc["svm_adj_A", "ba_mean"]),
                  "vFtLfEA": pct(F.loc["e2e_A", "ba_mean"]) if "e2e_A" in F.index else "--",
                  "vFtLfLo": f"{100 * F.loc['svm_A', 'ci_low']:.0f}", "vFtLfHi": f"{100 * F.loc['svm_A', 'ci_high']:.0f}",
                  "vFtLfC": pct(F.loc["svm_C", "ba_mean"])})
        fp = DEV / "linefree_finetune/finetune_grey_pitfalls.csv"
        if fp.exists():
            Q = pd.read_csv(fp).iloc[0]
            m.update({"vFtLfWid": pct(Q.writer_id_top1), "vFtLfAcqTime": pct(Q.acq_time), "vFtLfTime": pct(Q.time_adj)})
    lf = DEV / "linefree_secondary"                         # run_linefree_secondary.py, extract_linefree_secondary.py
    if (lf / "secondary_metrics_summary.csv").exists():
        s2 = pd.read_csv(lf / "secondary_metrics_summary.csv")
        g2 = lambda st, r: s2[(s2.study == st) & (s2.rep == r)].ba_mean.iloc[0]
        for b, k in (("hinge", "Hinge"), ("direction", "Dir"), ("orient_slant", "Orient"), ("endings", "End"), ("rates", "Rates"), ("turning", "Turn")):
            m[f"vBlkLf{k}"] = pct(g2("ablation", f"stroke_lf_{b}"))
        m["vLearnHHcLf"] = pct(g2("learning_0.50", "hc183_lf"))
        ev = pd.read_csv(lf / "evidence_curve.csv"); m["vEvHcLfOne"] = pct(ev[(ev.rep == "hc183_lf") & (ev.k == 1)].ba.mean())
        mt = pd.read_csv(lf / "mirror_test_by_split.csv"); ma = pd.read_csv(lf / "mirror_augmented_stroke.csv")
        me = pd.read_csv(lf / "mirror_effect_permutation.csv").iloc[0]
        m.update(vStrokeLfMirOrig=pct(mt.ba_original.mean()), vStrokeLfMir=pct(mt.ba_mirrored.mean()),
                 vStrokeLfMirFlip=f"{100 * mt.flip_fraction.mean():.0f}", vStrokeLfMirEff=f"{100 * me.observed:.1f}",
                 vStrokeLfMirEffP=f"{me.p:.3f}", vStrokeLfAugOrig=pct(ma.ba_original.mean()), vStrokeLfAugMir=pct(ma.ba_mirrored.mean()))
        sp = pd.read_csv(lf / "writer_correctness_spearman.csv", index_col=0); m["vSpStrokeHcLf"] = f"{sp.loc['stroke_lf', 'hc183_lf']:.2f}"
        d = pd.read_csv(lf / "stroke_bin_differences.csv")
        for b, k in (("slant", "Slant"), ("hinge", "Hinge"), ("orientation", "Orient"), ("turning", "Turn"), ("direction", "Dir"), ("endings", "End")):
            sub = d[d.block == b]; m[f"vSigLf{k}"] = str(int((sub.q_bh_block < 0.10).sum())); m[f"vNLf{k}"] = str(len(sub))
        sl = d[d.block == "slant"].reset_index(drop=True)
        m["vSlantBackLfG"] = f"$-${abs(sl.hedges_g[6:9]).min():.2f} to $-${abs(sl.hedges_g[6:9]).max():.2f}"
        m["vSlantBackLfQ"] = f"{sl.q_bh_block[6:9].max():.3f}"
        m["vSlantBackLfL"], m["vSlantBackLfR"] = pct(sl.left_mean[6:9].sum()), pct(sl.right_mean[6:9].sum())
        T = pd.read_csv(lf / "pseudo_pressure_tests.csv").set_index("measure")
        for k, n in (("core_darkness", "Core"), ("darkness_at_ref_width", "Ref"), ("edge_darkness", "Edge"), ("stroke_width", "Width")):
            m[f"vPp{n}G"] = f"{T.loc[k, 'hedges_g']:.2f}".replace("-", "$-$"); m[f"vPp{n}P"] = f"{T.loc[k, 'p_perm']:.3f}"
        m["vPpRefQ"] = f"{T.loc['darkness_at_ref_width', 'q_bh']:.3f}"
        m["vPpTimeMax"] = f"{max(abs(T.loc['core_darkness', 'rho_time']), abs(T.loc['darkness_at_ref_width', 'rho_time'])):.2f}"
    if rows:
        s_ = R.loc["stroke_lf"] if "stroke_lf" in R.index else None
        if s_ is not None:
            m.update(vStrokeLfWid=pct(s_.writer_id_top1), vStrokeLfParaW=pct(s_.para_ba_writer), vStrokeLfLeaky=pct(s_.para_ba_leaky))
    OUT.write_text("% generated by scripts/make_ijdar_layout_macros.py\n" +
                   "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
    print(OUT.read_text())


if __name__ == "__main__":
    main()
