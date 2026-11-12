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
    OUT.write_text("% generated by scripts/make_ijdar_layout_macros.py\n" +
                   "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
    print(OUT.read_text())


if __name__ == "__main__":
    main()
