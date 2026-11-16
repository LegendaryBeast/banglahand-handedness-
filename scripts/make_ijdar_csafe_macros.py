#!/usr/bin/env python3
"""IJDAR: macros and the table body for the CSAFE external application (run_csafe_evaluation.py outputs).

Reads results/csafe_ijdar_2026-11-12/csafe_table.csv (one row per representation, written as each one finishes) and
csafe_layout_tests.csv (when present). Writes manuscripts/ijdar/figures/csafe_macros.tex and table_csafe_ijdar.tex,
and pastes the table body into the manuscript between "% BEGIN table_csafe_ijdar" / "% END table_csafe_ijdar".
Representations that have not finished yet are left out of the table and their macros are set to "--".
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CS = ROOT / "results/csafe_ijdar_2026-11-12"
FIG = ROOT / "manuscripts/ijdar/figures"
TEX = ROOT / "manuscripts/ijdar/ijdar_manuscript.tex"
GROUPS = [("Ours", [("hc183_tb", "\\textbf{HC-183}, text block"), ("hc183_page", "HC-183, whole page"), ("stroke", "STROKE")]),
          ("Frozen ImageNet embeddings", [("resnet18", "ResNet-18"), ("efficientnet_b0", "EffNet-B0"), ("vit_b_16", "ViT-B/16")]),
          ("Earlier methods", [("edge_direction", "Edge-direction $f_1$"), ("edge_hinge", "Edge-hinge $f_2$"),
                               ("chain_code", "Chain code + curvature"), ("hog_glbp", "HOG + GLBP"), ("lbp", "Uniform LBP")]),
          ("Controls (no handwriting shape)", [("global_intensity", "Scan intensity"), ("background", "Paper only"),
                                                ("layout", "Layout (lines, block position)")])]
pct = lambda v: "--" if pd.isna(v) else f"{100 * v:.1f}"
ci = lambda v: "--" if pd.isna(v) else f"{100 * v:.0f}"
pv = lambda v: "--" if pd.isna(v) else f"{v:.3f}"

def minus(v):
    """Typeset a leading minus sign as a math minus, not a hyphen."""
    v = str(v)
    return "$-$" + v[1:] if re.match(r"^-\d", v) else v


def main():
    T = pd.read_csv(CS / "csafe_table.csv").set_index("rep")
    rows = []
    for title, reps in GROUPS:
        have = [(r, n) for r, n in reps if r in T.index]
        if not have:
            continue
        rows += ([] if not rows else ["\\addlinespace[2pt]"]) + [f"\\multicolumn{{12}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for r, n in have:
            x = T.loc[r]
            rows.append(f"{n} & {int(x.dims):,} & {pct(x.A)}$\\pm${pct(x.A_sd)} & {100 * x.A_lo:.0f}--{100 * x.A_hi:.0f} & {pv(x.A_p)} & "
                        f"{pct(x.A_adjusted)} & {pct(x.X)} & {pv(x.X_p)} & {pct(x.C2)} & {pct(x.C3)} & {pct(x.para_ba_leaky)} & "
                        f"{pct(x.writer_id_top1)} \\\\")
    (FIG / "table_csafe_ijdar.tex").write_text("\n".join(rows) + "\n")

    g = lambda r, c: T.loc[r, c] if r in T.index else np.nan
    m = {"vCsN": "96", "vCsLeft": "48", "vCsRight": "48", "vCsPages": "864",
         "vCsHcA": pct(g("hc183_tb", "A")), "vCsHcSd": pct(g("hc183_tb", "A_sd")),
         "vCsHcLo": ci(g("hc183_tb", "A_lo")), "vCsHcHi": ci(g("hc183_tb", "A_hi")),
         "vCsHcAuc": "--" if pd.isna(g("hc183_tb", "A_auc")) else f"{g('hc183_tb', 'A_auc'):.2f}",
         "vCsHcP": pv(g("hc183_tb", "A_p")), "vCsHcHolm": pv(g("hc183_tb", "H1_holm_p")),
         "vCsHcAdj": pct(g("hc183_tb", "A_adjusted")), "vCsHcLR": pct(g("hc183_tb", "A_LR")), "vCsHcRF": pct(g("hc183_tb", "A_RF")),
         "vCsHcX": pct(g("hc183_tb", "X")), "vCsHcXLo": ci(g("hc183_tb", "X_lo")), "vCsHcXHi": ci(g("hc183_tb", "X_hi")),
         "vCsHcXP": pv(g("hc183_tb", "X_p")), "vCsHcXHolm": pv(g("hc183_tb", "H2_holm_p")), "vCsHcXAdj": pct(g("hc183_tb", "X_adjusted")),
         "vCsHcCTwo": pct(g("hc183_tb", "C2")), "vCsHcCThree": pct(g("hc183_tb", "C3")),
         "vCsHcLeaky": pct(g("hc183_tb", "para_ba_leaky")), "vCsHcParaW": pct(g("hc183_tb", "para_ba_writer")),
         "vCsHcWid": pct(g("hc183_tb", "writer_id_top1")),
         "vCsHcPageA": pct(g("hc183_page", "A")), "vCsHcPageX": pct(g("hc183_page", "X")),
         "vCsStrokeA": pct(g("stroke", "A")), "vCsStrokeP": pv(g("stroke", "A_p")), "vCsStrokeX": pct(g("stroke", "X")),
         "vCsStrokeAdj": pct(g("stroke", "A_adjusted")),
         "vCsPhotoA": pct(g("global_intensity", "A")), "vCsPhotoP": pv(g("global_intensity", "A_p")),
         "vCsPaperA": pct(g("background", "A")), "vCsPaperP": pv(g("background", "A_p")),
         "vCsLayoutA": pct(g("layout", "A")), "vCsLayoutP": pv(g("layout", "A_p"))}
    nets = [r for r in ("resnet18", "efficientnet_b0", "vit_b_16") if r in T.index]
    m["vCsNetRange"] = f"{100 * T.loc[nets, 'A'].min():.1f}--{100 * T.loc[nets, 'A'].max():.1f}" if nets else "--"
    m["vCsNetXRange"] = f"{100 * T.loc[nets, 'X'].min():.1f}--{100 * T.loc[nets, 'X'].max():.1f}" if nets else "--"
    m["vCsNetPmax"] = pv(T.loc[nets, "A_p"].max()) if nets else "--"
    m["vCsNetAdjRange"] = f"{100 * T.loc[nets, 'A_adjusted'].min():.1f}--{100 * T.loc[nets, 'A_adjusted'].max():.1f}" if nets else "--"
    old = [r for r in ("edge_direction", "edge_hinge", "chain_code", "hog_glbp", "lbp") if r in T.index]
    m["vCsOldRange"] = f"{100 * T.loc[old, 'A'].min():.1f}--{100 * T.loc[old, 'A'].max():.1f}" if old else "--"
    m["vCsHingeLeaky"], m["vCsHingeWid"] = pct(g("edge_hinge", "para_ba_leaky")), pct(g("edge_hinge", "writer_id_top1"))
    m["vCsHingeParaW"] = pct(g("edge_hinge", "para_ba_writer"))
    m["vCsWidMax"] = pct(T.writer_id_top1.max())
    m["vCsLeakyGainMax"] = f"{100 * (T.para_ba_leaky - T.para_ba_writer).max():.0f}"
    for r, k in (("hog_glbp", "Hog"), ("edge_hinge", "Hinge"), ("chain_code", "Chain"), ("edge_direction", "EdgeDir"), ("lbp", "Lbp")):
        m[f"vCs{k}A"], m[f"vCs{k}P"] = pct(g(r, "A")), pv(g(r, "A_p"))
        m[f"vCs{k}Adj"], m[f"vCs{k}X"] = pct(g(r, "A_adjusted")), pct(g(r, "X"))
    m["vCsLayoutX"], m["vCsLayoutXP"] = pct(g("layout", "X")), pv(g("layout", "X_p"))
    m["vCsStrokeLeaky"], m["vCsStrokeParaW"], m["vCsStrokeWid"] = pct(g("stroke", "para_ba_leaky")), pct(g("stroke", "para_ba_writer")), pct(g("stroke", "writer_id_top1"))
    m["vCsNetWidRange"] = f"{100 * T.loc[nets, 'writer_id_top1'].min():.1f}--{100 * T.loc[nets, 'writer_id_top1'].max():.1f}" if nets else "--"
    f = CS / "csafe_layout_tests.csv"
    if f.exists():
        L = pd.read_csv(f)
        m["vCsLayoutNsig"], m["vCsLayoutNtot"] = str(int((L.q_bh < 0.05).sum())), str(len(L))
        m["vCsLayoutQmin"] = f"{L.q_bh.min():.3f}"
        m["vCsInkG"] = f"{L.set_index('measure').loc['ink_textblock', 'hedges_g']:.2f}"
        m["vCsInkP"] = f"{L.set_index('measure').loc['ink_textblock', 'p_perm']:.3f}"
        Li = L.set_index("measure")
        for k, nm in (("skew_deg", "Skew"), ("line_slope_mean", "Slope"), ("spacing_rel", "Spacing"), ("ruled_lines", "Ruled"),
                      ("block_cx", "BlockX"), ("block_cy", "BlockY")):
            m[f"vCs{nm}G"], m[f"vCs{nm}Q"] = f"{Li.loc[k, 'hedges_g']:.2f}".replace("-", "$-$"), f"{Li.loc[k, 'q_bh']:.3f}"
        m["vCsSkewL"], m["vCsSkewR"] = f"{Li.loc['skew_deg', 'left_mean']:+.2f}".replace("-", "$-$"), f"{Li.loc['skew_deg', 'right_mean']:+.2f}".replace("-", "$-$")
        pos = [x for x in ("block_cx", "block_cy", "block_w", "block_h", "ruled_lines") if x in Li.index]
        m["vCsPlaceQmin"] = f"{Li.loc[pos, 'q_bh'].min():.2f}"
    else:
        m.update(vCsLayoutNsig="--", vCsLayoutNtot="--", vCsLayoutQmin="--", vCsInkG="--", vCsInkP="--")
    f = CS / "csafe_orientation_check.csv"                  # post hoc line-orientation check (run_csafe_orientation_check.py)
    if f.exists():
        O = pd.read_csv(f); o = lambda r, c, col: O[(O.rep == r) & (O.covariates == c)][col].iloc[0]
        nm = {"hc183_tb": "HC-183, text block", "stroke": "STROKE", "hog_glbp": "HOG + GLBP"}
        cv = {"none": "none", "orientation": "orientation", "orientation+ink+scan": "orientation + ink + scan",
              "all layout": "all layout", "all layout+ink+scan": "all layout + ink + scan"}
        rows_o = []
        for r in ("hc183_tb", "stroke", "hog_glbp"):
            if rows_o:
                rows_o.append("\\addlinespace[2pt]")
            for i, c in enumerate(cv):
                rows_o.append(f"{nm[r] if i == 0 else ''} & {cv[c]} & {pct(o(r, c, 'A'))} & {pct(o(r, c, 'X'))} \\\\")
        (FIG / "table_orient_csafe.tex").write_text("\n".join(rows_o) + "\n")
        for r, k in (("hc183_tb", "Hc"), ("stroke", "Stroke"), ("hog_glbp", "Hog")):
            m[f"vCs{k}OriA"], m[f"vCs{k}OriX"] = pct(o(r, "orientation", "A")), pct(o(r, "orientation", "X"))
            m[f"vCs{k}OriAllA"], m[f"vCs{k}OriAllX"] = pct(o(r, "all layout+ink+scan", "A")), pct(o(r, "all layout+ink+scan", "X"))
        hp = O[(O.rep == "hc183_tb") & (O.covariates == "orientation")].iloc[0]
        m["vCsHcOriP"] = pv(hp["A_p"]) if "A_p" in O.columns else "--"
        m["vCsHcOriXP"] = pv(hp["X_p"]) if "X_p" in O.columns else "--"
        sup = TEX.with_name("supplementary.tex")
        if sup.exists():
            t = sup.read_text(); b_, e_ = "% BEGIN table_orient_csafe\n", "% END table_orient_csafe\n"
            i, j = t.index(b_) + len(b_), t.index(e_)
            sup.write_text(t[:i] + (FIG / "table_orient_csafe.tex").read_text() + t[j:])
    (FIG / "csafe_macros.tex").write_text("% generated by scripts/make_ijdar_csafe_macros.py\n" +
                                          "".join(f"\\newcommand{{\\{k}}}{{{minus(v)}}}\n" for k, v in m.items()))
    t = TEX.read_text(); b, e = "% BEGIN table_csafe_ijdar\n", "% END table_csafe_ijdar\n"
    if b in t:
        i, j = t.index(b) + len(b), t.index(e)
        TEX.write_text(t[:i] + (FIG / "table_csafe_ijdar.tex").read_text() + t[j:])
    print(f"{len(T)} representations;", {k: v for k, v in m.items() if k.startswith("vCsHc")})


if __name__ == "__main__":
    main()
