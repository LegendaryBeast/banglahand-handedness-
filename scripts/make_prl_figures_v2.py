#!/usr/bin/env python3
"""PRL figures and table rows for a finished run (default results/final42_2026-10-19). No refitting."""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

REPS = ["hc183", "style6", "stroke", "combined", "resnet18", "efficientnet_b0", "vit_b_16", "stroke_cnn"]
CTRL = ["global_intensity", "background"]
NAME = {"hc183": "HC-183", "style6": "Style-6", "stroke": "STROKE", "combined": "Combined", "resnet18": "ResNet-18",
        "efficientnet_b0": "EffNet-B0", "vit_b_16": "ViT-B/16", "stroke_cnn": "Stroke-CNN\n(frozen)",
        "finetuned": "Stroke-CNN\n(fine-tuned)", "global_intensity": "Photo\nint.", "background": "Paper\nonly"}
CLF = {"svm": "Linear SVM", "logistic": "Logistic reg.", "random_forest": "Random forest"}
ACOL = {"A": "#1d4ed8", "C": "#059669", "B": "#d97706", "ALL8": "#6b7280"}
ALAB = {"A": "A: shared 6", "C": "C: shared-model transfer", "B": "B: writer-specific 2", "ALL8": "All 8"}
LEFT_C, RIGHT_C = "#c2410c", "#1d4ed8"
RCOL = {"hc183": "#111827", "style6": "#059669", "stroke": "#c2410c", "combined": "#7c3aed", "resnet18": "#9ca3af",
        "efficientnet_b0": "#60a5fa", "vit_b_16": "#a78bfa", "stroke_cnn": "#f59e0b"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=250); plt.close(fig)


def fmt(v, sd=None):
    if v is None or pd.isna(v):
        return "--"
    return f"{100 * v:.1f}" + (f"$\\pm${100 * sd:.1f}" if sd is not None and not pd.isna(sd) else "")


def fig_main(run, out):
    m = pd.read_csv(run / "metrics_by_split.csv")
    s = pd.read_csv(run / "secondary_metrics_by_split.csv")
    ctrl = s[(s.study == "control") & (s.clf == "svm")]
    fig, ax = plt.subplots(figsize=(7.16, 2.2), constrained_layout=True)
    order = ["A", "C", "B", "ALL8"]
    x, ticks = 0, []
    rng = np.random.default_rng(3)
    groups = REPS + CTRL
    for rep in groups:
        if rep in REPS:
            d = m[(m.rep == rep) & (m.clf == "svm")]; series = {a: d[d.analysis == a].ba.to_numpy() for a in order}
        else:
            series = {"A": ctrl[ctrl.rep == rep].ba.to_numpy()}
        start = x
        for a, v in series.items():
            ax.scatter(x + rng.uniform(-0.17, 0.17, len(v)), v, s=4, color=ACOL[a], alpha=0.45, lw=0)
            ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt="o", ms=3.4, color=ACOL[a], mec="white", mew=0.5, capsize=1.5, lw=0.9)
            x += 1
        ticks.append(((start + x - 1) / 2, NAME[rep])); x += 0.7 if rep not in ("stroke_cnn", "global_intensity") else 1.3
        if rep in ("combined", "stroke_cnn"):
            ax.axvline(x - 0.85, color="#d1d5db", lw=0.6)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks], fontsize=5.8)
    ax.set_ylabel("writer balanced accuracy"); ax.set_ylim(0.25, 1.0)
    ax.text(0, 0.95, "handwriting features", fontsize=6, color="#6b7280")
    ax.text(ticks[4][0] - 1.5, 0.95, "deep features", fontsize=6, color="#6b7280")
    ax.text(ticks[-2][0] - 0.9, 0.95, "controls", fontsize=6, color="#6b7280")
    for a in order:
        ax.scatter([], [], color=ACOL[a], s=12, label=ALAB[a])
    ax.legend(ncol=4, loc="lower left", frameon=False, bbox_to_anchor=(0.0, -0.02))
    save(fig, out, "fig2_main_results")


def fig_mechanism(run, out):
    """(a) permutation nulls, (b) STROKE block ablation, (c) mirror test."""
    perm = pd.read_csv(run / "permutation_tests.csv").set_index("rep")
    s = pd.read_csv(run / "secondary_metrics_summary.csv")
    ma = pd.read_csv(run / "mirror_augmented_stroke.csv")              # STROKE SVM trained on original + mirrored ink
    mt = pd.read_csv(run / "mirror_test_by_split.csv") if (run / "mirror_test_by_split.csv").is_file() else None
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.95), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.5, 1.0, 1.1]))
    ax = axes[0]
    reps = [r for r in REPS + CTRL if r in perm.index]
    for i, rep in enumerate(reps):
        nm = pd.read_csv(run / f"permutation_null_{rep}.csv").null_mean_ba
        vp = ax.violinplot(nm, positions=[i], widths=0.8, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor("#d1d5db"); b.set_alpha(0.9); b.set_edgecolor("none")
        ax.scatter(i, perm.loc[rep, "observed_mean_ba"], color="#dc2626", s=12, zorder=3)
        ax.text(i, 0.95, f"{perm.loc[rep, 'p_mean']:.3f}", ha="center", va="bottom", fontsize=4.8, rotation=90)
    ax.set_xticks(range(len(reps))); ax.set_xticklabels([NAME[r].replace("\n", " ") for r in reps], rotation=50, ha="right", fontsize=5.3)
    ax.set_ylim(0.25, 1.1); ax.set_ylabel("repeated-CV mean BA"); ax.set_title("(a) permutation null vs observed", fontsize=6.6)
    ax = axes[1]
    blocks = [("stroke_hinge", "hinge"), ("stroke_direction", "direction 360°"), ("stroke_orient_slant", "orient./slant"),
              ("stroke_endings", "stroke ends"), ("stroke_rates", "end/junction rates"), ("stroke_turning", "turning (mirror-inv.)")]
    vals = [s[(s.study == "ablation") & (s.rep == b) & (s.analysis == "A")].iloc[0] for b, _ in blocks]
    y = np.arange(len(blocks))
    ax.barh(y, [v.ba_mean for v in vals], xerr=[v.ba_sd for v in vals], color="#c2410c", height=0.6, error_kw=dict(lw=0.6, capsize=1.2))
    ax.axvline(0.5, color="k", lw=0.6, ls=":"); ax.set_yticks(y); ax.set_yticklabels([l for _, l in blocks], fontsize=5.6)
    ax.invert_yaxis(); ax.set_xlim(0.3, 0.85); ax.set_xlabel("writer BA (SVM, A)"); ax.set_title("(b) STROKE blocks alone", fontsize=6.6)
    ax = axes[2]
    bars = [("STROKE mirror-\ntrained, original", ma.ba_original), ("STROKE mirror-\ntrained, mirrored", ma.ba_mirrored)]
    if mt is not None:
        bars = [("STROKE SVM\noriginal", mt.ba_original), ("STROKE SVM\nmirrored", mt.ba_mirrored)] + bars
    cols = ["#c2410c", "#fdba74"] * 4
    for i, (lab, v) in enumerate(bars):
        ax.bar(i, v.mean(), yerr=v.std(ddof=1), color=cols[i], width=0.7, error_kw=dict(lw=0.6, capsize=1.2))
    ax.axhline(0.5, color="k", lw=0.6, ls=":"); ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=4.6, rotation=90)
    ax.set_ylim(0.3, 0.85); ax.set_ylabel("writer BA (A)"); ax.set_title("(c) mirror test", fontsize=6.6)
    save(fig, out, "fig4_mechanism")


def fig_curves(run, out):
    ev = pd.read_csv(run / "evidence_curve.csv"); s = pd.read_csv(run / "secondary_metrics_summary.csv")
    m = pd.read_csv(run / "metrics_by_split.csv")
    show = ["hc183", "style6", "stroke", "combined", "efficientnet_b0"]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95), layout="constrained")
    for rep in show:
        g = ev[(ev.rep == rep) & (ev.clf == "svm")].groupby("k").ba
        axes[0].errorbar(g.mean().index, g.mean(), yerr=g.std(), color=RCOL[rep], lw=0.9, ms=2.5, marker="o", capsize=1.2, label=NAME[rep])
        pts = [(fr, s[(s.study == f"learning_{fr:.2f}") & (s.rep == rep)].ba_mean.iloc[0]) for fr in (0.25, 0.5, 0.75)]
        pts.append((1.0, m[(m.rep == rep) & (m.clf == "svm") & (m.analysis == "A")].ba.mean()))
        p = np.array(pts); axes[1].plot(p[:, 0] * 29.6, p[:, 1], color=RCOL[rep], lw=0.9, marker="o", ms=2.5)
    for ax, t, xl in [(axes[0], "(a) paragraphs per test writer", "k (shared paragraphs)"), (axes[1], "(b) training writers per fold", "training writers (mean)")]:
        ax.axhline(0.5, color="k", lw=0.5, ls=":"); ax.set_title(t, fontsize=6.8); ax.set_xlabel(xl); ax.set_ylim(0.45, 0.85)
    axes[0].set_xticks(range(1, 7)); axes[0].set_ylabel("writer balanced accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=5, frameon=False, fontsize=5.4, handlelength=1.0, columnspacing=0.8)
    save(fig, out, "fig3_curves")


def fig_writers(run, out):
    wr = pd.read_csv(run / "writer_predictions.csv.gz")
    d = wr[(wr.clf == "svm") & (wr.analysis == "A")]
    rows = ["hc183", "style6", "stroke", "combined", "efficientnet_b0"]
    acc = d.assign(c=(d.pred == d.label)).groupby(["writer_id", "rep"]).c.mean().unstack()[rows]
    acc["lab"] = d.groupby("writer_id").label.first(); acc["mean"] = acc[rows].mean(axis=1)
    acc = acc.sort_values(["lab", "mean"], ascending=[False, True])
    cmap = LinearSegmentedColormap.from_list("wr", ["#9f2a2a", "#e8908a", "#f0efec", "#86b6ef", "#184f95"])
    fig, ax = plt.subplots(figsize=(3.5, 2.05), constrained_layout=True)
    z = acc[rows].T.to_numpy()
    im = ax.pcolormesh(np.arange(z.shape[1] + 1) - 0.5, np.arange(z.shape[0] + 1) - 0.5, z, cmap=cmap, vmin=0, vmax=1, edgecolors="#fcfcfb", linewidth=0.35)
    ax.invert_yaxis(); ax.set_yticks(range(len(rows))); ax.set_yticklabels(["Fine-tuned CNN" if r == "finetuned" else NAME[r].replace("\n", " ") for r in rows], fontsize=5.8)
    nl = int((acc.lab == 1).sum()); ax.axvline(nl - 0.5, color="#0b0b0b", lw=0.9)
    ax.set_xticks(range(len(acc))); ax.set_xticklabels([w.replace("_p", "") for w in acc.index], rotation=90, fontsize=4.4)
    ax.tick_params(axis="x", length=1.2, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(nl / 2 - 0.5, -0.75, f"left-handed (n={nl})", ha="center", va="bottom", fontsize=6.2)
    ax.text(nl + (len(acc) - nl) / 2 - 0.5, -0.75, f"right-handed (n={len(acc) - nl})", ha="center", va="bottom", fontsize=6.2)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01, ticks=[0, 0.5, 1]); cb.outline.set_visible(False)
    cb.ax.set_yticklabels(["0 (always\nwrong)", "0.5", "1 (always\nright)"], fontsize=5.0)
    save(fig, out, "fig5_writer_errors")
    return acc[rows].corr(method="spearman")


def fig_differences(run, out):
    """Writer-level left-right differences of STROKE bins (amendment 4, descriptive)."""
    d = pd.read_csv(run / "stroke_bin_differences.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.15, 1.0, 1.15]))
    # (a) slant of near-vertical strokes
    ax = axes[0]; sl = d[d.block == "slant"].reset_index(drop=True)
    cen = np.arange(-40, 41, 10); w = 4
    ax.bar(cen - w / 2, sl.left_mean, width=w, color=LEFT_C, label="left-handed")
    ax.bar(cen + w / 2, sl.right_mean, width=w, color=RIGHT_C, label="right-handed")
    for c, q, top in zip(cen, sl.q_bh_block, np.maximum(sl.left_mean, sl.right_mean)):
        if q < 0.10: ax.text(c, top + 0.004, "*", ha="center", fontsize=8)
    ax.set_xticks(cen); ax.set_xticklabels([f"{c}" for c in cen], fontsize=5.6)
    ax.set_xlabel("tilt from vertical (deg): $-$ forward '/', + backward '\\'"); ax.set_ylabel("share of near-vertical stroke pixels")
    ax.set_title("(a) slant distribution", fontsize=6.8); ax.legend(frameon=False, fontsize=5.6)
    # (b) hinge map of Hedges g
    ax = axes[1]; hg = d[d.block == "hinge"]
    G = hg.hedges_g.to_numpy().reshape(12, 12); Q = hg.q_bh_block.to_numpy().reshape(12, 12)
    cmap = LinearSegmentedColormap.from_list("rb", [RIGHT_C, "#f0efec", LEFT_C])
    im = ax.imshow(G, cmap=cmap, vmin=-1.2, vmax=1.2, origin="lower")
    for i, j in zip(*np.where(Q < 0.10)):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=0.8))
    ticks = [0, 3, 6, 9]; ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([f"{30 * t}°" for t in ticks], fontsize=5.6); ax.set_yticklabels([f"{30 * t}°" for t in ticks], fontsize=5.6)
    ax.set_xlabel("incoming leg direction $\\phi_2$"); ax.set_ylabel("outgoing leg direction $\\phi_1$")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label("Hedges g (left $-$ right)", fontsize=5.4); cb.ax.tick_params(labelsize=5)
    ax.set_title("(b) contour hinge (boxes: q < 0.10)" if (Q < 0.10).any() else "(b) contour hinge (no cell q < 0.10)", fontsize=6.8)
    # (c) effect sizes by block
    ax = axes[2]
    order = [("direction", "direction 360°"), ("hinge", "hinge"), ("endings", "stroke ends"), ("orientation", "orientation"),
             ("slant", "slant"), ("turning", "turning"), ("rates", "rates")]
    rng = np.random.default_rng(4)
    for i, (b, lab) in enumerate(order):
        sub = d[d.block == b]
        ax.scatter(sub.hedges_g, i + rng.uniform(-0.18, 0.18, len(sub)), s=5,
                   c=np.where(sub.q_bh_block < 0.10, "#dc2626", "#9ca3af"), lw=0)
    ax.axvline(0, color="k", lw=0.5); ax.set_yticks(range(len(order))); ax.set_yticklabels([l for _, l in order], fontsize=5.8)
    ax.invert_yaxis(); ax.set_xlabel("Hedges g per bin (left $-$ right)"); ax.set_title("(c) all STROKE bins (red: q < 0.10)", fontsize=6.8)
    save(fig, out, "fig6_stroke_differences")


def tables(run, out):
    sm = pd.read_csv(run / "metrics_summary.csv"); cs = pd.read_csv(run / "controls_metrics_summary.csv")
    ci = pd.read_csv(run / "writer_bootstrap_ci.csv").set_index("rep")         # amendment 3
    cis = lambda r: f"{100 * ci.loc[r, 'ci_low']:.0f}--{100 * ci.loc[r, 'ci_high']:.0f}"
    gs = pd.read_csv(run / "finetune_grey_summary.csv").set_index("readout") if (run / "finetune_grey_summary.csv").is_file() else None
    gname = "ResNet-18 (fine-tuned)"                                           # amendment 5, post hoc baseline
    rows = []
    for rep in REPS + CTRL:
        src = sm if rep in REPS else cs
        g = lambda an, clf="svm", col="ba_mean": src[(src.rep == rep) & (src.clf == clf) & (src.analysis == an)][col].iloc[0]
        rows.append(f"{NAME[rep].replace(chr(10), ' ')} & {fmt(g('A'), g('A', col='ba_sd'))} & {cis(rep)} & {fmt(g('A', 'logistic'))} & {fmt(g('A', 'random_forest'))} & "
                    f"{fmt(g('C'), g('C', col='ba_sd'))} & {fmt(g('B'), g('B', col='ba_sd'))} & {fmt(g('ALL8'), g('ALL8', col='ba_sd'))} & {g('A', col='auc_mean'):.2f} \\\\")
        if rep == "stroke_cnn" and gs is not None:
            s = gs.loc["svm_A"]
            gv = lambda r, sd=True: fmt(gs.loc[r, "ba_mean"], gs.loc[r, "ba_sd"] if sd else None) if r in gs.index else "--"
            rows.append(f"{gname} & {gv('svm_A')} & {100 * s.ci_low:.0f}--{100 * s.ci_high:.0f} & {gv('logistic_A', False)} & "
                        f"{gv('random_forest_A', False)} & {gv('svm_C')} & {gv('svm_B')} & {gv('svm_ALL8')} & {s.auc_mean:.2f} \\\\")
        if rep in ("combined", "stroke_cnn"):
            rows.append("\\addlinespace[2pt]")
    (out / "table_main.tex").write_text("\n".join(rows) + "\n")
    ss = pd.read_csv(run / "secondary_metrics_summary.csv"); perm = pd.read_csv(run / "permutation_tests.csv").set_index("rep")
    pf = pd.read_csv(run / "pitfalls.csv").set_index("rep"); ad = pd.read_csv(run / "adjusted.csv")
    csn = pd.read_csv(run / "controls_sensitivity.csv").set_index("rep")
    ta = pd.read_csv(run / "capture_time_adjusted.csv")                       # amendment 3
    tadj = lambda r, c: fmt(ta[(ta.rep == r) & (ta.covariates == c)].ba_mean.iloc[0]) if ((ta.rep == r) & (ta.covariates == c)).any() else "n/a"
    con = pd.concat([pd.read_csv(run / "contrasts_by_split.csv"), pd.read_csv(run / "controls_contrasts_by_split.csv")])
    sgn = lambda v: ("0.0" if abs(v) < 0.05 else f"+{v:.1f}" if v > 0 else f"$-${abs(v):.1f}")
    rows = []
    for rep in REPS + CTRL:
        p = perm.loc[rep]
        if rep in REPS:
            s2 = ss[(ss.study == "S2_no_truncated_images") & (ss.rep == rep) & (ss.analysis == "A")].ba_mean.iloc[0]
            adj = fmt(ad[(ad.rep == rep) & (ad.subset == "all")].adjusted_mean.iloc[0])
        else:
            s2, adj = csn.loc[rep, "S2"], "n/a"
        r = pf.loc[rep]; c = con[(con.rep == rep) & (con.clf == "svm")]
        rows.append(f"{NAME[rep].replace(chr(10), ' ')} & {fmt(p.observed_mean_ba)} & {p.p_mean:.3f} & {adj} & {tadj(rep, 'time')} & {tadj(rep, 'acquisition_and_time')} & "
                    f"{fmt(r.para_ba_writer_split)} & {fmt(r.para_ba_leaky)} & {fmt(r.writer_id_top1)} & "
                    f"{sgn(100 * c.transfer_C_minus_D.mean())} / {sgn(100 * c.varied_B_minus_E.mean())} \\\\")
        if rep == "stroke_cnn" and gs is not None:
            rows.append(f"{gname} & {fmt(gs.loc['svm_A', 'ba_mean'])} & -- & {fmt(gs.loc['svm_adj_A', 'ba_mean'])} & -- & -- & -- & -- & -- & -- \\\\")
    (out / "table_pitfalls.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for rep in ["hc183", "style6", "stroke", "combined"]:
        g = lambda an: sm[(sm.rep == rep) & (sm.clf == "svm") & (sm.analysis == an)].ba_mean.iloc[0]
        c = con[(con.rep == rep) & (con.clf == "svm")]
        rows.append(f"{NAME[rep]} & {fmt(g('A'))} & " + " & ".join(fmt(g(a)) for a in ["D_S1", "D_S2", "D_S3", "C"]) + f" & {sgn(100 * c.transfer_C_minus_D.mean())} & "
                    + " & ".join(fmt(g(a)) for a in ["E_S1", "E_S2", "E_S3", "B"]) + f" & {sgn(100 * c.varied_B_minus_E.mean())} \\\\")
    (out / "table_text.tex").write_text("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--output", required=True)
    a = ap.parse_args(); run, out = ROOT / a.run, ROOT / a.output; out.mkdir(parents=True, exist_ok=True)
    fig_main(run, out); fig_curves(run, out); fig_mechanism(run, out)
    fig_writers(run, out).to_csv(run / "writer_correctness_spearman.csv"); tables(run, out)
    if (run / "stroke_bin_differences.csv").is_file():
        fig_differences(run, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
