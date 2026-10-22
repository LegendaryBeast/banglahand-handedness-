#!/usr/bin/env python3
"""Build the Kaggle upload: notebooks/*.ipynb (sources) and release/kaggle/ (bangla-handedness-42.zip + notebooks + README).

Zip layout (Kaggle shows it under /kaggle/input/<your-dataset-slug>/):
  bangla-handedness-42/
    dataset/          Left-Handed/, Right-Handed/, metadata/, SHA256SUMS.txt, README.md   (the released data)
    code/             analysis scripts used for the PRL paper
    weights/torch/hub/checkpoints/   ImageNet ResNet-18, EfficientNet-B0, ViT-B/16 (offline use)
    extra/excluded_raw/P3, P19       raw camera files of the two right-handed folders not released (pitfall and sensitivity analyses)
    extra/audit/geometric_pairs.csv.gz   SIFT-RANSAC duplicate-screen scores (Fig. 1f)
    KAGGLE_README.md
"""
import shutil
import zipfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release/kaggle"
NB = ROOT / "notebooks"
TOP = "bangla-handedness-42"
CODE = ["final_study_pipeline.py", "run_crop_evaluation.py", "stroke_features.py", "extract_final_features.py",
        "extract_stroke_features.py", "run_final_experiments.py", "run_pitfall_analyses.py", "run_posthoc_confound_analyses.py",
        "run_secondary_analyses.py", "run_finetuned_stroke_cnn.py", "run_adjusted_permutation.py", "run_framing_check.py",
        "run_tilt_check.py", "run_stroke_differences.py", "run_s3_sensitivity.py", "run_revision_analyses.py", "run_pseudo_pressure.py",
        "make_prl_figures_v2.py", "make_prl_macros.py", "make_prl_overview_figures.py"]
WEIGHTS = ["resnet18-f37072fd.pth", "efficientnet_b0_rwightman-7f5810bc.pth", "vit_b_16-c867db91.pth"]

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = r'''import os, sys, glob, shutil, subprocess, time, json
from pathlib import Path
# --- locate the uploaded Kaggle dataset (any slug) ---
KAGGLE = os.environ.get("KAGGLE_ROOT", "/kaggle")      # override only for local testing
hits = glob.glob(f"{KAGGLE}/input/**/dataset/metadata/images.csv", recursive=True)
assert hits, "Add the uploaded dataset to this notebook (Add data -> Your datasets)."
INPUT = Path(hits[0]).parents[2]
PROJECT = Path(f"{KAGGLE}/working/project")
(PROJECT / "results").mkdir(parents=True, exist_ok=True)
if not (PROJECT / "scripts").exists():
    shutil.copytree(INPUT / "code", PROJECT / "scripts")
if not (PROJECT / "dataset").exists():
    os.symlink(INPUT / "dataset", PROJECT / "dataset")
raw = PROJECT / "camera_originals/Right-Handed"; raw.mkdir(parents=True, exist_ok=True)
for f in ["P3", "P19"]:
    if not (raw / f).exists():
        os.symlink(INPUT / "extra/excluded_raw" / f, raw / f)
geo = PROJECT / "audit/geometric"; geo.mkdir(parents=True, exist_ok=True)
for f in ["geometric_pairs.csv.gz", "summary.json"]:                # dataset validation (Fig. 1f, audit numbers in the text)
    if not (geo / f).exists():
        os.symlink(INPUT / "extra/audit" / f, geo / f)
os.environ["TORCH_HOME"] = str(INPUT / "weights/torch")          # offline ImageNet weights
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
PY = sys.executable
JOBS = os.cpu_count() or 4
print("input:", INPUT, "| project:", PROJECT, "| cpus:", JOBS)
import torch; print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

def run(args, log, done_file=None):
    """Run one pipeline script; skip if its main output already exists (lets you resume after a timeout)."""
    if done_file and Path(done_file).exists():
        print("skip (exists):", done_file); return
    t = time.time(); print("running:", " ".join(map(str, args)), flush=True)
    with open(log, "w") as fh:
        r = subprocess.run([PY, *map(str, args)], cwd=PROJECT, stdout=fh, stderr=subprocess.STDOUT)
    tail = Path(log).read_text().splitlines()[-15:]
    print("\n".join(l for l in tail if "resource_tracker" not in l and "KeyError" not in l))
    print(f"exit {r.returncode} after {time.time() - t:.0f}s"); assert r.returncode == 0, f"failed, see {log}"
'''


def nb_validate():
    cells = [md("""# 1. Dataset check — Bangla handedness paragraphs (42 writers)
Verifies every image against `SHA256SUMS.txt`, summarises the metadata and shows sample paragraphs.
Runs on CPU in about 2 minutes. No GPU needed."""),
             code(SETUP),
             code(r'''import hashlib, pandas as pd
from PIL import Image
D = PROJECT / "dataset"
m = pd.read_csv(D / "metadata/images.csv")
sums = dict(l.split("  ")[::-1] for l in (D / "SHA256SUMS.txt").read_text().strip().splitlines())
bad = [r.relative_path for r in m.itertuples()
       if hashlib.sha256((D / r.relative_path).read_bytes()).hexdigest() != sums[r.relative_path]]
print("images:", len(m), "| writers:", m.writer_id.nunique(), m.groupby("hand").writer_id.nunique().to_dict())
print("checksum mismatches:", len(bad))
print("sizes:", (m.width.astype(str) + "x" + m.height.astype(str)).value_counts().to_dict())
print("paragraphs per writer:", m.groupby("writer_id").size().unique())
print("truncated photos:", int(m.text_truncated_at_edge.notna().sum()))
display(pd.read_csv(D / "metadata/writers.csv").head(10))'''),
             code(r'''import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for ax, (w, k) in zip(axes.ravel(), [("L_p3", 1), ("L_p10", 1), ("R_p7", 1), ("R_p16", 1), ("L_p3", 7), ("L_p10", 7), ("R_p7", 7), ("R_p16", 7)]):
    r = m[(m.writer_id == w) & (m.paragraph_number == k)].iloc[0]
    ax.imshow(Image.open(D / r.relative_path)); ax.set_title(f"{w} ({r.hand}) paragraph {k}"); ax.axis("off")
plt.tight_layout()''')]
    return cells


def nb_experiments():
    cells = [md("""# 2. All experiments of the PRL paper (Kaggle, GPU T4)

**Settings:** Accelerator **GPU T4 x1**, Internet **off** is fine (weights are in the dataset).
The notebook runs the declared protocol of the paper end to end and writes everything to `/kaggle/working/project/results/RUN`.
Each step is skipped if its output already exists, so after a session timeout you can re-run all cells and it continues.
Download `/kaggle/working/results_RUN.zip` at the end.

| step | what | device | time on Kaggle (approx.) |
|---|---|---|---|
| 1 | features: HC-183, frozen ResNet-18/EfficientNet-B0/ViT-B/16, controls, style, STROKE, stroke tiles | GPU | 25-35 min |
| 2 | 20 x 5-fold writer CV, all representations x 3 classifiers x text conditions; controls; ablations; sensitivity; learning curves; 999 whole-writer permutations | CPU | 2.5-4 h (QUICK: 15 min) |
| 3 | pitfalls, adjustment, control conditions, single measures, mirror test | CPU | 30-45 min |
| 4 | fine-tuned stroke ResNet-18 and mirror-trained control (5 x 5 folds each) | GPU | 30-50 min |
| 5 | post hoc: adjusted permutation, framing, tilt, stroke-bin differences, outlier-folder sensitivity, capture time, mirror-augmented STROKE, writer-bootstrap intervals | CPU | 90-120 min |
| 6 | figures, LaTeX table rows and the macro file used by the paper | CPU | 2 min |

Set `QUICK = True` for a fast functional test (fewer permutations; numbers then differ from the paper)."""),
             code(SETUP),
             code(r'''QUICK = False                         # True: 49 permutations instead of 999 (smoke test only)
N_PERM = 49 if QUICK else 999
RUN = PROJECT / "results" / ("final42_kaggle_quick" if QUICK else "final42_kaggle")
RUN.mkdir(parents=True, exist_ok=True)
print(RUN)'''),
             md("## Step 1 — features (GPU)"),
             code(r'''run(["scripts/extract_final_features.py", "--output", RUN.relative_to(PROJECT) / "base"], RUN / "log_base.txt", RUN / "base/features.npz")
run(["scripts/extract_stroke_features.py", "--base", RUN.relative_to(PROJECT) / "base", "--output", RUN.relative_to(PROJECT) / "features", "--excluded", "42"],
    RUN / "log_stroke.txt", RUN / "features/features_excluded.npz")'''),
             md("## Step 2 — declared cross-validation grid and permutation tests (CPU)"),
             code(r'''run(["scripts/run_final_experiments.py", "--run", RUN.relative_to(PROJECT), "--n-perm", N_PERM, "--jobs", JOBS],
    RUN / "log_experiments.txt", RUN / "run_provenance.json")'''),
             md("## Step 3 — secondary analyses (CPU)"),
             code(r'''run(["scripts/run_secondary_analyses.py", "--run", RUN.relative_to(PROJECT), "--jobs", JOBS], RUN / "log_secondary.txt", RUN / "mirror_test.csv")'''),
             md("## Step 4 — fine-tuned stroke network and mirror-trained control (GPU)"),
             code(r'''run(["scripts/run_finetuned_stroke_cnn.py", "--run", RUN.relative_to(PROJECT)], RUN / "log_finetune.txt", RUN / "finetune_metrics_by_split.csv")
run(["scripts/run_finetuned_stroke_cnn.py", "--run", RUN.relative_to(PROJECT), "--mirror-augment"], RUN / "log_finetune_mirror.txt", RUN / "finetune_mirroraug_metrics_by_split.csv")'''),
             md("## Step 5 — post hoc analyses (CPU)"),
             code(r'''run(["scripts/run_adjusted_permutation.py", "--run", RUN.relative_to(PROJECT), "--n-perm", N_PERM, "--jobs", JOBS], RUN / "log_adjperm.txt", RUN / "adjusted_permutation.csv")
run(["scripts/run_framing_check.py", "--run", RUN.relative_to(PROJECT)], RUN / "log_framing.txt", RUN / "framing_check_tests.csv")
run(["scripts/run_tilt_check.py", "--run", RUN.relative_to(PROJECT)], RUN / "log_tilt.txt", RUN / "tilt_tests.csv")
run(["scripts/run_stroke_differences.py", "--run", RUN.relative_to(PROJECT)], RUN / "log_strokediff.txt", RUN / "stroke_bin_differences.csv")
run(["scripts/run_s3_sensitivity.py", "--run", RUN.relative_to(PROJECT), "--jobs", JOBS], RUN / "log_s3.txt", RUN / "s3_sensitivity.csv")
run(["scripts/run_revision_analyses.py", "--run", RUN.relative_to(PROJECT), "--n-perm", N_PERM, "--jobs", JOBS], RUN / "log_revision.txt", RUN / "writer_bootstrap_ci.csv")
run(["scripts/run_pseudo_pressure.py", "--run", RUN.relative_to(PROJECT)], RUN / "log_pressure.txt", RUN / "pseudo_pressure_tests.csv")'''),
             md("## Step 6 — figures, tables and paper macros"),
             code(r'''FIG = RUN / "figures"
run(["scripts/make_prl_overview_figures.py", "--output", FIG.relative_to(PROJECT), "--run", RUN.relative_to(PROJECT)], RUN / "log_overview.txt")
run(["scripts/make_prl_figures_v2.py", "--run", RUN.relative_to(PROJECT), "--output", FIG.relative_to(PROJECT)], RUN / "log_figures.txt")
run(["scripts/make_prl_macros.py", "--run", RUN.relative_to(PROJECT), "--output", (FIG / "results_macros.tex").relative_to(PROJECT)], RUN / "log_macros.txt")'''),
             md("## Main results"),
             code(r'''import pandas as pd
pd.set_option("display.precision", 3)
s = pd.read_csv(RUN / "metrics_summary.csv")
display((100 * s[s.analysis.isin(["A", "C", "B", "ALL8"])].pivot_table(index=["rep", "clf"], columns="analysis", values="ba_mean")[["A", "C", "B", "ALL8"]]).round(1))
display(pd.read_csv(RUN / "permutation_tests.csv")[["rep", "observed_mean_ba", "p_mean", "null_q95"]])
for f in ["finetune_metrics_by_split.csv", "finetune_mirroraug_metrics_by_split.csv"]:
    print(f); display(pd.read_csv(RUN / f).drop(columns="seed").agg(["mean", "std"]))
display(pd.read_csv(RUN / "mirror_test.csv", index_col=0))
display(pd.read_csv(RUN / "adjusted.csv"))
display(pd.read_csv(RUN / "adjusted_permutation.csv"))
print("Sensitivity: re-include the outlier folder (raw RH/P3)"); display(pd.read_csv(RUN / "s3_sensitivity.csv"))'''),
             code(r'''from IPython.display import Image as Img, display as show
for f in ["fig1_dataset.png", "fig2_method.png", "fig2_main_results.png", "fig4_mechanism.png", "fig5_writer_errors.png", "fig3_curves.png", "fig6_stroke_differences.png"]:
    p = RUN / "figures" / f
    if p.exists(): show(Img(filename=str(p)))'''),
             code(r'''shutil.make_archive(f"{KAGGLE}/working/results_{RUN.name}", "zip", RUN)
print("download:", f"{KAGGLE}/working/results_{RUN.name}.zip")''')]
    return cells


def write_nb(cells, path):
    nb = nbf.v4.new_notebook(); nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb["metadata"]["kaggle"] = {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": False}
    nbf.write(nb, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True); NB.mkdir(exist_ok=True)
    for name, cells in [("01_dataset_check.ipynb", nb_validate()), ("02_prl_experiments.ipynb", nb_experiments())]:
        write_nb(cells, NB / name); shutil.copy(NB / name, OUT / name)
    readme = (NB / "KAGGLE_README.md").read_text()
    (OUT / "KAGGLE_README.md").write_text(readme)
    zpath = OUT / f"{TOP}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:   # JPEGs do not compress further
        D = ROOT / "dataset"
        for f in sorted(D.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                z.write(f, f"{TOP}/dataset/{f.relative_to(D)}")
        for c in CODE:
            z.write(ROOT / "scripts" / c, f"{TOP}/code/{c}")
        for w in WEIGHTS:
            z.write(Path.home() / ".cache/torch/hub/checkpoints" / w, f"{TOP}/weights/torch/hub/checkpoints/{w}")
        for fld in ["P3", "P19"]:
            for f in sorted((ROOT / "camera_originals/Right-Handed" / fld).glob("*.jpg")):
                z.write(f, f"{TOP}/extra/excluded_raw/{fld}/{f.name}")
        for f in ["geometric_pairs.csv.gz", "summary.json"]:
            z.write(ROOT / "audit/geometric" / f, f"{TOP}/extra/audit/{f}")
        z.writestr(f"{TOP}/KAGGLE_README.md", readme)
    print("written", zpath, f"{zpath.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
