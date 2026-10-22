#!/usr/bin/env python3
"""Build the Kaggle package for the grey-image fine-tuned CNN baseline (run_finetuned_grey_cnn.py).

Writes notebooks/03_grey_cnn_baseline.ipynb and release/kaggle_grey_cnn/ with
  banglahand-grey-cnn.zip   dataset/ (images + metadata), code/ (5 scripts), weights/ (ResNet-18 only),
                            extra/run_features/ (saved features.npz and descriptors.csv of the reported run, used for the
                            photometric covariates and the writer order)
  03_grey_cnn_baseline.ipynb
Runtime on a Kaggle T4: about 1 h for 20 repetitions (mixed precision, augmentation on the GPU).
"""
import shutil
import zipfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release/kaggle_grey_cnn"
NB = ROOT / "notebooks"
TOP = "banglahand-grey-cnn"
RUN = ROOT / "results/final42_2026-10-19"
CODE = ["run_finetuned_grey_cnn.py", "extract_final_features.py", "run_final_experiments.py", "final_study_pipeline.py",
        "run_crop_evaluation.py"]
WEIGHT = "resnet18-f37072fd.pth"

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    md("""# 3. Baseline: ResNet-18 fine-tuned on grey photographs (Kaggle, GPU T4)

**Settings:** Accelerator **GPU T4 x1**. Internet can stay **off** (the ImageNet weights are in the dataset).
Attach the uploaded dataset *banglahand-grey-cnn* (Add data -> Your datasets), then **Save Version -> Save & Run All**.

What it does: fine-tunes ResNet-18 (layer4 + new head, 4 epochs, AdamW) on 224-px tiles of the grey photographs of the
training writers in each fold, over the same 20 repetitions of writer-independent 5-fold cross-validation as the paper,
and reports writer balanced accuracy end to end, with a linear SVM on the fine-tuned features, and after regressing
ink density and photo intensity out of those features.

Runtime: about 1 h on a T4 (the first few minutes cut the tiles). Download **`/kaggle/working/grey_cnn_results.zip`** at the end and send it back."""),
    code(r'''import os, sys, glob, shutil, subprocess, time
from pathlib import Path
KAGGLE = os.environ.get("KAGGLE_ROOT", "/kaggle")          # override only for local testing
hits = glob.glob(f"{KAGGLE}/input/**/dataset/metadata/images.csv", recursive=True)
assert hits, "Attach the uploaded dataset to this notebook (Add data -> Your datasets)."
INPUT = Path(hits[0]).parents[2]
PROJECT = Path(f"{KAGGLE}/working/project"); PROJECT.mkdir(parents=True, exist_ok=True)
if not (PROJECT / "scripts").exists():
    shutil.copytree(INPUT / "code", PROJECT / "scripts")
if not (PROJECT / "dataset").exists():
    os.symlink(INPUT / "dataset", PROJECT / "dataset")
RUN = PROJECT / "results" / "final42_grey"
(RUN / "features").mkdir(parents=True, exist_ok=True)
for f in ["features.npz", "descriptors.csv"]:
    if not (RUN / "features" / f).exists():
        shutil.copy(INPUT / "extra/run_features" / f, RUN / "features" / f)
os.environ["TORCH_HOME"] = str(INPUT / "weights/torch")
import torch; print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("input:", INPUT, "| run folder:", RUN)'''),
    code(r'''t = time.time()
p = subprocess.Popen([sys.executable, "scripts/run_finetuned_grey_cnn.py", "--run", str(RUN)], cwd=PROJECT,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
with open(RUN / "log_grey_cnn.txt", "w") as log:
    for line in p.stdout:
        log.write(line)
        if "done" in line or "grey tiles" in line or "ba_" in line or "mean" in line or "Error" in line:
            print(line.rstrip(), flush=True)
assert p.wait() == 0, "failed, see " + str(RUN / "log_grey_cnn.txt")
print(f"finished in {(time.time() - t) / 60:.0f} min")'''),
    code(r'''import pandas as pd
m = pd.read_csv(RUN / "finetune_grey_metrics_by_split.csv")
display((100 * m.drop(columns="seed").agg(["mean", "std"])).round(1))'''),
    code(r'''keep = ["finetune_grey_writer_predictions.csv", "finetune_grey_metrics_by_split.csv", "finetune_grey_provenance.json", "log_grey_cnn.txt"]
with __import__("zipfile").ZipFile(f"{KAGGLE}/working/grey_cnn_results.zip", "w") as z:
    for f in keep:
        z.write(RUN / f, f)
print("download:", f"{KAGGLE}/working/grey_cnn_results.zip")'''),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True); NB.mkdir(exist_ok=True)
    nb = nbf.v4.new_notebook(); nb["cells"] = CELLS
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb["metadata"]["kaggle"] = {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": False}
    nbf.write(nb, NB / "03_grey_cnn_baseline.ipynb"); shutil.copy(NB / "03_grey_cnn_baseline.ipynb", OUT / "03_grey_cnn_baseline.ipynb")
    zpath = OUT / f"{TOP}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:          # JPEGs do not compress further
        D = ROOT / "dataset"
        for f in sorted(D.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                z.write(f, f"{TOP}/dataset/{f.relative_to(D)}")
        for c in CODE:
            z.write(ROOT / "scripts" / c, f"{TOP}/code/{c}")
        z.write(Path.home() / ".cache/torch/hub/checkpoints" / WEIGHT, f"{TOP}/weights/torch/hub/checkpoints/{WEIGHT}")
        for f in ["features.npz", "descriptors.csv"]:
            z.write(RUN / "features" / f, f"{TOP}/extra/run_features/{f}")
    print("written", zpath, f"{zpath.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
