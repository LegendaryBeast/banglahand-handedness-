# Reproducibility guide

This guide explains how to regenerate every number, table and figure in the PRL paper and the Data in Brief article, starting from the released dataset. There are two routes:

- **A. Local** (any machine with Python 3.13; GPU optional): the steps below.
- **B. Kaggle** (free T4 GPU, no installation): build the Kaggle package with `scripts/make_kaggle_package.py` from a local copy of the data, upload it as a Kaggle dataset, then run `notebooks/02_prl_experiments.ipynb`. See `notebooks/KAGGLE_README.md`.

The paper reports the run saved in `results/final42_2026-10-19/`, which is included in this repository, so every number can be checked without rerunning anything.

**Getting the data.** Download the BanglaHand dataset from Mendeley Data (doi:[10.17632/n4sjstdzfy.1](https://doi.org/10.17632/n4sjstdzfy.1)), unzip it and rename the folder to `dataset/` in the repository root.

## 0. Environment

```bash
python -m pip install -r requirements.txt   # pinned versions used for the paper (Python 3.13.9)
python -m pytest -q tests                   # 56 tests: folds, metrics, ink mask, HC-183, STROKE invariances
```

- The ImageNet weights for ResNet-18, EfficientNet-B0 and ViT-B/16 are downloaded by torchvision on first use. For offline runs, use the copies in the Kaggle zip and set `TORCH_HOME`.
- The manuscripts need TeX Live (`pdflatex`, `bibtex`).
- Each script picks the device itself: CUDA, then Apple MPS, then CPU.

## 1. Inputs

| Path | Content |
|---|---|
| `dataset/` | the released dataset: 336 JPEGs, `metadata/images.csv` (one row per image), `writers.csv`, `prompts.csv`, `excluded_writers.csv`, `rh_reordering_provenance.csv`, `SHA256SUMS.txt` |
| `camera_originals/Right-Handed/P3`, `P19` | the two unreleased right-handed camera folders (16 images). **Not public.** They are used only for the pitfall analysis ("+2 folders") and the outlier-folder sensitivity analysis; their features are included in `results/final42_2026-10-19/features/features_excluded.npz`, so those analyses run from the saved features. Steps 1 and 4 below need the folders only for these 16 images |

`scripts/dataset_build/` records how `dataset/` was made from `camera_originals/`: orientation, renumbering, paragraph-order fix and metadata. The steps are already applied. Do not rerun them, because they overwrite `dataset/`.

## 2. Pipeline

Set `R=results/final42_2026-10-19` to rerun into the published folder, or use a new folder name to compare against it. Times were measured on an 8-core Apple M-series laptop; on a Kaggle T4 the GPU steps are faster.

| # | Command | Writes | Time | Used in |
|---|---|---|---|---|
| 1 | `python scripts/validate_dataset.py` | `audit/geometric/` (integrity, SIFT-RANSAC duplicate screen, summary) | ~20 min | DiB §3.3, Fig. 3; PRL §3, Fig. 1f |
| 2 | `python scripts/make_review_sheets.py --output audit/review_sheets` | per-writer contact sheets and the paragraph 7–8 review PDF | ~3 min | manual paragraph-identity review |
| 3 | `python scripts/extract_final_features.py --output $R/base` | HC-183, frozen ResNet-18/EfficientNet-B0/ViT-B/16, photo controls, descriptors | ~10 min (GPU/MPS) | all models |
| 4 | `python scripts/extract_stroke_features.py --base $R/base --output $R/features --excluded 42` | STROKE (208-d), stroke tiles, frozen Stroke-CNN, features of the 16 unreleased images | ~10 min | all models |
| 5 | `python scripts/run_final_experiments.py --run $R` | 20×5-fold writer CV for every representation, classifier and text condition, controls, ablations, learning curves, evidence curve, 999 whole-writer permutations | ~2 h | PRL Tables 1, 3; Figs. 3, 4a, 5, 6 |
| 6 | `python scripts/run_secondary_analyses.py --run $R` | pitfalls, adjustment for ink density and photo intensity, control conditions, truncated-image sensitivity, mirror test | ~10 min | PRL Table 2, Fig. 4c |
| 7a | `python scripts/run_finetuned_stroke_cnn.py --run $R` | fine-tuned stroke ResNet-18 (5×5 folds) | ~50 min (MPS) | results folder only |
| 7b | `python scripts/run_finetuned_stroke_cnn.py --run $R --mirror-augment` | mirror-trained control network | ~55 min (MPS) | results folder only |
| 7c | `python scripts/run_finetuned_grey_cnn.py --run $R` · `summarise_finetuned_grey_cnn.py --run $R` | ResNet-18 fine-tuned on grey photograph tiles (20×5 folds): end to end, SVM on its features, and after the ink/photo adjustment; writer-bootstrap intervals | ~55 min (Kaggle T4) | PRL Tables 1–2, §4.1–4.2 |
| 8 | `python scripts/run_adjusted_permutation.py --run $R` · `run_framing_check.py --run $R` · `run_tilt_check.py --run $R` · `run_stroke_differences.py --run $R` · `run_s3_sensitivity.py --run $R` · `run_revision_analyses.py --run $R` · `run_pseudo_pressure.py --run $R` · `run_adjusted_permutation.py --run $R --reps hc183,combined --output adjusted_permutation_999.csv` | permutation tests after adjustment, acquisition measures, page tilt, STROKE bin differences, outlier-folder sensitivity, capture-time analysis, mirror-augmented STROKE, writer-bootstrap intervals | ~80 min | PRL §5.2, §5.3, §5.5, Tables 1–2, Fig. 7 |
| 9 | `python scripts/make_prl_overview_figures.py --output manuscripts/prl/figures` · `make_prl_figures_v2.py --run $R --output manuscripts/prl/figures` · `make_prl_macros.py --run $R --output manuscripts/prl/figures/results_macros.tex` · `make_dataset_figures.py --output manuscripts/data_in_brief/figures --features $R/features/descriptors.csv` | every PRL/DiB figure, the PRL table rows and `results_macros.tex` (every number in the PRL text) | ~3 min | both papers |
| 10 | `python scripts/verify_manuscript_numbers.py` (authors only: needs the manuscript sources) | checks only: the macros regenerated from `$R` equal the paper's, every macro is defined, the table rows match, and the DiB counts and quoted results match the data and results | 1 min | — |
| 11 | `python scripts/make_submission_folders.py` (authors only) | compiles both papers; `release/prl_submission(.zip)`, `release/dib_submission(.zip)` | 1 min | submission |
| 12 | `python scripts/make_kaggle_package.py` · `python scripts/make_release_packages.py` (authors only) | `release/kaggle/`, `release/dataset/`, `release/code/`, `release/SHA256SUMS.txt` | 3 min | sharing / deposit |

Steps 3–8 are the analysis. The PRL text contains no hand-typed result: the numbers are LaTeX macros from step 9, and the table rows are pasted verbatim from the generated `figures/table_*.tex`. Step 10 fails if anything has drifted.

## 3. What reproduces exactly

- **Deterministic:** dataset validation (exact matching, fixed RANSAC seed), the classical models (HC-183, Style-6, STROKE, Combined, controls), all cross-validation folds (seeds 2026–2045) and all permutation tests (seed 2029). With the pinned library versions they reproduce the saved files exactly.
- **May differ by about a point:** frozen CNN/ViT embeddings and the fine-tuned networks on a different GPU/MPS/CPU or library version, because of floating-point differences. The paper reports the saved run.

## 4. Design facts needed to read the results

- Writer-independent evaluation: every image of a test writer is held out. The endpoint is writer balanced accuracy, averaged over 20 repetitions of stratified 5-fold cross-validation.
- Text conditions: A, shared prompts 1–6; B, own texts 7–8; C, model A applied to 7–8; D/E, matched single shared pairs.
- Primary classifier: linear SVM with C = 0.1 (fixed). Logistic regression and random forests are tuned inside the training writers only.
- Inference: 999 whole-writer label permutations with fold regeneration, p = (b + 1)/1000.
- No writer or image was selected or excluded on the basis of predictions.

## 5. Kaggle route

1. Build the package with `python scripts/make_kaggle_package.py` and upload `release/kaggle/bangla-handedness-42.zip` as a new Kaggle dataset.
2. Import `01_dataset_check.ipynb` (CPU, about 2 min) and `02_prl_experiments.ipynb` (GPU T4, about 6–8 h; `QUICK = True` takes about 45 min, as a functional test only).
   The grey-image fine-tuned ResNet-18 alone can also be run with `03_grey_cnn_baseline.ipynb` (GPU T4, about 1 h) on the smaller package from `python scripts/make_kaggle_grey_cnn_package.py` (`release/kaggle_grey_cnn/`); the paper's values come from this run.
3. Attach the dataset and choose *Save & Run All*. Notebook 2 resumes after a timeout, because finished steps are skipped.
4. Download `results_final42_kaggle.zip` from `/kaggle/working/`. It contains every result file, figure and the macro file.
