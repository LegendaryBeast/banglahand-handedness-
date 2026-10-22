# Bangla handedness — 42 writers (Kaggle package)

Upload `bangla-handedness-42.zip` as a **new Kaggle dataset** (Datasets → New dataset → upload the zip; keep it private until the papers are accepted). Then create notebooks from the two `.ipynb` files next to the zip (File → Import notebook) and attach the dataset (Add data → Your datasets).

| notebook | accelerator | time | purpose |
|---|---|---|---|
| `01_dataset_check.ipynb` | none | ~2 min | checksums, counts, sample paragraphs |
| `02_prl_experiments.ipynb` | **GPU T4 x1** | ~6–8 h (QUICK: ~45 min) | every experiment, figure, table and number of the PRL paper |
| `03_grey_cnn_baseline.ipynb` | **GPU T4 x1** | ~1 h | only the ResNet-18 fine-tuned on grey photographs; uses its own smaller package `banglahand-grey-cnn.zip` (`scripts/make_kaggle_grey_cnn_package.py`) |

Contents of the dataset:

```
bangla-handedness-42/
  dataset/                 336 photographs, metadata, SHA256SUMS.txt, README.md
  code/                    analysis scripts (identical to the project's scripts/)
  weights/torch/hub/checkpoints/   ImageNet weights (ResNet-18, EfficientNet-B0, ViT-B/16) -> runs with Internet OFF
  extra/excluded_raw/P3, P19       raw photos of the two right-handed folders not released (pitfall and sensitivity analyses)
  extra/audit/                     dataset validation: SIFT-RANSAC duplicate-screen scores (Fig. 1f) and summary
```

Notes

- Notebook 2 resumes: if a session stops, run all cells again; finished steps are skipped (their outputs are kept in `/kaggle/working` only while the session lives — use "Save Version → Save & Run All (Commit)" for a full unattended run, up to 12 h).
- Results are written to `/kaggle/working/project/results/final42_kaggle/` and zipped to `/kaggle/working/results_final42_kaggle.zip`.
- CPU results (all classical models, permutations) reproduce the paper exactly for the same library versions. CNN/ViT embeddings and the fine-tuned network are computed on a different GPU than in the paper, so their values can differ by about a point; the paper reports the values of the authors' run.
- `QUICK = True` uses 49 instead of 999 permutations (functional test only).
