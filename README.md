# BanglaHand: handedness identification from Bangla handwriting

Code and saved results for the study of writer-independent handedness identification on **BanglaHand**, a balanced dataset of handwritten Bangla paragraphs from 42 writers (21 left-handed, 21 right-handed).

- **Dataset:** BanglaHand, Mendeley Data, V1, doi:[10.17632/n4sjstdzfy.1](https://doi.org/10.17632/n4sjstdzfy.1) (CC BY 4.0). It is described in the data article *BanglaHand: A Dataset of Handwritten Paragraphs for Writer-Independent Handedness Recognition* (Data in Brief, submitted).
- **Research article:** *Handedness identification from Bangla handwriting: A writer-independent evaluation across text conditions* (Pattern Recognition Letters, submitted).

The images are **not** stored in this repository. Download them from Mendeley Data, unzip, and rename the folder to `dataset/` in the repository root.

## Repository layout

```
├── README.md, REPRODUCIBILITY.md, requirements.txt, LICENSE, CITATION.cff
├── scripts/                     analysis pipeline (see REPRODUCIBILITY.md)
│   └── dataset_build/           record of how the released images were prepared from the camera originals
├── tests/                       regression tests (pytest)
├── notebooks/                   Kaggle notebooks: dataset check, full pipeline on a T4 GPU
├── audit/                       dataset validation: SIFT-RANSAC duplicate screen, integrity table, fix log
└── results/final42_2026-10-19/  the run reported in the papers: features, predictions, permutation nulls, summaries
```

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
```

The saved results in `results/final42_2026-10-19/` contain every number reported in the papers. To regenerate them from the images, follow `REPRODUCIBILITY.md`. It lists every command with its outputs and run time; the full pipeline takes about 4–5 hours on an 8-core laptop.

## Main components

- **Evaluation protocol:** writer-independent, stratified five-fold cross-validation over writers (20 repetitions), whole-writer permutation tests, writer-bootstrap intervals, acquisition controls and adjustments.
- **Representations:** HC-183 ink-gradient descriptor; STROKE, 208 writing-only stroke features from binary, size-normalised outlines; frozen ResNet-18, EfficientNet-B0 and ViT-B/16; stroke-tile CNNs.
- **Diagnostics:** mirror test, capture-time analysis, pseudo-pressure measures, leakage pitfalls (writer identification, image-level splits).

## Citation

Please cite the dataset and the articles; see `CITATION.cff`.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0, distributed through Mendeley Data.

## Authors

Tanzim Hasan Prappo (corresponding, tanzimh46@gmail.com), Sayeed Hassan, Sayem Ali, Sukhendra Nath Dhar, M. Shahidur Rahman. Department of Computer Science and Engineering, Shahjalal University of Science and Technology, Sylhet, Bangladesh.
