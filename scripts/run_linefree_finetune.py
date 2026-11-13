#!/usr/bin/env python3
"""Fine-tuned grey ResNet-18 on line-free text blocks (IJDAR, printed-border control).

1. Builds the tile cache <dev>/linefree_finetune/features/grey_tiles.npz with run_finetuned_grey_cnn.grey_tiles applied to
   run_lineless_embeddings.lineless_gray (printed lines painted out, everything outside the text block set to paper).
2. Links the PRL run's features.npz and descriptors.csv (metadata order and photometric covariates) into that folder.
3. Runs run_finetuned_grey_cnn.py unchanged (20 x 5 folds, layer4 + head, 4 epochs, AdamW, same seeds) in a separate
   process with --run <dev>/linefree_finetune, so its outputs land there and the PRL results are never touched.
   (Running it in the same process as the OpenCV tile builder segfaulted on macOS/MPS.)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def build_cache(cache):
    import run_finetuned_grey_cnn as ft
    import run_lineless_embeddings as le
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    tiles, index = [], []
    for i, rel in enumerate(meta.relative_path):
        t = ft.grey_tiles(le.lineless_gray(rel)); tiles.append(t); index += [i] * len(t)
        if i % 50 == 0:
            print("line-free grey tiles", i, flush=True)
    np.savez_compressed(cache, tiles=np.concatenate(tiles), image_index=np.asarray(index))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/final42_2026-10-19")
    ap.add_argument("--dev", default="results/ijdar_dev_2026-11-12")
    a = ap.parse_args()
    out = ROOT / a.dev / "linefree_finetune"; feat = out / "features"; feat.mkdir(parents=True, exist_ok=True)
    if not (feat / "grey_tiles.npz").is_file():
        subprocess.run([sys.executable, __file__, "--build-cache-only", str(feat / "grey_tiles.npz")], check=True)
    for name in ("features.npz", "descriptors.csv"):
        link = feat / name
        if not link.exists():
            os.symlink(ROOT / a.base / "features" / name, link)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_finetuned_grey_cnn.py"), "--run", str(out)], check=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--build-cache-only":
        build_cache(Path(sys.argv[2]))
    else:
        main()
