#!/usr/bin/env python3
"""STROKE features of line-free text blocks (IJDAR, printed-border control).

stroke_features.stroke_features on run_lineless_embeddings.lineless_gray (printed lines painted out, everything
outside the text block set to paper). Writes <dev>/features/stroke_linefree.npz (key stroke_lf).
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_lineless_embeddings as le  # noqa: E402
import stroke_features as sf  # noqa: E402


def job(rel):
    import cv2
    cv2.setNumThreads(1)
    return sf.stroke_features(le.lineless_gray(rel))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="results/ijdar_dev_2026-11-12")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    with ProcessPoolExecutor(a.workers) as ex:
        X = np.array(list(ex.map(job, meta.relative_path, chunksize=2)), np.float32)
    np.savez_compressed(ROOT / a.dev / "features/stroke_linefree.npz", image_id=meta.image_id.to_numpy(), stroke_lf=X)
    print(X.shape)


if __name__ == "__main__":
    main()
