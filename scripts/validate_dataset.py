#!/usr/bin/env python3
"""Validate the released dataset (read-only on images). Writes audit/geometric/{integrity.csv, geometric_pairs.csv.gz, summary.json}.

1. Integrity: every image decodes, is 4080x2296, has no EXIF orientation tag, and matches its SHA-256 in
   metadata/images.csv and SHA256SUMS.txt; files on disk equal the metadata rows.
2. Geometric duplicate screen (SIFT, exact nearest-neighbour matching with ratio test 0.75, RANSAC homography with a
   fixed seed; images resized to 1400 px width, CLAHE). The output is deterministic for a given OpenCV version:
   same_prompt_different_writers  every pair of writers on the same paragraph slot
   same_writer                    every pair of paragraphs within a writer
   unreleased_vs_released         the two unreleased right-handed camera folders (P3, P19) vs every released
                                  right-handed image of the same slot
   The number of RANSAC inliers measures whether two photographs show the same physical ink.
"""
import argparse
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
UNRELEASED = ROOT / "camera_originals/Right-Handed"
WIDTH, NFEAT = 1400, 3000
_F = {}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_gray(path, rot=0):
    if rot:
        a = np.asarray(Image.open(path).rotate(rot, expand=True).convert("L"))
    else:
        a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = a.shape
    a = cv2.resize(a, (WIDTH, int(round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(a)


def extract(item):
    key, path, rot = item
    kp, des = cv2.SIFT_create(nfeatures=NFEAT).detectAndCompute(load_gray(path, rot), None)
    return key, (np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)), des


def _init(f):
    _F.update(f)


def match(pair):
    a, b, kind = pair
    pa, da = _F[a]; pb, db = _F[b]
    out = dict(a=a, b=b, kind=kind, inliers=0)
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return out
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)          # exact matching: deterministic
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return out
    cv2.setRNGSeed(2026)                                              # fixed RANSAC sampling
    _, mask = cv2.findHomography(pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]], cv2.RANSAC, 8.0)
    if mask is not None:
        out["inliers"] = int(mask.sum())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", default="audit/geometric"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); out = ROOT / a.output; out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(DATA / "metadata/images.csv")

    sums = dict(l.split("  ")[::-1] for l in (DATA / "SHA256SUMS.txt").read_text().strip().splitlines())
    rows = []
    for r in meta.itertuples():
        p = DATA / r.relative_path
        with Image.open(p) as im:
            im.load(); size = im.size; ori = im.getexif().get(274)
        h = sha256(p)
        rows.append(dict(image_id=r.image_id, exists=p.is_file(), sha_meta=h == r.sha256, sha_sums=sums.get(r.relative_path) == h,
                         size_ok=size == (4080, 2296), exif_orientation=ori))
    integ = pd.DataFrame(rows); integ.to_csv(out / "integrity.csv", index=False)
    on_disk = sorted(str(p.relative_to(DATA)) for p in DATA.glob("*-Handed/P*/*.jpg"))
    summary = dict(images=len(meta), writers=meta.writer_id.nunique(), per_hand=meta.groupby("hand").writer_id.nunique().to_dict(),
                   files_on_disk=len(on_disk), disk_equals_metadata=set(on_disk) == set(meta.relative_path),
                   **{k: int(integ[k].sum()) for k in ["exists", "sha_meta", "sha_sums", "size_ok"]},
                   exif_orientation_tags=int(integ.exif_orientation.notna().sum()), unique_sha=int(meta.sha256.nunique()))

    items = [(f"released/{r.image_id}", DATA / r.relative_path, 0) for r in meta.itertuples()]
    for fld in ["P3", "P19"]:
        for k, f in enumerate(sorted((UNRELEASED / fld).glob("*.jpg")), 1):
            w, h = Image.open(f).size
            items.append((f"unreleased/{fld}_para{k:02d}", f, 270 if h > w else 180))
    with ProcessPoolExecutor(a.workers) as ex:
        feats = {k: (p, d) for k, p, d in ex.map(extract, items, chunksize=4)}

    slot = lambda key: int(key.rsplit("para", 1)[1])
    wid = lambda key: key.split("/")[1].rsplit("_para", 1)[0]
    rel = [k for k in feats if k.startswith("released/")]
    unrel = [k for k in feats if k.startswith("unreleased/")]
    jobs = []
    for s in range(1, 9):
        jobs += [(x, y, "same_prompt_different_writers") for x, y in itertools.combinations([k for k in rel if slot(k) == s], 2)]
    by_w = {}
    for k in rel:
        by_w.setdefault(wid(k), []).append(k)
    for ks in by_w.values():
        jobs += [(x, y, "same_writer") for x, y in itertools.combinations(ks, 2)]
    jobs += [(x, y, "unreleased_vs_released") for x in unrel for y in rel if y.startswith("released/R_") and slot(x) == slot(y)]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(feats,)) as ex:
        res = pd.DataFrame(list(ex.map(match, jobs, chunksize=48)))
    res.to_csv(out / "geometric_pairs.csv.gz", index=False)

    for kind, g in res.groupby("kind"):
        summary[f"{kind}_pairs"] = len(g); summary[f"{kind}_max_inliers"] = int(g.inliers.max())
    p19 = res[res.a.str.startswith("unreleased/P19") & res.b.str.startswith("released/R_p6_")]
    summary["unreleased_P19_vs_R_p6_inliers"] = [int(p19.inliers.min()), int(p19.inliers.max())]
    top = res.sort_values("inliers", ascending=False).head(25)
    summary["top_pairs"] = top[["a", "b", "kind", "inliers"]].to_dict("records")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "top_pairs"}, indent=2, default=str))


if __name__ == "__main__":
    main()
