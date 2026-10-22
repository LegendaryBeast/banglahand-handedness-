#!/usr/bin/env python3
"""Per-image feature extraction for the 38-writer study (deterministic, no labels).

Representations (one row per image, rows aligned with dataset/metadata/images.csv):
  hc183            handcrafted ink descriptor (scripts/run_crop_evaluation.ink_features)
  resnet18         ImageNet ResNet-18, mean of 224x224 tile embeddings (512-d)
  efficientnet_b0  ImageNet EfficientNet-B0, mean of tile embeddings (1280-d)
  vit_b_16         ImageNet ViT-B/16 CLS token, mean of tile embeddings (768-d)
  global_intensity 7 grey-level statistics of the whole photograph (acquisition control)
  background       7 grey-level statistics of paper pixels outside dilated ink (control)
  descriptors      interpretable writer-style measures (slant, orientation histogram,
                   stroke width, ink density) computed at 1536 px width

Tiling preserves aspect ratio and stroke scale: the photograph is resized to
1568 px width (7 tiles), padded with paper colour to whole tiles, and only tiles
containing ink (>=1% of pixels) are embedded. No image is squashed to 224x224.
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
import torchvision
import torchvision.models as tvm
from PIL import Image
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_crop_evaluation import image_to_ink, ink_features  # noqa: E402
import final_study_pipeline as study  # noqa: E402

TILE, TILE_W = 224, 1568
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_gray(path):
    with Image.open(path) as im:
        return im.convert("L").copy()


def tiles(gray):
    w, h = gray.size
    im = gray.resize((TILE_W, round(TILE_W * h / w)), Image.Resampling.LANCZOS)
    a = np.asarray(im, np.float32)
    ink = np.asarray(Image.fromarray(np.uint8(image_to_ink(gray) * 255)).resize(im.size, Image.Resampling.NEAREST)) > 0
    rows = int(np.ceil(a.shape[0] / TILE))
    paper = float(np.median(a))
    pad = rows * TILE - a.shape[0]
    a = np.pad(a, ((0, pad), (0, 0)), constant_values=paper)
    ink = np.pad(ink, ((0, pad), (0, 0)), constant_values=False)
    out, frac = [], []
    for r in range(rows):
        for c in range(TILE_W // TILE):
            sl = (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE))
            out.append(a[sl]); frac.append(ink[sl].mean())
    frac = np.array(frac)
    keep = np.flatnonzero(frac >= 0.01)
    if keep.size == 0:
        keep = np.arange(len(out))
    x = np.stack([out[i] for i in keep]) / 255.0
    x = np.repeat(x[:, None], 3, axis=1)
    x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
    return torch.from_numpy(x.astype(np.float32)), len(keep), len(out)


def backbones(device):
    r18 = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1); r18.fc = torch.nn.Identity()
    eff = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1); eff.classifier = torch.nn.Identity()
    vit = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1); vit.heads = torch.nn.Identity()
    return {k: m.eval().to(device) for k, m in [("resnet18", r18), ("efficientnet_b0", eff), ("vit_b_16", vit)]}


def descriptors(gray, width=1536):
    """Interpretable style measures; unsigned static geometry, not pen dynamics."""
    w, h = gray.size
    a = np.asarray(gray.resize((width, round(width * h / w)), Image.Resampling.LANCZOS), np.float32)
    s = 17 * width / 768
    shade = ndi.gaussian_filter(a, s)
    ink = (shade - a > 17) & (shade > 120) & (a < 205)
    b = int(round(4 * width / 768)); ink[:b] = ink[-b:] = False; ink[:, :b] = ink[:, -b:] = False
    ink = ndi.binary_opening(ink, np.ones((3, 3)))
    f = ndi.gaussian_filter(ink.astype(np.float32), 1.0)
    gy, gx = np.gradient(f)
    mag = np.hypot(gx, gy)
    theta = (np.arctan2(gy, gx) + np.pi) % np.pi            # gradient (edge-normal) direction in [0, pi)
    stroke = (theta + np.pi / 2) % np.pi                     # local stroke direction in [0, pi), image y down
    hist = np.histogram(stroke, bins=18, range=(0, np.pi), weights=mag)[0]
    hist = hist / (hist.sum() + 1e-12)
    # Slant of near-vertical strokes: stroke directions within 30 deg of vertical.
    # Image y points down, so stroke angle pi/2 is vertical; angle < pi/2 leans right.
    near = (np.abs(stroke - np.pi / 2) < np.deg2rad(30)) & (mag > np.quantile(mag[mag > 0], 0.5))
    wts = mag[near]
    slant = float(np.degrees(np.average(np.pi / 2 - stroke[near], weights=wts))) if wts.size else np.nan
    dist = ndi.distance_transform_edt(ink)
    sk = skeletonize(ink)
    # Mean (not median) of the medial-axis distance: the median is quantised at this resolution.
    stroke_width = float(2 * np.mean(dist[sk])) if sk.any() else np.nan
    rows = ink.mean(axis=1)
    return dict(slant_deg=slant, stroke_width_px1536=stroke_width, ink_density=float(ink.mean()),
                horizontal_energy=float(hist[[0, 1, 17]].sum()), vertical_energy=float(hist[[8, 9]].sum()),
                row_profile_cv=float(rows.std() / (rows.mean() + 1e-12))), hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--descriptors-only", action="store_true", help="recompute descriptors.csv only")
    a = ap.parse_args()
    out = ROOT / a.output
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    if a.descriptors_only:
        rows, hists = [], []
        old = pd.read_csv(out / "descriptors.csv").set_index("image_id")
        for r in meta.itertuples():
            d, hist = descriptors(load_gray(ROOT / "dataset" / r.relative_path))
            rows.append(dict(image_id=r.image_id, **d, background_fraction=old.loc[r.image_id, "background_fraction"]))
            hists.append(hist)
        d = pd.DataFrame(rows)
        d[[f"orient_{10 * j:03d}_{10 * j + 10:03d}" for j in range(18)]] = np.asarray(hists)
        d.to_csv(out / "descriptors.csv", index=False)
        print("descriptors recomputed", d.shape)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    nets = backbones(device)
    feats = {k: [] for k in ["hc183", "resnet18", "efficientnet_b0", "vit_b_16", "global_intensity", "background"]}
    desc_rows, hists, tile_counts = [], [], []
    for i, r in enumerate(meta.itertuples()):
        gray = load_gray(ROOT / "dataset" / r.relative_path)
        ink = image_to_ink(gray)
        assert ink.any(), r.image_id
        feats["hc183"].append(ink_features(ink))
        g768 = np.asarray(gray.resize((768, max(32, round(768 * gray.height / gray.width))), Image.Resampling.LANCZOS), float)
        feats["global_intensity"].append(study._intensity_features(g768))
        bg = study.masked_background_features(g768, ink.astype(bool))
        feats["background"].append(bg["features"])
        x, kept, total = tiles(gray)
        tile_counts.append((r.image_id, kept, total))
        with torch.no_grad():
            for k, net in nets.items():
                emb = torch.cat([net(x[j:j + 32].to(device)).float().cpu() for j in range(0, len(x), 32)])
                feats[k].append(emb.mean(0).numpy())
        d, hist = descriptors(gray)
        desc_rows.append(dict(image_id=r.image_id, **d, background_fraction=bg["valid_pixel_fraction"]))
        hists.append(hist)
        if i % 25 == 0:
            print(i, r.image_id, kept, "tiles", flush=True)
    arrays = {k: np.asarray(v, np.float32) for k, v in feats.items()}
    for k, v in arrays.items():
        assert v.shape[0] == len(meta) and np.isfinite(v).all(), k
    np.savez_compressed(out / "features.npz", image_id=meta.image_id.to_numpy(), **arrays)
    d = pd.DataFrame(desc_rows)
    d[[f"orient_{10 * j:03d}_{10 * j + 10:03d}" for j in range(18)]] = np.asarray(hists)
    d.to_csv(out / "descriptors.csv", index=False)
    pd.DataFrame(tile_counts, columns=["image_id", "tiles_embedded", "tiles_total"]).to_csv(out / "tile_counts.csv", index=False)
    (out / "feature_provenance.json").write_text(json.dumps(dict(
        dims={k: int(v.shape[1]) for k, v in arrays.items()}, device=str(device), python=platform.python_version(),
        torch=torch.__version__, torchvision=torchvision.__version__,
        weights=dict(resnet18="IMAGENET1K_V1", efficientnet_b0="IMAGENET1K_V1", vit_b_16="IMAGENET1K_V1"),
        tile=TILE, tile_width=TILE_W, tile_ink_threshold=0.01), indent=2))
    print("saved", {k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    main()
