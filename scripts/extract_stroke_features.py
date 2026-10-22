#!/usr/bin/env python3
"""Build the full feature set of the study from the base features (extract_final_features.py).

* Reuses the label-free per-image base features (HC-183, frozen CNN/ViT, controls, descriptors).
* Adds writing-only representations (scripts/stroke_features.py):
    stroke         208-d direction / hinge / curvature / stroke-ending / slant histograms
    stroke_mirror  the same features of the left-right mirrored ink (diagnostic only)
    stroke_cnn     ImageNet ResNet-18 embedding of black-on-white, stroke-width-normalised ink tiles
* Computes every representation for the 16 images of the two unreleased right-handed camera folders
  (P3, P19) for the pitfall and sensitivity analyses.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models as tvm
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import extract_final_features as ef  # noqa: E402
import final_study_pipeline as study  # noqa: E402
import stroke_features as sf  # noqa: E402

# camera folders not released (rotated upright on load)
EXCLUDED_42 = [("R_rawP3", 0, ROOT / "camera_originals/Right-Handed/P3"),
               ("R_rawP19", 0, ROOT / "camera_originals/Right-Handed/P19")]


def paragraph_files(folder):
    named = sorted(folder.glob("paragraph_*.jpg"))
    return named if named else sorted(folder.glob("*.jpg"))


def load_any(path):
    """Curated files are upright; raw camera files are rotated as in the dataset build (180, or 270 if portrait)."""
    if path.name.startswith("paragraph_"):
        return ef.load_gray(path)
    with Image.open(path) as im:
        w, h = im.size
        return im.rotate(270 if h > w else 180, expand=True).convert("L")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def stroke_reps(gray, net, device):
    v, ink, sw = sf.stroke_features(gray)
    vm, _, _ = sf.stroke_features(gray, mirror=True)
    t = torch.from_numpy(sf.binary_tiles(ink, sw)).float()[:, None].repeat(1, 3, 1, 1)
    t = (t - MEAN) / STD
    with torch.no_grad():
        e = torch.cat([net(t[j:j + 32].to(device)).float().cpu() for j in range(0, len(t), 32)]).mean(0).numpy()
    return v, vm, e, sw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True)
    ap.add_argument("--base", required=True, help="folder with label-free base features.npz/descriptors.csv")
    ap.add_argument("--excluded", choices=["42"], default="42")
    args = ap.parse_args(); out = ROOT / args.output; out.mkdir(parents=True, exist_ok=True)
    base = ROOT / args.base
    EXCLUDED = EXCLUDED_42
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    r18 = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1); r18.fc = torch.nn.Identity(); r18 = r18.eval().to(device)

    # 1. retained images: reuse base features, add stroke representations
    z = np.load(base / "features.npz", allow_pickle=True)
    pos = {k: i for i, k in enumerate(z["image_id"])}
    idx = [pos[i] for i in meta.image_id]
    arrays = {k: z[k][idx] for k in z.files if k != "image_id"}
    S, SM, SC, SW = [], [], [], []
    for i, r in enumerate(meta.itertuples()):
        v, vm, e, sw = stroke_reps(ef.load_gray(ROOT / "dataset" / r.relative_path), r18, device)
        S.append(v); SM.append(vm); SC.append(e); SW.append(sw)
        if i % 25 == 0:
            print("dataset", i, r.image_id, flush=True)
    arrays.update(stroke=np.asarray(S), stroke_mirror=np.asarray(SM), stroke_cnn=np.asarray(SC))
    np.savez_compressed(out / "features.npz", image_id=meta.image_id.to_numpy(), **arrays)
    d = pd.read_csv(base / "descriptors.csv").set_index("image_id").loc[meta.image_id].reset_index()
    d["stroke_width_binary"] = SW
    d.to_csv(out / "descriptors.csv", index=False)

    # 2. excluded images: every representation
    nets = ef.backbones(device)
    rows, X = [], {k: [] for k in ["hc183", "resnet18", "efficientnet_b0", "vit_b_16", "global_intensity", "background",
                                   "style6", "stroke", "stroke_mirror", "stroke_cnn"]}
    for wid, label, folder in EXCLUDED:
        for k, fpath in enumerate(paragraph_files(folder), 1):
            gray = load_any(fpath)
            ink = ef.image_to_ink(gray)
            X["hc183"].append(ef.ink_features(ink))
            g768 = np.asarray(gray.resize((768, max(32, round(768 * gray.height / gray.width))), Image.Resampling.LANCZOS), float)
            X["global_intensity"].append(study._intensity_features(g768))
            X["background"].append(study.masked_background_features(g768, ink.astype(bool))["features"])
            x, _, _ = ef.tiles(gray)
            with torch.no_grad():
                for name, net in nets.items():
                    X[name].append(torch.cat([net(x[j:j + 32].to(device)).float().cpu() for j in range(0, len(x), 32)]).mean(0).numpy())
            dd, _ = ef.descriptors(gray)
            X["style6"].append([dd[c] for c in ["slant_deg", "stroke_width_px1536", "ink_density", "horizontal_energy",
                                                 "vertical_energy", "row_profile_cv"]])
            v, vm, e, _ = stroke_reps(gray, r18, device)
            X["stroke"].append(v); X["stroke_mirror"].append(vm); X["stroke_cnn"].append(e)
            rows.append(dict(image_id=f"{wid}_para{k:02d}", writer_id=wid, label=label, paragraph_number=k))
        print("excluded", wid, flush=True)
    pd.DataFrame(rows).to_csv(out / "excluded_images.csv", index=False)
    np.savez_compressed(out / "features_excluded.npz", **{k: np.asarray(v, np.float64) for k, v in X.items()})
    (out / "feature_provenance.json").write_text(json.dumps(dict(
        reused_from=str(base.relative_to(ROOT)), stroke_dims=len(sf.NAMES), stroke_names=sf.NAMES,
        stroke_cnn="ResNet-18 IMAGENET1K_V1 on binary tiles (stroke width normalised to 3 px, 224 px, >=2% ink)",
        device=str(device), torch=torch.__version__), indent=2))
    print("done")


if __name__ == "__main__":
    main()
