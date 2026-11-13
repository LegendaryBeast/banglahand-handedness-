#!/usr/bin/env python3
"""Features of the CSAFE Handwriting Database v2 pages for the external application (IJDAR).

Protocol: results/csafe_ijdar_2026-11-12/protocol.json (written before any outcome). Uses the BanglaHand extractors
unchanged; no label is read here.
  stage "hand"  (OpenCV/scipy, multiprocessing): hc183_page, hc183_tb, stroke, the five earlier-method baselines,
                global_intensity, background, ink density, layout measures (run_layout_check.job measures)
  stage "deep"  (PyTorch, separate process): frozen ResNet-18, EfficientNet-B0, ViT-B/16 tile embeddings
Writes <out>/features/csafe_meta.csv, csafe_hand.npz, csafe_deep.npz.
"""
import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CS = ROOT / "external/csafe_v2"
PARA = {(1, "LND", 1): 1, (1, "LND", 2): 2, (1, "LND", 3): 3, (1, "WOZ", 1): 4, (1, "WOZ", 2): 5, (1, "WOZ", 3): 6,
        (2, "LND", 1): 7, (2, "WOZ", 1): 8, (3, "WOZ", 1): 9}


def build_meta():
    survey = pd.read_csv(CS / "surveydata_v2.csv", dtype={"wid": str})
    rows = []
    for f in sorted(CS.glob("selected/Session */*.png")):
        m = re.match(r"w(\d+)_s(\d+)_p(\w+)_r(\d+)\.png", f.name)
        wid, s, p, r = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        rows.append(dict(image_id=f.stem, writer_id=f"CSAFE_{wid}", wid=wid, session=s, prompt=p, repetition=r,
                         paragraph_number=PARA[(s, p, r)], relative_path=str(f.relative_to(CS))))
    meta = pd.DataFrame(rows).merge(survey[["wid", "hand"]], on="wid", how="left")
    meta["label"] = (meta.hand == "left").astype(int)              # 1 = left, as BanglaHand
    assert meta.hand.isin(["left", "right"]).all() and len(meta) == 864
    return meta.sort_values(["writer_id", "paragraph_number"]).reset_index(drop=True)


def load_gray(rel):
    with Image.open(CS / rel) as im:
        return im.convert("L").copy()


def hand_job(rel):
    import cv2
    cv2.setNumThreads(1)
    import extract_final_features as ef
    import extract_prior_baselines as pb
    import final_study_pipeline as study
    import run_crop_evaluation as rc
    import run_layout_check as lc
    import stroke_features as sf
    g = load_gray(rel)
    ink = rc.image_to_ink(g) > 0                                     # 768-px ink, HC-183 rule
    rows = np.flatnonzero(ink.mean(1) > 0.01); cols = np.flatnonzero(ink.mean(0) > 0.005)
    r0, r1 = np.percentile(rows, [0.5, 99.5]).astype(int); c0, c1 = np.percentile(cols, [0.5, 99.5]).astype(int)
    p = 5
    tb = ink[max(0, r0 - p):r1 + p, max(0, c0 - p):c1 + p]
    out = dict(hc183_page=rc.ink_features(ink.astype(np.float32)), hc183_tb=rc.ink_features(tb.astype(np.float32)),
               stroke=sf.stroke_features(g)[0])
    g768 = np.asarray(g.resize((768, max(32, round(768 * g.height / g.width))), Image.Resampling.LANCZOS), float)
    out["global_intensity"] = study._intensity_features(g768)
    out["background"] = study.masked_background_features(g768, ink)["features"]
    d, _ = ef.descriptors(g)
    out["ink_density"] = np.array([d["ink_density"], float(tb.mean())])
    ink15, box = sf.ink_mask(g, return_box=True)
    out["edge_direction"], out["edge_hinge"], out["chain_code"] = pb.contour_baselines(ink15)
    out["hog_glbp"], out["lbp"] = pb.texture_baselines(pb.grey_block(g, box))
    a = np.asarray(g.resize((lc.W, round(lc.W * g.height / g.width)), Image.Resampling.LANCZOS), np.float32)
    lines, info = lc.ruled_lines(a)
    ink_l, _ = lc.ink_of(a); text = ink_l & ~lines
    ys, xs = np.nonzero(text); ys = ys.astype(float); xs = xs.astype(float)
    skew = lc.skew_of(ys, xs); bxl = lc.text_box(text); H, W = a.shape
    lay = dict(ruled_lines=float(len(info["h"]) + len(info["v"])), skew_deg=skew,
               block_cx=(bxl[2] + bxl[3]) / 2 / W, block_cy=(bxl[0] + bxl[1]) / 2 / H,
               block_w=(bxl[3] - bxl[2]) / W, block_h=(bxl[1] - bxl[0]) / H)
    lay.update(lc.line_measures(ys, xs, skew))
    out["layout"] = np.array([lay[k] for k in LAYOUT], float)
    return out


LAYOUT = ["ruled_lines", "skew_deg", "block_cx", "block_cy", "block_w", "block_h", "n_lines", "spacing_rel", "spacing_cv",
          "line_slope_mean", "line_slope_sd", "line_slope_trend", "left_start_sd", "left_start_trend", "right_end_sd",
          "line_len_cv"]


def deep_stage(meta, out):
    import torch
    import extract_final_features as ef
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nets = ef.backbones(device); E = {k: [] for k in nets}
    for i, rel in enumerate(meta.relative_path):
        x, kept, total = ef.tiles(load_gray(rel))
        with torch.no_grad():
            for k, net in nets.items():
                E[k].append(torch.cat([net(x[j:j + 32].to(device)).float().cpu() for j in range(0, len(x), 32)]).mean(0).numpy())
        if i % 100 == 0:
            print("deep", i, rel, kept, "of", total, "tiles", flush=True)
    np.savez_compressed(out / "csafe_deep.npz", image_id=meta.image_id.to_numpy(), **{k: np.array(v, np.float32) for k, v in E.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/csafe_ijdar_2026-11-12")
    ap.add_argument("--stage", choices=["hand", "deep"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(); out = ROOT / a.out / "features"; out.mkdir(parents=True, exist_ok=True)
    meta = build_meta(); meta.drop(columns="hand").to_csv(out / "csafe_meta.csv", index=False)
    if a.stage == "deep":
        deep_stage(meta, out); return
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(hand_job, meta.relative_path, chunksize=2))
    arrays = {k: np.array([r[k] for r in res], np.float64) for k in res[0]}
    np.savez_compressed(out / "csafe_hand.npz", image_id=meta.image_id.to_numpy(), layout_names=np.array(LAYOUT), **arrays)
    print({k: v.shape for k, v in arrays.items()})


if __name__ == "__main__":
    main()
