#!/usr/bin/env python3
"""Baseline: ResNet-18 fine-tuned on grey photograph tiles (post hoc; requested by the pre-submission review).

Same tiles as the frozen ImageNet embeddings (224-px tiles of the grey photograph at 1,568 px width, tiles with
at least 1% ink) and the same recipe as the stroke network: layer4 and a new two-way head are trained on the
training writers' shared paragraphs of each outer fold only, 4 epochs, AdamW (lr 3e-4, weight decay 1e-4),
class-balanced loss, random shifts of up to 16 px and scaling 0.9-1.1 (applied on the GPU, borders replicated),
no flips; mixed precision on CUDA. Folds and seeds are those of
the primary protocol (stratified 5-fold over writers, seeds 2026-2045).

For every fold three writer-level predictions are made for the held-out writers:
  e2e      end to end: mean left-handed probability of the test tiles (condition A and C)
  svm      linear SVM (C = 0.1) on the fine-tuned 512-d embedding averaged per image, trained on the training
           writers' shared paragraphs (condition A)
  svm_adj  the same after regressing ink density and the seven photo-intensity statistics out of the embedding
           inside the training fold (the adjustment used for every other representation)

Writes to --run: finetune_grey_writer_predictions.csv, finetune_grey_metrics_by_split.csv,
finetune_grey_provenance.json and finetune_grey_embeddings.npz (the 512-d fine-tuned embedding of every image, averaged
over its tiles, for every seed and fold; used by summarise_finetuned_grey_cnn.py for the other classifiers and text
conditions); grey tiles are cached in --run/features/grey_tiles.npz.
Covariates come from --run/features/features.npz and descriptors.csv (the saved features of the reported run).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import extract_final_features as ef  # noqa: E402
import run_final_experiments as rf  # noqa: E402

SEEDS = rf.SEEDS                                        # 20 repetitions, as for the primary analyses
if os.environ.get("HANDEDNESS_SMOKE_SEEDS"):            # functional tests only
    SEEDS = SEEDS[:int(os.environ["HANDEDNESS_SMOKE_SEEDS"])]
EPOCHS = int(os.environ.get("GREY_CNN_EPOCHS", 4))      # 4 as declared for the stroke network; env override for tests only
LR, WD, BATCH = 3e-4, 1e-4, 64
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def grey_tiles(gray):
    """uint8 224x224 tiles of the grey photograph, exactly as extract_final_features.tiles (before normalisation)."""
    w, h = gray.size
    im = gray.resize((ef.TILE_W, round(ef.TILE_W * h / w)), Image.Resampling.LANCZOS)
    a = np.asarray(im, np.float32)
    ink = np.asarray(Image.fromarray(np.uint8(ef.image_to_ink(gray) * 255)).resize(im.size, Image.Resampling.NEAREST)) > 0
    rows = int(np.ceil(a.shape[0] / ef.TILE)); paper = float(np.median(a)); pad = rows * ef.TILE - a.shape[0]
    a = np.pad(a, ((0, pad), (0, 0)), constant_values=paper); ink = np.pad(ink, ((0, pad), (0, 0)), constant_values=False)
    out, frac = [], []
    for r in range(rows):
        for c in range(ef.TILE_W // ef.TILE):
            sl = (slice(r * ef.TILE, (r + 1) * ef.TILE), slice(c * ef.TILE, (c + 1) * ef.TILE))
            out.append(a[sl]); frac.append(ink[sl].mean())
    keep = np.flatnonzero(np.array(frac) >= 0.01)
    if keep.size == 0:
        keep = np.arange(len(out))
    return np.stack([out[i] for i in keep]).round().clip(0, 255).astype(np.uint8)


def build_tiles(meta, cache):
    if cache.is_file():
        z = np.load(cache)
        return z["tiles"], z["image_index"]
    tiles, index = [], []
    for i, r in enumerate(meta.itertuples()):
        t = grey_tiles(ef.load_gray(ROOT / "dataset" / r.relative_path))
        tiles.append(t); index += [i] * len(t)
        if i % 50 == 0:
            print("grey tiles", i, flush=True)
    tiles = np.concatenate(tiles); index = np.asarray(index)
    np.savez_compressed(cache, tiles=tiles, image_index=index)
    return tiles, index


def augment(x):
    """Random shift (up to 16 px) and scale (0.9-1.1) of a batch on the device; borders replicated (paper)."""
    b = x.shape[0]
    s = torch.empty(b, device=x.device).uniform_(0.9, 1.1)
    tx, ty = (torch.empty(2, b, device=x.device).uniform_(-16 / 224, 16 / 224) * 2)
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = 1 / s; theta[:, 1, 1] = 1 / s; theta[:, 0, 2] = tx; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


def to_input(x_uint8, device):
    x = torch.from_numpy(x_uint8).float().div(255)[:, None].repeat(1, 3, 1, 1)
    return ((x - MEAN) / STD).to(device)


def make_model(device):
    m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.layer4.parameters():
        p.requires_grad = True
    m.fc = nn.Linear(512, 2)
    return m.to(device)


def infer(m, tiles, device):
    """Left-handed probability and 512-d embedding of every tile."""
    body = nn.Sequential(*list(m.children())[:-1])
    m.eval(); probs, embs = [], []
    with torch.no_grad():
        for j in range(0, len(tiles), 128):
            z = body(to_input(tiles[j:j + 128], device)).flatten(1)
            probs.append(torch.softmax(m.fc(z), 1)[:, 1].cpu().numpy()); embs.append(z.cpu().numpy())
    return np.concatenate(probs), np.concatenate(embs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    args = ap.parse_args(); run = ROOT / args.run
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    D = rf.Data(run)                                     # metadata order and covariates of the reported run
    meta = D.meta
    tiles, idx = build_tiles(meta, run / "features/grey_tiles.npz")
    cov = np.column_stack([D.desc.ink_density.to_numpy(), D.X["global_intensity"]])
    w_img, y_img, p_img = D.w, D.y, D.para
    writers, wl = D.writers, D.wlabels
    n_img = len(meta)
    amp = device.type == "cuda"                          # mixed precision on the GPU (T4) for speed
    torch.backends.cudnn.benchmark = amp
    rows, embs, t0 = [], [], time.time()
    for seed in SEEDS:
        fold = dict(zip(writers, rf.writer_folds(writers, wl, seed)))
        fw_img = np.array([fold[u] for u in w_img])
        for f in range(5):
            torch.manual_seed(seed * 10 + f); np.random.seed(seed * 10 + f)
            tr_img = np.flatnonzero((fw_img != f) & (p_img <= 6))
            tr = np.flatnonzero(np.isin(idx, tr_img))
            ytr = y_img[idx[tr]]
            cw = torch.tensor(len(ytr) / (2 * np.bincount(ytr, minlength=2)), dtype=torch.float32, device=device)
            m = make_model(device)
            opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
            lossf = nn.CrossEntropyLoss(weight=cw)
            scaler = torch.amp.GradScaler("cuda") if amp else None
            for _ in range(EPOCHS):
                m.train(); perm = np.random.permutation(len(tr))
                for j in range(0, len(perm), BATCH):
                    b = tr[perm[j:j + BATCH]]
                    x = augment(torch.from_numpy(tiles[b]).to(device).float().div(255)[:, None]).repeat(1, 3, 1, 1)
                    x = (x - MEAN.to(device)) / STD.to(device)
                    yb = torch.from_numpy(y_img[idx[b]]).to(device)
                    opt.zero_grad()
                    if amp:
                        with torch.autocast("cuda", dtype=torch.float16):
                            loss = lossf(m(x), yb)
                        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    else:
                        loss = lossf(m(x), yb); loss.backward(); opt.step()
            prob, emb = infer(m, tiles, device)
            # per-image means
            cnt = np.bincount(idx, minlength=n_img)
            e_im = np.zeros((n_img, emb.shape[1])); np.add.at(e_im, idx, emb); e_im /= cnt[:, None]
            embs.append(e_im.astype(np.float32))
            # SVM on the fine-tuned embedding, unadjusted and adjusted (training writers' shared paragraphs)
            te_img = np.flatnonzero((fw_img == f) & (p_img <= 6))
            svm_scores = {}
            for name, X in [("svm", e_im), ("svm_adj", None)]:
                if name == "svm_adj":
                    reg = LinearRegression().fit(cov[tr_img], e_im[tr_img])
                    X = e_im - reg.predict(cov)
                clf = rf.estimator("svm"); clf.fit(X[tr_img], y_img[tr_img])
                svm_scores[name] = dict(zip(te_img, clf.decision_function(X[te_img])))
            for u in [u for u in writers if fold[u] == f]:
                a_im = np.flatnonzero((w_img == u) & (p_img <= 6)); c_im = np.flatnonzero((w_img == u) & (p_img >= 7))
                rows.append(dict(seed=seed, fold=f, writer_id=u, label=int(y_img[w_img == u][0]),
                                 A=float(np.mean(prob[np.isin(idx, a_im)])), C=float(np.mean(prob[np.isin(idx, c_im)])),
                                 svm=float(np.mean([svm_scores["svm"][i] for i in a_im])),
                                 svm_adj=float(np.mean([svm_scores["svm_adj"][i] for i in a_im]))))
            print(f"seed {seed} fold {f} done {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(run / "finetune_grey_embeddings.npz", emb=np.stack(embs).reshape(len(SEEDS), 5, n_img, -1),
                        seeds=np.array(SEEDS), image_id=meta.image_id.to_numpy())
    df = pd.DataFrame(rows); df.to_csv(run / "finetune_grey_writer_predictions.csv", index=False)
    out = []
    for seed, g in df.groupby("seed"):
        out.append(dict(seed=seed, ba_A=balanced_accuracy_score(g.label, (g.A >= 0.5).astype(int)),
                        ba_C=balanced_accuracy_score(g.label, (g.C >= 0.5).astype(int)),
                        ba_A_svm=balanced_accuracy_score(g.label, (g.svm >= 0).astype(int)),
                        ba_A_svm_adj=balanced_accuracy_score(g.label, (g.svm_adj >= 0).astype(int))))
    met = pd.DataFrame(out); met.to_csv(run / "finetune_grey_metrics_by_split.csv", index=False)
    print(met.round(3).to_string()); print(met.drop(columns="seed").agg(["mean", "std"]).round(3).to_string())
    (run / "finetune_grey_provenance.json").write_text(json.dumps(dict(
        seeds=SEEDS, epochs=EPOCHS, lr=LR, weight_decay=WD, batch=BATCH, n_tiles=int(len(tiles)), device=str(device),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", torch=torch.__version__,
        runtime_s=round(time.time() - t0)), indent=2))


if __name__ == "__main__":
    main()
