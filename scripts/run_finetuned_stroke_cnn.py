#!/usr/bin/env python3
"""Fine-tuned ResNet-18 on binary stroke tiles (declared in protocol_declared_before_fitting.json, amendment).

Writer-independent: for every outer fold the network is trained on tiles of the training writers'
shared paragraphs only, for a fixed number of epochs, and applied to the held-out writers.
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
import torchvision.models as tvm
import torchvision.transforms.v2 as T
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import extract_final_features as ef  # noqa: E402
import run_final_experiments as rf  # noqa: E402
import stroke_features as sf  # noqa: E402

SEEDS = [2026, 2027, 2028, 2029, 2030]
if os.environ.get("HANDEDNESS_SMOKE_SEEDS"):          # functional tests only
    SEEDS = SEEDS[:int(os.environ["HANDEDNESS_SMOKE_SEEDS"])]
EPOCHS, LR, WD, BATCH = 4, 3e-4, 1e-4, 64
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_tiles(meta, cache):
    if cache.is_file():
        z = np.load(cache)
        return z["tiles"], z["image_index"]
    tiles, index = [], []
    for i, r in enumerate(meta.itertuples()):
        ink = sf.ink_mask(ef.load_gray(ROOT / "dataset" / r.relative_path))
        sw, _ = sf.stroke_width(ink)
        t = sf.binary_tiles(ink, sw)
        tiles.append((t * 255).astype(np.uint8)); index += [i] * len(t)
        if i % 50 == 0:
            print("tiles", i, flush=True)
    tiles = np.concatenate(tiles); index = np.asarray(index)
    np.savez_compressed(cache, tiles=tiles, image_index=index)
    return tiles, index


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


def predict(m, tiles, device, flip=False):
    m.eval(); out = []
    with torch.no_grad():
        for j in range(0, len(tiles), 128):
            x = tiles[j:j + 128]
            if flip:
                x = np.ascontiguousarray(x[:, :, ::-1])
            out.append(torch.softmax(m(to_input(x, device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    ap.add_argument("--mirror-augment", action="store_true", help="amendment 1 control: random left-right flips in training")
    args = ap.parse_args(); run = ROOT / args.run
    tag = "finetune_mirroraug" if args.mirror_augment else "finetune"
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    meta = pd.read_csv(ROOT / "dataset/metadata/images.csv")
    tiles, idx = build_tiles(meta, run / "features/binary_tiles.npz")
    w_img, y_img, p_img = meta.writer_id.to_numpy(), meta.label.to_numpy(), meta.paragraph_number.to_numpy()
    writers = np.array(sorted(set(w_img))); wl = np.array([y_img[w_img == u][0] for u in writers])
    aug = T.RandomAffine(degrees=0, translate=(16 / 224, 16 / 224), scale=(0.9, 1.1), fill=1.0)
    rows, t0 = [], time.time()
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
            for ep in range(EPOCHS):
                m.train(); perm = np.random.permutation(len(tr))
                for j in range(0, len(perm), BATCH):
                    b = tr[perm[j:j + BATCH]]
                    x = torch.from_numpy(tiles[b]).float().div(255)[:, None]
                    if args.mirror_augment:
                        flip = torch.rand(len(x)) < 0.5
                        x[flip] = torch.flip(x[flip], dims=[3])
                    x = torch.stack([aug(xx) for xx in x]).repeat(1, 3, 1, 1)
                    x = ((x - MEAN) / STD).to(device)
                    loss = lossf(m(x), torch.from_numpy(y_img[idx[b]]).to(device))
                    opt.zero_grad(); loss.backward(); opt.step()
            test_w = [u for u in writers if fold[u] == f]
            for u in test_w:
                rec = dict(seed=seed, fold=f, writer_id=u, label=int(y_img[w_img == u][0]))
                for name, paras, flip in [("A", (1, 2, 3, 4, 5, 6), False), ("C", (7, 8), False), ("A_mirror", (1, 2, 3, 4, 5, 6), True)]:
                    im = np.flatnonzero((w_img == u) & np.isin(p_img, paras))
                    rec[name] = float(predict(m, tiles[np.isin(idx, im)], device, flip).mean())
                rows.append(rec)
            print(f"seed {seed} fold {f} done {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows); df.to_csv(run / f"{tag}_writer_predictions.csv", index=False)
    out = []
    for seed, g in df.groupby("seed"):
        po = (g.A >= 0.5).astype(int); pc = (g.C >= 0.5).astype(int); pm = (g.A_mirror >= 0.5).astype(int)
        out.append(dict(seed=seed, ba_A=balanced_accuracy_score(g.label, po), ba_C=balanced_accuracy_score(g.label, pc),
                        ba_A_mirror=balanced_accuracy_score(g.label, pm), flip_fraction=float((po != pm).mean())))
    met = pd.DataFrame(out); met.to_csv(run / f"{tag}_metrics_by_split.csv", index=False)
    print(met.round(3).to_string()); print(met.drop(columns="seed").agg(["mean", "std"]).round(3).to_string())
    (run / f"{tag}_provenance.json").write_text(json.dumps(dict(seeds=SEEDS, epochs=EPOCHS, lr=LR, weight_decay=WD, batch=BATCH,
        mirror_augment=bool(args.mirror_augment), n_tiles=int(len(tiles)), device=str(device), torch=torch.__version__, runtime_s=round(time.time() - t0)), indent=2))


if __name__ == "__main__":
    main()
