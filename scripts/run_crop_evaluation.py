#!/usr/bin/env python3
"""
Comprehensive Evaluation of Crop-Expanded Handwriting Datasets (4x and 6x)
with Leakage-Free Stratified Group Leave-One-Writer-Out Cross-Validation.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from scipy.stats import beta
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = (BASE_DIR / "dataset") if (BASE_DIR / "dataset").exists() else BASE_DIR
RESULTS_DIR = (BASE_DIR / "results") if (BASE_DIR / "results").exists() else BASE_DIR

MODELS = {
    "Linear SVM (C=0.1)": lambda: SVC(kernel="linear", C=0.1, probability=True, random_state=2026),
    "Logistic Regression (C=0.1)": lambda: LogisticRegression(C=0.1, max_iter=2000, random_state=2026),
    "Random Forest (n=100)": lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2026),
}
# Pre-specified primary configuration (fixed before looking at results)
PRIMARY = ("dataset_crops_4x", "Linear SVM (C=0.1)")


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

def load_photo(path: Path) -> Image.Image:
    with Image.open(path) as src:
        return ImageOps.exif_transpose(src).convert('L').copy()

def image_to_ink(im: Image.Image, width: int = 768) -> np.ndarray:
    # Aspect-preserving resize: stroke orientation angles must not be distorted
    w, h = im.size
    im = im.resize((width, max(32, round(width * h / w))), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    shade = ndi.gaussian_filter(a, sigma=17)
    ink = (shade - a > 17) & (shade > 120) & (a < 205)
    ink[:4, :] = False; ink[-4:, :] = False; ink[:, :4] = False; ink[:, -4:] = False
    ink = ndi.binary_opening(ink, structure=np.ones((2, 2)))
    return ink.astype(np.float32)

def ink_features(ink: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(ndi.gaussian_filter(ink, .7))
    mag = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) + np.pi) % np.pi
    features = []
    # 4x4 spatial orientation histogram
    for row in np.array_split(np.arange(ink.shape[0]), 4):
        for col in np.array_split(np.arange(ink.shape[1]), 4):
            a = angle[np.ix_(row, col)].ravel()
            m = mag[np.ix_(row, col)].ravel()
            hist = np.histogram(a, bins=9, range=(0, np.pi), weights=m)[0]
            features.extend((hist / (hist.sum() + 1e-8)).tolist())
            features.append(float(ink[np.ix_(row, col)].mean()))
    # Horizontal and vertical ink projections
    for axis in (0, 1):
        proj = ink.mean(axis=axis)
        features.extend(np.quantile(proj, [0, .1, .25, .5, .75, .9, 1]).tolist())
        features.extend([float(proj.mean()), float(proj.std())])
    # Connected component size quantiles
    labels, n = ndi.label(ink)
    sizes = np.bincount(labels.ravel().astype(int))[1:]
    sizes = sizes[sizes >= 3]
    features.extend(np.quantile(sizes, [.1, .25, .5, .75, .9]).tolist() if sizes.size else [0.]*5)
    return np.asarray(features, dtype=np.float32)

def extract(dataset_dir_name: str, meta_csv_name: str):
    df_meta = pd.read_csv(DATASET_DIR / meta_csv_name)
    data_dir = DATASET_DIR / dataset_dir_name
    print(f"Extracting features for {len(df_meta)} crops in {dataset_dir_name}...")
    X_list, P_list = [], []
    for _, row in df_meta.iterrows():
        im = load_photo(data_dir / row["relative_path"])
        X_list.append(ink_features(image_to_ink(im)))
        # Capture-statistics control: global intensity statistics only (no stroke shape)
        raw = np.asarray(im.resize((128, 128)), dtype=np.float32)
        P_list.append([raw.mean(), raw.std(), *np.quantile(raw, [.05, .25, .5, .75, .95])])
    X = np.vstack(X_list)
    P = np.asarray(P_list, dtype=np.float32)
    assert np.isfinite(X).all()
    return df_meta, X, P


def lowo_probs(X, y, writers, make_clf):
    probs = np.empty(len(y), dtype=float)
    for w_held in np.unique(writers):
        tr, te = writers != w_held, writers == w_held
        pipe = make_pipeline(StandardScaler(), make_clf())
        pipe.fit(X[tr], y[tr])
        probs[te] = pipe.predict_proba(X[te])[:, 1]
    return probs


def writer_bacc(probs, y, writers):
    d = pd.DataFrame({"w": writers, "y": y, "p": probs}).groupby("w").agg(y=("y", "first"), p=("p", "mean"))
    return balanced_accuracy_score(d["y"], (d["p"] >= 0.5).astype(int))


def evaluate(name, df_meta, X, P):
    writers = df_meta["writer_id"].to_numpy()
    y = df_meta["label"].to_numpy(dtype=int)
    results, crop_rows = [], []
    for m_name, make_clf in MODELS.items():
        probs = lowo_probs(X, y, writers, make_clf)
        pred = (probs >= 0.5).astype(int)
        df_res = df_meta[["writer_id", "label", "crop_filename"]].copy()
        df_res["prob_left"], df_res["pred_left"] = probs, pred
        df_res["dataset"], df_res["model"] = name, m_name
        crop_rows.append(df_res)
        w = df_res.groupby("writer_id").agg(left=("label", "first"), prob_left=("prob_left", "mean"),
                                            hard=("pred_left", lambda s: int(s.mean() >= 0.5)))
        soft_pred = (w["prob_left"] >= 0.5).astype(int)
        k = int((soft_pred == w["left"]).sum())
        lo, hi = clopper_pearson(k, len(w))
        results.append({
            "dataset": name, "samples": len(df_meta), "model": m_name,
            "crop_accuracy": accuracy_score(y, pred),
            "crop_balanced_accuracy": balanced_accuracy_score(y, pred),
            "writer_correct_soft": k, "writers": len(w),
            "writer_balanced_accuracy_soft": balanced_accuracy_score(w["left"], soft_pred),
            "writer_balanced_accuracy_hard": balanced_accuracy_score(w["left"], w["hard"]),
            "writer_ci95_low": lo, "writer_ci95_high": hi,
        })
        print(f"  {m_name:30s} crop bacc {results[-1]['crop_balanced_accuracy']:.3f} | writers {k}/{len(w)}")

    prov = lowo_probs(P, y, writers, MODELS["Logistic Regression (C=0.1)"])
    prov_crop = balanced_accuracy_score(y, (prov >= 0.5).astype(int))
    prov_w = writer_bacc(prov, y, writers)
    print(f"  Capture-statistics control: crop bacc {prov_crop:.3f}, writer bacc {prov_w:.3f}")
    return results, pd.concat(crop_rows), {"dataset": name, "control_crop_bacc": prov_crop,
                                          "control_writer_bacc": prov_w}


def permutation_test(data, observed):
    """Exact writer-level permutation test: for each of the C(8,4)=70 ways to assign
    4 'left' labels to 8 writers, re-run the full LOWO pipeline for all 6 configs."""
    any_df = next(iter(data.values()))[0]
    writer_ids = sorted(any_df["writer_id"].unique())
    rows = []
    for i, left_set in enumerate(combinations(writer_ids, 4)):
        row = {"perm": i, "left_writers": "|".join(left_set)}
        for ds, (df_meta, X, _) in data.items():
            writers = df_meta["writer_id"].to_numpy()
            y_perm = np.isin(writers, left_set).astype(int)
            for m_name, make_clf in MODELS.items():
                row[f"{ds}|{m_name}"] = writer_bacc(lowo_probs(X, y_perm, writers, make_clf), y_perm, writers)
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  permutations done: {i + 1}/70", flush=True)
    perm = pd.DataFrame(rows)
    cols = [c for c in perm.columns if "|" in c and c not in ("left_writers",)]
    perm["max_over_configs"] = perm[cols].max(axis=1)
    perm.to_csv(RESULTS_DIR / "permutation_null_distribution.csv", index=False)

    out = []
    for key, obs in observed.items():
        p = float((perm[key] >= obs - 1e-9).mean())
        out.append({"config": key, "observed_writer_bacc": obs, "p_exact": p})
    best_obs = max(observed.values())
    out.append({"config": "max_over_6_configs", "observed_writer_bacc": best_obs,
                "p_exact": float((perm["max_over_configs"] >= best_obs - 1e-9).mean())})
    return pd.DataFrame(out)


def main():
    data = {
        "dataset_crops_4x": extract("dataset_crops_4x", "dataset_crops_4x_metadata.csv"),
        "dataset_crops_6x": extract("dataset_crops_6x", "dataset_crops_6x_metadata.csv"),
    }
    all_res, all_crops, controls = [], [], []
    for ds, (df_meta, X, P) in data.items():
        print(f"\n=== {ds} ===")
        r, c, ctl = evaluate(ds, df_meta, X, P)
        all_res += r; all_crops.append(c); controls.append(ctl)

    summary = pd.DataFrame(all_res)
    summary.to_csv(RESULTS_DIR / "crop_evaluation_summary.csv", index=False)
    pd.concat(all_crops).to_csv(RESULTS_DIR / "classical_crop_predictions.csv", index=False)
    pd.DataFrame(controls).to_csv(RESULTS_DIR / "capture_control_summary.csv", index=False)

    observed = {f"{r['dataset']}|{r['model']}": r["writer_balanced_accuracy_soft"] for r in all_res}
    print("\nRunning exact permutation test (70 assignments x 6 configs, full retraining)...")
    pt = permutation_test(data, observed)
    pt["primary"] = pt["config"] == f"{PRIMARY[0]}|{PRIMARY[1]}"
    pt.to_csv(RESULTS_DIR / "permutation_test_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(pt.to_string(index=False))


if __name__ == "__main__":
    main()
