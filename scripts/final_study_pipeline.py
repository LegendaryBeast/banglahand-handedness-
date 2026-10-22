"""Manifest-first, writer-separated classical handedness experiments.

Images are paragraph photographs, not independent writers. Importing this
module never discovers data, downloads weights, trains, or writes results.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from math import comb
from pathlib import Path, PurePosixPath
import platform
import re
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from run_crop_evaluation import image_to_ink, ink_features


LABELS = {"right": 0, "left": 1}
REQUIRED_COLUMNS = [
    "image_id", "source_path", "archive_member", "sha256", "writer_id", "label",
    "condition", "paragraph_instance_id", "prompt_id", "physical_page_id", "pair_id",
    "session_id", "capture_version", "is_canonical", "identity_verified",
    "prompt_verified", "quality_approved", "pilot_status", "pilot_writer_id", "rotation_deg",
    "writing_hand", "label_verified",
]
BOOL_COLUMNS = ["is_canonical", "identity_verified", "prompt_verified", "quality_approved", "label_verified"]
TRUE = {"true", "1", "yes"}
FALSE = {"false", "0", "no"}


class ManifestValidationError(ValueError):
    """Input facts or study eligibility are incomplete or inconsistent."""


def _text(value):
    return "" if pd.isna(value) else str(value).strip()


def _boolean(value):
    token = _text(value).lower()
    if token in TRUE:
        return True
    if token in FALSE:
        return False
    raise ValueError(f"Expected explicit true/false, received {value!r}")


def load_manifest(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _source_path(root, relative):
    relative = _text(relative)
    part = Path(relative)
    if not relative or part.is_absolute() or ".." in part.parts:
        raise ManifestValidationError("source_path must be a nonempty project-relative path without '..'.")
    root = Path(root).resolve()
    result = (root / part).resolve()
    if not result.is_relative_to(root):
        raise ManifestValidationError("Source resolves outside the selected data root.")
    return result


def read_source_bytes(row, root):
    path = _source_path(root, row["source_path"])
    member = _text(row.get("archive_member", ""))
    if member:
        parsed = PurePosixPath(member)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise ManifestValidationError("Unsafe archive member path.")
        with zipfile.ZipFile(path) as archive:
            return archive.read(member)
    return path.read_bytes()


def make_inventory(root, sources=None, output_csv=None):
    """Inventory explicit sources once; never merge ZIP and extracted copies.

    Unknown provenance stays blank/unverified. This output is a mapping template,
    not an analysis-ready manifest. Each source is {'source': relative_path,
    'handedness': 'left'|'right'}; directories must directly contain P* folders.
    """
    root = Path(root).resolve()
    if sources is None:
        sources = [{"source": "LeftHandedNew.zip", "handedness": "left"},
                   {"source": "RightHandNew.zip", "handedness": "right"}]
    rows = []
    seen_sources = set()
    for spec in sources:
        relative = spec.get("source", spec.get("source_path", spec.get("path", "")))
        hand = _text(spec.get("handedness", "")).lower()
        if hand not in LABELS:
            raise ManifestValidationError("Every source needs explicit left/right handedness.")
        path = _source_path(root, relative)
        if path in seen_sources:
            raise ManifestValidationError("The same source was selected twice.")
        seen_sources.add(path)
        if path.is_dir():
            entries = [(str(p.relative_to(root)), "", p.read_bytes(), p.parent.name)
                       for p in sorted(path.glob("P*/*"))
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        else:
            with zipfile.ZipFile(path) as archive:
                entries = []
                for item in sorted(archive.infolist(), key=lambda i: i.filename):
                    p = PurePosixPath(item.filename)
                    if item.is_dir() or "__MACOSX" in p.parts or p.name.startswith("."):
                        continue
                    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
                    if not re.fullmatch(r"P\d+", p.parent.name):
                        raise ManifestValidationError(f"Unexpected writer-folder layout: {item.filename}")
                    entries.append((str(Path(relative)), item.filename, archive.read(item), p.parent.name))
        for source_path, member, raw, participant in entries:
            sha = hashlib.sha256(raw).hexdigest()
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                width, height = image.size
            row = dict.fromkeys(REQUIRED_COLUMNS, "")
            row.update(image_id=f"img_{sha[:20]}", source_path=source_path,
                       archive_member=member, sha256=sha,
                       writer_id=f"{'L' if hand == 'left' else 'R'}_p{int(participant[1:])}",
                       label=LABELS[hand], is_canonical=False, identity_verified=False,
                       prompt_verified=False, quality_approved=False, label_verified=False,
                       filename=PurePosixPath(member or source_path).name,
                       width=width, height=height)
            rows.append(row)
    frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS + ["filename", "width", "height"])
    if frame.empty:
        raise ManifestValidationError("No images in the explicitly selected sources.")
    if frame["sha256"].duplicated().any():
        raise ManifestValidationError("Selected sources contain identical image bytes; choose one source version explicitly.")
    if output_csv is not None:
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite an existing mapping template: {output}")
        frame.to_csv(output, index=False)
    return frame


def _canonical(frame):
    result = frame.copy()
    for name in BOOL_COLUMNS:
        result[name] = result[name].map(_boolean)
    result = result[result["is_canonical"]].copy().reset_index(drop=True)
    result["label"] = pd.to_numeric(result["label"], errors="raise").astype(int)
    return result


def validate_manifest(df, expected_writers=40, expected_per_class=20, root=None, verify_files=False):
    """Return findings without guessing missing participant/page/prompt facts."""
    errors, warnings = [], []
    eligibility = dict.fromkeys(["A", "B", "C", "D", "E"], False)
    report = {"valid": False, "errors": errors, "warnings": warnings, "counts": {},
              "analysis_eligibility": eligibility}
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        errors.append("Missing columns: " + ", ".join(missing))
        return report
    data = df.copy()
    if data.empty:
        errors.append("Manifest has no images.")
        return report
    for name in ["image_id", "source_path", "sha256", "writer_id"]:
        if data[name].map(_text).eq("").any():
            errors.append(f"{name} contains missing values.")
    for _, row in data.iterrows():
        source = Path(_text(row["source_path"]))
        member = PurePosixPath(_text(row["archive_member"]))
        if source.is_absolute() or ".." in source.parts or member.is_absolute() or ".." in member.parts:
            errors.append("Source references must stay within the explicit data root; absolute/parent paths are forbidden.")
            break
    if data["image_id"].duplicated().any():
        errors.append("image_id must be unique across captures.")
    if not data["sha256"].map(lambda s: bool(re.fullmatch(r"[0-9a-fA-F]{64}", _text(s)))).all():
        errors.append("sha256 must contain 64 hexadecimal characters for each image.")
    try:
        numeric_labels = pd.to_numeric(data["label"], errors="raise")
        if not numeric_labels.isin([0, 1]).all():
            errors.append("Label encoding must be right=0, left=1.")
        data["label"] = numeric_labels
    except (ValueError, TypeError):
        errors.append("Labels must be numeric right=0, left=1.")
    if data.groupby("writer_id")["label"].nunique().gt(1).any():
        errors.append("A writer has inconsistent handedness labels.")
    try:
        for name in BOOL_COLUMNS:
            data[name] = data[name].map(_boolean)
    except ValueError as error:
        errors.append(str(error))
        return report
    canonical = data[data["is_canonical"]].copy()
    if canonical.empty:
        errors.append("No canonical images are approved; complete the provenance/quality mapping first.")
        report["counts"] = {"images": len(data), "canonical_images": 0,
                            "writer_folders": int(data["writer_id"].nunique())}
        return report
    for name in ["condition", "paragraph_instance_id", "prompt_id", "physical_page_id", "pair_id",
                 "session_id", "capture_version", "pilot_status", "rotation_deg", "writing_hand"]:
        if canonical[name].map(_text).eq("").any():
            errors.append(f"Canonical images need verified {name} values.")
    for name in ["identity_verified", "prompt_verified", "quality_approved", "label_verified"]:
        if not canonical[name].all():
            errors.append(f"All canonical images must have {name}=true.")
    if not canonical["writing_hand"].isin(LABELS).all():
        errors.append("Actual writing_hand must explicitly be left or right.")
    elif not canonical["writing_hand"].map(LABELS).eq(canonical["label"]).all():
        errors.append("Actual writing hand disagrees with the target label; resolve the target definition before including switched-hand samples.")
    if not canonical["condition"].isin(["shared", "different"]).all():
        errors.append("condition must be shared or different.")
    if not canonical["pilot_status"].isin(["new", "returning"]).all():
        errors.append("pilot_status must explicitly be new or returning.")
    if canonical.groupby("writer_id")["pilot_status"].nunique().gt(1).any():
        errors.append("pilot_status must be consistent within a writer.")
    if canonical.loc[canonical["pilot_status"] == "new", "pilot_writer_id"].map(_text).ne("").any():
        errors.append("A writer marked new has a pilot_writer_id; reconcile development identity before inference.")
    returning = canonical[canonical["pilot_status"] == "returning"]
    if not returning.empty:
        warnings.append("Returning pilot writers remain development participants; OOF exclusion does not establish independent confirmation.")
        if returning["pilot_writer_id"].map(_text).eq("").any():
            errors.append("Returning writers need a persistent pilot_writer_id.")
        links = returning[["writer_id", "pilot_writer_id"]].drop_duplicates()
        if links["writer_id"].duplicated().any() or links["pilot_writer_id"].duplicated().any():
            errors.append("Pilot identity links must be one-to-one.")
    if canonical.duplicated(["writer_id", "paragraph_instance_id"]).any():
        errors.append("Each writer/paragraph instance needs exactly one canonical capture.")
    # A photograph of the same paragraph may recur only as a noncanonical capture.
    for sha, group in data.groupby("sha256"):
        if len(group) > 1:
            if group["writer_id"].nunique() > 1 or group["paragraph_instance_id"].nunique() > 1:
                errors.append(f"Identical image bytes assigned to different writers or paragraphs: {sha[:12]}")
            elif group["is_canonical"].sum() > 1:
                errors.append("Identical image bytes have more than one canonical vote.")
            else:
                warnings.append("Identical noncanonical recapture bytes detected; retained only as provenance.")
    rotations = pd.to_numeric(canonical["rotation_deg"], errors="coerce")
    if not rotations.isin([0, 90, 180, 270]).all():
        errors.append("rotation_deg must explicitly be 0, 90, 180, or 270 degrees counterclockwise.")
    writers = canonical[["writer_id", "label"]].drop_duplicates()
    class_counts = writers["label"].value_counts().to_dict()
    report["counts"] = {"images": len(data), "canonical_images": len(canonical),
                        "writers": len(writers), "right_writers": int(class_counts.get(0, 0)),
                        "left_writers": int(class_counts.get(1, 0)),
                        "shared_images": int(canonical["condition"].eq("shared").sum()),
                        "different_images": int(canonical["condition"].eq("different").sum())}
    if expected_writers is not None and len(writers) != expected_writers:
        errors.append(f"Expected {expected_writers} writers; found {len(writers)}. This is not the completed final cohort.")
    if expected_per_class is not None and any(class_counts.get(k, 0) != expected_per_class for k in [0, 1]):
        errors.append(f"Expected {expected_per_class} writers per class; found right={class_counts.get(0, 0)}, left={class_counts.get(1, 0)}.")
    shared_sets, pair_sets = [], []
    for writer_id, group in canonical.groupby("writer_id", sort=True):
        shared = group[group["condition"] == "shared"]
        different = group[group["condition"] == "different"]
        if len(group) != 8 or len(shared) != 6 or len(different) != 2:
            errors.append(f"{writer_id}: expected 6 shared + 2 different canonical paragraph images.")
        if shared["prompt_id"].nunique() != 6 or different["prompt_id"].nunique() != 2:
            errors.append(f"{writer_id}: require six distinct shared and two distinct additional prompt IDs within their conditions.")
        pages = group.groupby("physical_page_id").size()
        if len(pages) != 4 or not pages.eq(2).all():
            errors.append(f"{writer_id}: expected four physical pages with two paragraphs each.")
        if Counter(shared["pair_id"]) != Counter({"S1": 2, "S2": 2, "S3": 2}) or not different["pair_id"].eq("D").all():
            errors.append(f"{writer_id}: shared pairs must be S1/S2/S3 (two each); different pair must be D.")
        if group.groupby("physical_page_id")["pair_id"].nunique().gt(1).any() or group.groupby("pair_id")["physical_page_id"].nunique().gt(1).any():
            errors.append(f"{writer_id}: pair IDs must map one-to-one to actual physical pages.")
        shared_sets.append(frozenset(shared["prompt_id"]))
        pair_sets.append({pair: frozenset(part["prompt_id"]) for pair, part in shared.groupby("pair_id")})
    if shared_sets and any(value != shared_sets[0] for value in shared_sets):
        errors.append("The six shared prompt IDs must be identical across writers.")
    pair_consistent = bool(pair_sets) and all(value == pair_sets[0] for value in pair_sets)
    if not pair_consistent:
        warnings.append("Shared physical-page prompt pairs differ across writers; D/E require a reconciled page-pair mapping.")
    common_ids = set(canonical.loc[canonical["condition"] == "shared", "prompt_id"])
    other = canonical[canonical["condition"] == "different"]
    common_overlap = bool(common_ids & set(other["prompt_id"]))
    cross_writer_overlap = bool(other.groupby("prompt_id")["writer_id"].nunique().gt(1).any())
    if common_overlap:
        warnings.append("Additional prompt IDs overlap common prompts; C is not a clean unseen-prompt transfer test.")
    if cross_writer_overlap:
        warnings.append("Additional prompt IDs recur across writers; B is varied-text, not uniformly unseen-text; verify C training-prompt exclusion separately.")
    if verify_files:
        if root is None:
            errors.append("verify_files requires an explicit data root.")
        else:
            for _, row in canonical.iterrows():
                try:
                    raw = read_source_bytes(row, root)
                    if hashlib.sha256(raw).hexdigest() != _text(row["sha256"]).lower():
                        raise ValueError("SHA-256 mismatch")
                    with Image.open(io.BytesIO(raw)) as source:
                        source.load()
                except Exception as error:
                    errors.append(f"{row['image_id']}: source verification failed: {error}")
    report["valid"] = not errors
    if not errors:
        eligibility.update(A=True, B=True, C=not common_overlap, D=pair_consistent and not common_overlap, E=pair_consistent)
    return report


def _require(df, expected_writers=40, expected_per_class=20, analyses=("A",)):
    report = validate_manifest(df, expected_writers, expected_per_class)
    if not report["valid"]:
        raise ManifestValidationError("; ".join(report["errors"]))
    unavailable = [a for a in analyses if not report["analysis_eligibility"].get(a, False)]
    if unavailable:
        raise ManifestValidationError("Requested analyses lack verified eligibility: " + ", ".join(unavailable))
    return _canonical(df), report


def load_paragraph(row, root):
    raw = read_source_bytes(row, root)
    if hashlib.sha256(raw).hexdigest() != _text(row["sha256"]).lower():
        raise ManifestValidationError(f"Source hash changed: {row['image_id']}")
    with Image.open(io.BytesIO(raw)) as source:
        source.load()
        image = ImageOps.exif_transpose(source).convert("L")
    rotation = int(float(row["rotation_deg"]))
    if rotation not in [0, 90, 180, 270]:
        raise ManifestValidationError("Invalid explicit rotation.")
    if rotation:
        image = image.rotate(rotation, expand=True)
    roi = [_text(row.get(key, "")) for key in ["roi_x0", "roi_y0", "roi_x1", "roi_y1"]]
    if any(roi):
        if not all(roi):
            raise ManifestValidationError("ROI needs all four pixel coordinates or none.")
        x0, y0, x1, y1 = map(int, roi)
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ManifestValidationError("ROI is outside the explicitly rotated paragraph image.")
        image = image.crop((x0, y0, x1, y1))
    return image


def extract_features(df, root):
    """Rows match input order; pass canonical rows used by run_classical_study."""
    if not all(df["is_canonical"].map(_boolean)):
        raise ManifestValidationError("Feature extraction requires a canonical-only manifest.")
    features = []
    for _, row in df.iterrows():
        ink = image_to_ink(load_paragraph(row, root))
        if not ink.any():
            raise ManifestValidationError(f"No ink extracted: {row['image_id']}; review image/segmentation before analysis.")
        features.append(ink_features(ink))
    result = np.asarray(features, dtype=np.float32)
    if result.shape != (len(df), 183) or not np.isfinite(result).all():
        raise ManifestValidationError("Invalid handcrafted feature matrix.")
    return result


def _intensity_features(values):
    values = np.asarray(values, dtype=float)
    return np.array([values.mean(), values.std(), *np.quantile(values, [.05, .25, .5, .75, .95])])


def masked_background_features(gray, ink_mask, dilation_iterations=3):
    gray, ink_mask = np.asarray(gray, dtype=float), np.asarray(ink_mask, dtype=bool)
    if gray.ndim != 2 or gray.shape != ink_mask.shape:
        raise ValueError("Gray image and ink mask must be matching two-dimensional arrays.")
    if not np.isfinite(gray).all():
        raise ValueError("Background control pixels must be finite.")
    if dilation_iterations < 0:
        raise ValueError("Mask dilation must be nonnegative.")
    excluded = ndi.binary_dilation(ink_mask, iterations=dilation_iterations) if dilation_iterations else ink_mask
    mask = ~excluded
    if mask.sum() < max(10, int(gray.size * .05)):
        raise ManifestValidationError("Insufficient unmasked paper for the background control.")
    return {"features": _intensity_features(gray[mask]),
            "valid_pixel_fraction": float(mask.mean()), "mask": mask}


def extract_controls(df, root):
    global_values, background_values, fractions = [], [], []
    for _, row in df.iterrows():
        image = load_paragraph(row, root)
        image = image.resize((768, max(32, round(768 * image.height / image.width))), Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=float)
        ink = image_to_ink(image)
        values = masked_background_features(array, ink)
        global_values.append(_intensity_features(array))
        background_values.append(values["features"])
        fractions.append(values["valid_pixel_fraction"])
    return {"global_intensity": np.asarray(global_values),
            "masked_background": np.asarray(background_values),
            "valid_pixel_fraction": np.asarray(fractions)}


def save_preprocessing_qc(df, root, output_dir):
    """Save prediction-free raw/mask panels and diagnostics before data freeze.

    Image IDs are shown instead of handedness labels. Diagnostic fraction flags
    request review and do not automatically exclude or tune any observation.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a new preprocessing-QC output directory.")
    diagnostics, thumbnails = [], []
    for _, row in df.iterrows():
        image = load_paragraph(row, root)
        image = image.resize((768, max(32, round(768 * image.height / image.width))), Image.Resampling.LANCZOS)
        raw = np.asarray(image, dtype=float)
        ink = image_to_ink(image).astype(bool)
        fraction = float(ink.mean())
        background, background_error = None, ""
        if ink.any():
            try:
                background = masked_background_features(raw, ink)
            except ManifestValidationError as error:
                background_error = str(error)
        diagnostics.append({"image_id": row["image_id"], "condition": row["condition"],
                            "prompt_id": row["prompt_id"], "ink_fraction": fraction,
                            "empty_mask": not ink.any(),
                            "review_extreme_ink_fraction": fraction < .001 or fraction > .6,
                            "background_control_review": background_error,
                            "masked_background_fraction": background["valid_pixel_fraction"] if background else np.nan,
                            "rotation_deg": int(float(row["rotation_deg"]))})
        views = [image.convert("RGB"), Image.fromarray(np.uint8((~ink) * 255)).convert("RGB")]
        masked = np.asarray(image.convert("RGB")).copy()
        if background is not None:
            masked[~background["mask"]] = [255, 180, 180]
        views.append(Image.fromarray(masked))
        for view in views:
            view.thumbnail((310, 180), Image.Resampling.LANCZOS)
        thumbnails.append((str(row["image_id"]), views))
    for start in range(0, len(thumbnails), 8):
        batch = thumbnails[start:start + 8]
        panel = Image.new("RGB", (960, 45 + 215 * len(batch)), "white")
        draw = ImageDraw.Draw(panel)
        for column, title in enumerate(["Entire paragraph, explicit rotation", "Extracted ink", "Excluded from background in pink"]):
            draw.text((column * 320 + 5, 10), title, fill="black")
        for ri, (image_id, views) in enumerate(batch):
            y = 45 + ri * 215
            for ci, view in enumerate(views):
                panel.paste(view, (ci * 320 + 5, y))
            draw.text((5, y + 185), image_id, fill="black")
        panel.save(output / f"preprocessing_{start // 8 + 1:03d}.jpg", quality=92)
    frame = pd.DataFrame(diagnostics)
    frame.to_csv(output / "preprocessing_diagnostics.csv", index=False)
    return frame


def make_writer_folds(df, n_splits=5, seed=2026, writer_labels=None):
    writers = df[["writer_id", "label"]].drop_duplicates().sort_values("writer_id").reset_index(drop=True)
    if writers["writer_id"].duplicated().any():
        raise ManifestValidationError("Each writer needs one label.")
    if writer_labels is not None:
        if isinstance(writer_labels, dict) or isinstance(writer_labels, pd.Series):
            writers["label"] = writers["writer_id"].map(writer_labels)
        else:
            if len(writer_labels) != len(writers):
                raise ValueError("writer_labels must align with sorted writer IDs.")
            writers["label"] = writer_labels
    labels = pd.to_numeric(writers["label"], errors="raise")
    if not labels.isin([0, 1]).all() or set(labels) != {0, 1}:
        raise ManifestValidationError("Fold labels must include right=0 and left=1.")
    writers["label"] = labels.astype(int)
    if n_splits < 2 or writers["label"].value_counts().min() < n_splits:
        raise ManifestValidationError("Each class must have at least n_splits writers.")
    writers["fold"] = -1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(writers["writer_id"], writers["label"])):
        writers.loc[test, "fold"] = fold
    return writers


def _estimator(name, parameters=None):
    parameters = parameters or {}
    if name == "svm":
        model = SVC(kernel="linear", C=.1, probability=False, **parameters)
    elif name == "logistic":
        model = LogisticRegression(C=parameters.get("C", .1), max_iter=2000, random_state=2026)
    elif name == "random_forest":
        model = RandomForestClassifier(n_estimators=100, max_depth=parameters.get("max_depth", 5), random_state=2026, n_jobs=1)
    else:
        raise ValueError(f"Unsupported implemented model: {name}")
    return make_pipeline(StandardScaler(), model)


def _scores(model, name, X):
    if list(model.classes_) != [0, 1]:
        raise AssertionError("Positive score must encode left=1, with classes_=[0,1].")
    return model.decision_function(X) if name == "svm" else model.predict_proba(X)[:, 1]


def _threshold(name):
    return 0.0 if name == "svm" else .5


def _fit(name, X, frame, indices, inner_splits, seed):
    best = {}
    if name != "svm":
        choices = [{"C": c} for c in [.01, .1, 1., 10.]] if name == "logistic" else [{"max_depth": d} for d in [3, 5, None]]
        subset = frame.iloc[indices].copy().reset_index(drop=True)
        inner = make_writer_folds(subset, inner_splits, seed)
        scored = []
        for params in choices:
            predictions = []
            for fold in range(inner_splits):
                test_ids = set(inner.loc[inner["fold"] == fold, "writer_id"])
                tr = np.flatnonzero(~subset["writer_id"].isin(test_ids).to_numpy())
                te = np.flatnonzero(subset["writer_id"].isin(test_ids).to_numpy())
                candidate = _estimator(name, params)
                candidate.fit(X[indices[tr]], subset.iloc[tr]["label"].to_numpy())
                scored_rows = subset.iloc[te][["writer_id", "label"]].copy()
                scored_rows["score"] = _scores(candidate, name, X[indices[te]])
                predictions.append(scored_rows)
            scores = pd.concat(predictions).groupby("writer_id").agg(label=("label", "first"), score=("score", "mean"))
            scored.append(balanced_accuracy_score(scores["label"], (scores["score"] >= _threshold(name)).astype(int)))
        # First declared choice wins ties; no outer-test selection.
        best = choices[int(np.argmax(scored))]
    pipeline = _estimator(name, best)
    pipeline.fit(X[indices], frame.iloc[indices]["label"].to_numpy())
    return pipeline, best


def _bootstrap_stat(values, labels, resamples, seed):
    values, labels = np.asarray(values, dtype=float), np.asarray(labels, dtype=int)
    if resamples < 1 or set(labels) != {0, 1}:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.zeros(resamples)
    for label in [0, 1]:
        group = values[labels == label]
        draws += .5 * group[rng.integers(0, len(group), size=(resamples, len(group)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(draws, [.025, .975]))


def summarize_writer_predictions(writers, bootstrap_resamples=10000, seed=2028):
    if writers.groupby("writer_id")["label"].nunique().gt(1).any():
        raise ManifestValidationError("Writer labels disagree across analyses.")
    rows = []
    for (model, feature_set, analysis), frame in writers.groupby(["model", "feature_set", "analysis_id"], sort=True):
        if frame["writer_id"].duplicated().any():
            raise AssertionError("A writer has multiple predictions in one analysis/model.")
        y, pred = frame["label"].to_numpy(), frame["prediction"].to_numpy()
        if set(y) != {0, 1}:
            raise ManifestValidationError("Balanced writer metrics require both classes in the eligible cohort.")
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        low, high = _bootstrap_stat(pred == y, y, bootstrap_resamples, seed)
        aucs, auc_weights = [], []
        for _, group in frame.groupby("fold"):
            if group["label"].nunique() == 2:
                aucs.append(roc_auc_score(group["label"], group["score"]))
                auc_weights.append(int(group["label"].eq(0).sum() * group["label"].eq(1).sum()))
        rows.append({"model": model, "feature_set": feature_set, "analysis_id": analysis,
                     "n_writers": len(frame), "n_right": int((y == 0).sum()), "n_left": int((y == 1).sum()),
                     "n_correct": int((pred == y).sum()), "balanced_accuracy": balanced_accuracy_score(y, pred),
                     "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                     "recall_right": tn / (tn + fp), "recall_left": tp / (tp + fn),
                     "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                     "ci95_low": low, "ci95_high": high,
                     "fold_auc_mean": float(np.average(aucs, weights=auc_weights)) if aucs else np.nan,
                     "fold_auc_n": len(aucs), "fold_auc_pair_count": int(sum(auc_weights)),
                     "auc_rule": "within-fold AUC weighted by eligible within-fold positive-negative writer pairs",
                     "uncertainty": "conditional stratified writer resampling interval; not validated population/CV coverage"})
    return pd.DataFrame(rows)


def paired_comparisons(writers, bootstrap_resamples=10000, seed=2028):
    if writers.groupby("writer_id")["label"].nunique().gt(1).any():
        raise ManifestValidationError("Paired conditions disagree on a writer's label.")
    rows = []
    for (model, feature_set), frame in writers.groupby(["model", "feature_set"]):
        scores = frame.assign(correct=(frame["prediction"] == frame["label"]).astype(float))
        table = scores.pivot(index="writer_id", columns="analysis_id", values="correct")
        labels = scores.groupby("writer_id")["label"].first()
        comparisons = [("transfer_vs_shared2", "C", ["D_S1", "D_S2", "D_S3"]),
                       ("varied_vs_matched_shared2", "B", ["E_S1", "E_S2", "E_S3"]),
                       ("varied2_vs_shared6_descriptive", "B", ["A"])]
        for name, target, reference in comparisons:
            if not {target, *reference}.issubset(table.columns):
                continue
            paired = table[[target, *reference]].dropna()
            y = labels.loc[paired.index].to_numpy()
            if set(y) != {0, 1}:
                raise ManifestValidationError("A paired balanced-accuracy comparison needs eligible writers from both classes.")
            difference = (paired[target] - paired[reference].mean(axis=1)).to_numpy()
            estimate = .5 * (difference[y == 0].mean() + difference[y == 1].mean())
            low, high = _bootstrap_stat(difference, y, bootstrap_resamples, seed)
            rows.append({"model": model, "feature_set": feature_set, "comparison": name,
                         "estimate": estimate, "ci95_low": low, "ci95_high": high, "n_writers": len(paired),
                         "scope": "paired conditional OOF difference; no causal text-effect or significance claim"})
    return pd.DataFrame(rows)


def summarize_paragraph_predictions(paragraphs, by_prompt=False):
    """Descriptive OOF image/prompt results, without independent-image inference.

    Writer-specific prompts can have only one writer and one class. Their
    accuracy is descriptive; two-class balanced accuracy/F1 are then undefined.
    No pooled margin AUC or paragraph-level uncertainty is manufactured here.
    """
    keys = ["model", "feature_set", "analysis_id"] + (["prompt_id"] if by_prompt else [])
    rows = []
    for values, frame in paragraphs.groupby(keys, sort=True):
        if frame["image_id"].duplicated().any():
            raise AssertionError("An image has repeated OOF predictions within one analysis/model.")
        y = frame["label"].to_numpy()
        pred = (frame["score"] >= frame["threshold"]).astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        both = set(y) == {0, 1}
        rows.append(dict(zip(keys, values)) | {
            "n_images": len(frame), "n_writers": frame["writer_id"].nunique(),
            "n_right_images": int((y == 0).sum()), "n_left_images": int((y == 1).sum()),
            "n_correct": int((pred == y).sum()), "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred) if both else np.nan,
            "macro_f1": f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0) if both else np.nan,
            "recall_right": tn / (tn + fp) if tn + fp else np.nan,
            "recall_left": tp / (tp + fn) if tp + fn else np.nan,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "status": "both classes observed" if both else "single class: two-class balanced metrics not estimable",
            "scope": "descriptive OOF paragraph results; correlated images do not add independent writers"})
    return pd.DataFrame(rows)


def _write_outputs(result, output_dir, parameters):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory; existing results will not be overwritten: {output}")
    for name, frame in result.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    versions = {name: __import__(name).__version__ for name in ["numpy", "pandas", "scipy", "sklearn", "PIL"]}
    (output / "run_metadata.json").write_text(json.dumps({"parameters": parameters, "versions": versions,
                                                          "pipeline_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                          "feature_source_sha256": hashlib.sha256(Path(__file__).with_name("run_crop_evaluation.py").read_bytes()).hexdigest(),
                                                          "python": platform.python_version()}, indent=2), encoding="utf-8")


def run_classical_study(df, X, models=("svm",), seed=2026, n_splits=5, inner_splits=4,
                        analyses=("A", "B", "C", "D", "E"), output_dir=None,
                        expected_writers=40, expected_per_class=20, bootstrap_resamples=10000,
                        feature_set="handcrafted183"):
    analyses = tuple(analyses)
    if not analyses or not set(analyses).issubset({"A", "B", "C", "D", "E"}):
        raise ValueError("Choose analyses from A/B/C/D/E.")
    frame, report = _require(df, expected_writers, expected_per_class, analyses)
    X = np.asarray(X, dtype=float)
    if len(frame) != len(df):
        raise ManifestValidationError("Pass canonical-only rows and features; filter both together before fitting.")
    if X.ndim != 2 or len(X) != len(frame) or not np.isfinite(X).all():
        raise ValueError("Features must be finite and aligned one-to-one with canonical manifest rows.")
    if feature_set == "handcrafted183" and X.shape[1] != 183:
        raise ValueError("The declared handcrafted183 representation must have exactly 183 columns.")
    if not models or len(set(models)) != len(models):
        raise ValueError("Specify distinct implemented models.")
    for model_name in models:
        _estimator(model_name)
    folds = make_writer_folds(frame, n_splits, seed)
    paragraph_rows, selection_rows = [], []
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame[REQUIRED_COLUMNS].astype(str), index=False).values.tobytes() + np.ascontiguousarray(X).tobytes()).hexdigest()[:12]
    for model_name in models:
        for fold in range(n_splits):
            test_ids = set(folds.loc[folds["fold"] == fold, "writer_id"])
            is_test = frame["writer_id"].isin(test_ids).to_numpy()
            conditions = []
            if set(analyses) & {"A", "C", "D"}:
                conditions.append(("A", frame["condition"].eq("shared").to_numpy()))
            if "B" in analyses:
                conditions.append(("B", frame["condition"].eq("different").to_numpy()))
            if "E" in analyses:
                conditions.extend((f"E_{pair}", frame["pair_id"].eq(pair).to_numpy()) for pair in ["S1", "S2", "S3"])
            for fit_condition, selected in conditions:
                train = np.flatnonzero(~is_test & selected)
                test = np.flatnonzero(is_test & selected)
                if set(frame.iloc[train]["writer_id"]) & test_ids:
                    raise AssertionError("Outer writer leakage.")
                fitted, params = _fit(model_name, X, frame, train, inner_splits, seed + 1)
                recipe = {"dataset_features": digest, "model": model_name, "fit_condition": fit_condition,
                          "fold": fold, "outer_seed": seed, "inner_seed": seed + 1,
                          "inner_splits": inner_splits, "feature_set": feature_set, "selected_params": params,
                          "training_image_ids": sorted(frame.iloc[train]["image_id"]),
                          "pipeline_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
                fitted_id = f"{model_name}_{fit_condition}_fold{fold}_" + hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
                selection_rows.append({"model": model_name, "feature_set": feature_set, "fold": fold,
                                       "fit_condition": fit_condition, "fitted_model_id": fitted_id,
                                       "train_writer_ids": "|".join(sorted(set(frame.iloc[train]["writer_id"]))),
                                       "test_writer_ids": "|".join(sorted(test_ids)),
                                       "n_train_images": len(train), "n_test_images": len(test),
                                       "best_params": json.dumps(params, sort_keys=True),
                                       "resolved_estimator_params": json.dumps(fitted.steps[-1][1].get_params(), sort_keys=True),
                                       "tuning_status": "fixed_no_inner_search" if model_name == "svm" else "inner_writer_selection"})
                targets = []
                if fit_condition == "A":
                    if "A" in analyses:
                        targets.append(("A", test))
                    if "C" in analyses or "D" in analyses:
                        targets.append(("C", np.flatnonzero(is_test & frame["condition"].eq("different").to_numpy())))
                    if "D" in analyses:
                        targets.extend((f"D_{pair}", np.flatnonzero(is_test & frame["pair_id"].eq(pair).to_numpy())) for pair in ["S1", "S2", "S3"])
                else:
                    targets.append((fit_condition, test))
                for analysis, indices in targets:
                    scores = _scores(fitted, model_name, X[indices])
                    for index, score in zip(indices, scores):
                        row = frame.iloc[index]
                        paragraph_rows.append({"analysis_id": analysis, "model": model_name, "feature_set": feature_set,
                                               "fold": fold, "seed": seed, "writer_id": row["writer_id"], "label": int(row["label"]),
                                               "image_id": row["image_id"], "paragraph_instance_id": row["paragraph_instance_id"],
                                               "prompt_id": row["prompt_id"], "physical_page_id": row["physical_page_id"],
                                               "capture_version": row["capture_version"], "session_id": row["session_id"],
                                               "pair_id": row["pair_id"], "score": float(score),
                                               "threshold": _threshold(model_name), "fitted_model_id": fitted_id,
                                               "pilot_status": row["pilot_status"]})
    paragraph = pd.DataFrame(paragraph_rows)
    keys = ["analysis_id", "model", "feature_set", "fold", "writer_id", "label", "threshold", "fitted_model_id", "pilot_status"]
    writers = paragraph.groupby(keys, as_index=False).agg(score=("score", "mean"),
                                                         n_paragraphs=("paragraph_instance_id", "nunique"),
                                                         physical_pages=("physical_page_id", "nunique"))
    writers["prediction"] = (writers["score"] >= writers["threshold"]).astype(int)
    result = {"paragraph_predictions": paragraph, "writer_predictions": writers,
              "metrics": summarize_writer_predictions(writers, bootstrap_resamples),
              "paragraph_metrics": summarize_paragraph_predictions(paragraph),
              "prompt_metrics": summarize_paragraph_predictions(paragraph, by_prompt=True),
              "paired_comparisons": paired_comparisons(writers, bootstrap_resamples),
              "fold_assignments": folds, "model_selection": pd.DataFrame(selection_rows)}
    naive = writers[writers["pilot_status"] == "new"].copy()
    if naive["label"].nunique() == 2:
        result["development_naive_metrics"] = summarize_writer_predictions(naive, bootstrap_resamples)
        result["development_naive_metrics"]["population"] = "development-naive test writers; original OOF fits retained"
        result["development_naive_paired_comparisons"] = paired_comparisons(naive, bootstrap_resamples)
    else:
        result["development_naive_metrics"] = pd.DataFrame([{
            "population": "development-naive test writers", "n_writers": naive["writer_id"].nunique(),
            "status": "not estimable: both eligible handedness classes are required"}])
        result["development_naive_paired_comparisons"] = pd.DataFrame()
    if output_dir is not None:
        _write_outputs(result, output_dir, {"models": list(models), "analyses": list(analyses), "fold_seed": seed,
                                          "n_splits": n_splits, "inner_splits": inner_splits,
                                          "bootstrap_resamples": bootstrap_resamples, "feature_set": feature_set,
                                          "manifest_feature_digest": digest, "validation_warnings": report["warnings"],
                                          "label_encoding": LABELS, "primary_C": .1, "probability": False})
    return result


def run_primary_permutation(df, X, n_permutations=999, seed=2029, fold_seed=2026,
                            n_splits=5, exchangeability_confirmed=False, output_dir=None,
                            expected_writers=40, expected_per_class=20,
                            allow_development_overlap=False, writer_blocks=None):
    frame, _ = _require(df, expected_writers, expected_per_class, ("A",))
    if exchangeability_confirmed is not True:
        raise ManifestValidationError("Document the exchangeability/capture design before enabling writer-label permutations.")
    if not isinstance(allow_development_overlap, bool):
        raise ValueError("allow_development_overlap must be an explicit boolean.")
    if frame["pilot_status"].eq("returning").any() and not allow_development_overlap:
        raise ManifestValidationError("Returning development writers require new-writer-only inference or an explicit descriptive-overlap analysis; recapture is not independent confirmation.")
    X = np.asarray(X, dtype=float)
    if len(df) != len(frame) or X.ndim != 2 or len(X) != len(frame) or not np.isfinite(X).all():
        raise ValueError("Pass canonical-only rows with aligned finite features.")
    if X.shape[1] != 183:
        raise ValueError("The primary permutation tests the declared 183-D representation only.")
    if not isinstance(n_permutations, int) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer; no exhaustive enumeration is used.")
    writers = frame[["writer_id", "label"]].drop_duplicates().sort_values("writer_id")
    base_labels = writers["label"].to_numpy(dtype=int)
    shared = frame["condition"].eq("shared").to_numpy()
    if writer_blocks is None:
        blocks = np.repeat("unrestricted", len(writers))
        exchangeability_mode = "unrestricted_writer_labels"
    else:
        mapping = pd.Series(writer_blocks)
        if not set(writers["writer_id"]).issubset(mapping.index):
            raise ManifestValidationError("Every writer needs an explicit exchangeability block.")
        mapped = writers["writer_id"].map(mapping)
        if mapped.map(_text).eq("").any():
            raise ManifestValidationError("Exchangeability block IDs cannot be missing.")
        blocks = mapped.astype(str).to_numpy()
        exchangeability_mode = "restricted_within_writer_blocks"
    block_indices = [np.flatnonzero(blocks == block) for block in sorted(set(blocks))]
    permitted_assignments = 1
    block_counts = {}
    for indices in block_indices:
        left = int(base_labels[indices].sum())
        permitted_assignments *= comb(len(indices), left)
        block_counts[str(blocks[indices[0]])] = {"writers": len(indices), "left": left, "right": len(indices) - left}
    if permitted_assignments < 2:
        raise ManifestValidationError("The exchangeability blocks permit no label changes; the restricted permutation test is uninformative.")

    def statistic(labels):
        assignments = dict(zip(writers["writer_id"], labels))
        folds = make_writer_folds(frame, n_splits, fold_seed, writer_labels=assignments)
        y = frame["writer_id"].map(assignments).to_numpy(dtype=int)
        predictions, targets = [], []
        for fold in range(n_splits):
            test_ids = set(folds.loc[folds["fold"] == fold, "writer_id"])
            test_mask = frame["writer_id"].isin(test_ids).to_numpy()
            tr, te = np.flatnonzero(~test_mask & shared), np.flatnonzero(test_mask & shared)
            pipeline = _estimator("svm")
            pipeline.fit(X[tr], y[tr])
            part = frame.iloc[te][["writer_id"]].copy()
            part["label"], part["score"] = y[te], _scores(pipeline, "svm", X[te])
            grouped = part.groupby("writer_id").agg(label=("label", "first"), score=("score", "mean"))
            predictions.extend((grouped["score"] >= 0).astype(int).tolist())
            targets.extend(grouped["label"].tolist())
        return float(balanced_accuracy_score(targets, predictions))

    observed = statistic(base_labels)
    rng = np.random.default_rng(seed)
    null_scores, assignments_log = [], []
    for number in range(n_permutations):
        permuted = base_labels.copy()
        for indices in block_indices:
            permuted[indices] = rng.permutation(base_labels[indices])
        null_scores.append(statistic(permuted))
        assignments_log.append({"permutation": number,
                                "left_writers": "|".join(writers.loc[permuted == 1, "writer_id"]),
                                "writer_balanced_accuracy": null_scores[-1]})
    null = np.asarray(null_scores)
    exceed = int((null >= observed - 1e-12).sum())
    result = {"observed": observed, "p_value": (exceed + 1) / (n_permutations + 1),
              "null_scores": null, "parameters": {"n_permutations": n_permutations, "seed": seed,
                  "fold_seed": fold_seed, "n_splits": n_splits, "exceedances": exceed,
                  "label_encoding": LABELS, "writer_labels_shuffled_as_blocks": True,
                  "label_dependent_folds_regenerated": True,
                  "exchangeability_mode": exchangeability_mode, "exchangeability_block_counts": block_counts,
                  "distinct_permitted_assignments": permitted_assignments,
                  "pipeline_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "writer_order": writers["writer_id"].tolist(), "observed_writer_labels": base_labels.tolist(),
                  "scope": "descriptive development-overlap association" if frame["pilot_status"].eq("returning").any() else "prespecified primary writer-label association under exchangeability"}}
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError("Permutation output directory must be new.")
        pd.DataFrame(assignments_log).to_csv(output / "primary_permutation_null.csv", index=False)
        (output / "primary_permutation_summary.json").write_text(json.dumps({k: v for k, v in result.items() if k != "null_scores"}, indent=2), encoding="utf-8")
    return result
