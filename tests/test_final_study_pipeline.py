"""Fast regression tests for the manifest-first final-study workflow.

The fixtures describe synthetic people and prompts. They do not fill missing
collection facts in the real 37-writer inventory or train on real photographs.
Run with /opt/anaconda3/bin/python3 -m pytest tests/test_final_study_pipeline.py.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import final_study_pipeline as study


@pytest.fixture
def manifest():
    records = []
    for wi in range(40):
        writer = f"person_{wi:02d}"
        for pi in range(8):
            common = pi < 6
            pair = f"S{pi // 2 + 1}" if common else "D"
            image = f"{writer}_paragraph_{pi}"
            records.append({
                "image_id": image,
                "source_path": f"raw/{image}.png",
                "archive_member": "",
                "sha256": hashlib.sha256(image.encode()).hexdigest(),
                "writer_id": writer,
                "label": int(wi >= 20),
                "writing_hand": "left" if wi >= 20 else "right",
                "label_verified": True,
                "condition": "shared" if common else "different",
                "paragraph_instance_id": f"{writer}_instance_{pi}",
                "prompt_id": f"common_{pi}" if common else f"{writer}_text_{pi}",
                "physical_page_id": f"{writer}_sheet_{pi // 2}",
                "pair_id": pair,
                "session_id": "interleaved_session_1",
                "capture_version": "canonical_v1",
                "is_canonical": True,
                "identity_verified": True,
                "prompt_verified": True,
                "quality_approved": True,
                "pilot_status": "new",
                "pilot_writer_id": "",
                "rotation_deg": 0,
            })
    return pd.DataFrame(records)


@pytest.fixture
def features(manifest):
    rng = np.random.default_rng(91)
    values = rng.normal(size=(len(manifest), 183))
    # The raw first feature identifies which rows reach a fit spy; this is a
    # synthetic test marker, never a predictor proposed for actual experiments.
    values[:, 0] = np.arange(len(manifest))
    values[:, 1:4] += manifest.label.to_numpy()[:, None] * 0.8
    return values


def test_complete_synthetic_manifest_passes_and_csv_round_trips(manifest, tmp_path):
    report = study.validate_manifest(manifest)
    assert report["valid"], report["errors"]
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)
    loaded = study.load_manifest(path)
    assert study.validate_manifest(loaded)["valid"]
    assert len(loaded) == 320
    assert loaded.writer_id.nunique() == 40


def test_csv_false_flags_do_not_become_truthy_strings(manifest, tmp_path):
    manifest.loc[0, "identity_verified"] = False
    path = tmp_path / "unverified.csv"
    manifest.to_csv(path, index=False)
    loaded = study.load_manifest(path)
    assert not study.validate_manifest(loaded)["valid"]


@pytest.mark.parametrize("field,value", [
    ("writer_id", ""),
    ("paragraph_instance_id", ""),
    ("prompt_id", ""),
    ("physical_page_id", ""),
    ("session_id", ""),
    ("capture_version", ""),
    ("pilot_status", ""),
    ("writing_hand", ""),
    ("writing_hand", "left"),
    ("label_verified", False),
    ("identity_verified", False),
    ("prompt_verified", False),
    ("quality_approved", False),
    ("rotation_deg", 45),
])
def test_unknown_or_unapproved_collection_facts_fail(manifest, field, value):
    manifest.loc[0, field] = value
    report = study.validate_manifest(manifest)
    assert not report["valid"], (field, report)
    assert report["errors"]


def test_provisional_37_writer_collection_is_not_40_writer_freeze(manifest):
    available = manifest[~manifest.writer_id.isin(["person_19", "person_38", "person_39"])]
    assert len(available) == 296
    assert available.writer_id.nunique() == 37
    assert not study.validate_manifest(available)["valid"]
    report = study.validate_manifest(available, expected_writers=37, expected_per_class=None)
    assert report["valid"], report["errors"]


@pytest.mark.parametrize("defect", [
    "inconsistent_label", "duplicate_image_id", "duplicate_hash_across_writers",
    "missing_canonical", "duplicate_canonical", "wrong_common_prompt",
    "wrong_page_pair",
    "returning_without_pilot_identity", "new_with_recorded_pilot_identity",
])
def test_manifest_rejects_identity_capture_and_prompt_contradictions(manifest, defect):
    if defect == "inconsistent_label":
        manifest.loc[0, "label"] = 1
    elif defect == "duplicate_image_id":
        manifest.loc[1, "image_id"] = manifest.loc[0, "image_id"]
    elif defect == "duplicate_hash_across_writers":
        manifest.loc[8, "sha256"] = manifest.loc[0, "sha256"]
    elif defect == "missing_canonical":
        manifest.loc[0, "is_canonical"] = False
    elif defect == "duplicate_canonical":
        duplicate = manifest.iloc[[0]].copy()
        duplicate["image_id"] = "recaptured_same_handwriting"
        duplicate["source_path"] = "raw/recaptured.png"
        duplicate["sha256"] = "f" * 64
        duplicate["capture_version"] = "v2"
        manifest = pd.concat([manifest, duplicate], ignore_index=True)
    elif defect == "wrong_common_prompt":
        manifest.loc[0, "prompt_id"] = "unverified_other_text"
    elif defect == "wrong_page_pair":
        manifest.loc[1, "physical_page_id"] = manifest.loc[2, "physical_page_id"]
    elif defect == "returning_without_pilot_identity":
        manifest.loc[manifest.writer_id == "person_00", "pilot_status"] = "returning"
    elif defect == "new_with_recorded_pilot_identity":
        manifest.loc[manifest.writer_id == "person_00", "pilot_writer_id"] = "pilot_person_04"
    report = study.validate_manifest(manifest)
    assert not report["valid"], (defect, report)
    assert report["errors"]


def test_additional_text_overlap_warns_without_invalidating_shared_training_transfer(manifest):
    manifest.loc[14, "prompt_id"] = manifest.loc[6, "prompt_id"]
    report = study.validate_manifest(manifest)
    assert report["valid"], report["errors"]
    assert report["analysis_eligibility"]["A"]
    assert report["analysis_eligibility"]["C"]
    assert report["warnings"]


def test_additional_common_prompt_overlap_preserves_primary_but_disables_transfer(manifest, features):
    manifest.loc[6, "prompt_id"] = "common_0"
    report = study.validate_manifest(manifest)
    assert report["valid"], report["errors"]
    assert report["analysis_eligibility"]["A"]
    assert report["analysis_eligibility"]["B"]
    assert not report["analysis_eligibility"]["C"]
    assert not report["analysis_eligibility"]["D"]
    assert report["warnings"]
    with pytest.raises(ValueError, match="(?i)eligibility"):
        study.run_classical_study(manifest, features, analyses=("C",), bootstrap_resamples=20)


def _png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_inventory_reads_only_explicit_archives_and_leaves_facts_unverified(tmp_path):
    for filename, group, color in [
        ("LeftHandedNew.zip", "LeftHandedNew", "white"),
        ("RightHandNew.zip", "Right-hand-new", "gray"),
    ]:
        raw = _png_bytes(color)
        with zipfile.ZipFile(tmp_path / filename, "w") as archive:
            archive.writestr(f"{group}/P1/capture.png", raw)
        extracted = tmp_path / group / "P1"
        extracted.mkdir(parents=True)
        (extracted / "capture.png").write_bytes(raw)
    # An unrelated pilot archive and extracted right representation must not
    # become extra canonical images through recursive input discovery.
    with zipfile.ZipFile(tmp_path / "RIGHT_HANDED.zip", "w") as archive:
        archive.writestr("RIGHT_HANDED/P1/pilot.png", _png_bytes("black"))
    inventory = study.make_inventory(tmp_path)
    assert len(inventory) == 2
    assert not study.validate_manifest(inventory)["valid"]
    for field in ["prompt_id", "physical_page_id", "session_id", "pilot_status", "writing_hand"]:
        assert inventory[field].fillna("").eq("").all()
    for flag in ["is_canonical", "identity_verified", "prompt_verified", "quality_approved", "label_verified"]:
        assert not inventory[flag].astype(bool).any()


def test_folds_partition_writers_and_ignore_paragraph_multiplicity(manifest):
    folds = study.make_writer_folds(manifest)
    assert len(folds) == 40
    assert folds.writer_id.is_unique
    assert set(folds.fold) == set(range(5))
    for _, test in folds.groupby("fold"):
        assert test.label.value_counts().to_dict() == {0: 4, 1: 4}
        assert len(test) == 8
    repeated_rows = pd.concat([manifest, manifest.iloc[:32]], ignore_index=True)
    again = study.make_writer_folds(repeated_rows)
    pd.testing.assert_frame_equal(
        folds.sort_values("writer_id").reset_index(drop=True),
        again.sort_values("writer_id").reset_index(drop=True),
    )


def test_fold_override_broadcasts_one_permuted_label_per_writer(manifest):
    original = manifest.groupby("writer_id").label.first()
    permuted = dict(zip(original.index, np.random.default_rng(6).permutation(original.to_numpy())))
    folds = study.make_writer_folds(manifest, writer_labels=permuted)
    assert folds.set_index("writer_id").label.to_dict() == permuted
    for _, test in folds.groupby("fold"):
        assert test.label.value_counts().to_dict() == {0: 4, 1: 4}


def test_verified_file_gate_checks_the_actual_bytes(manifest, tmp_path):
    for index, row in manifest.iterrows():
        path = tmp_path / row.source_path
        path.parent.mkdir(exist_ok=True, parents=True)
        raw = _png_bytes((index % 256, index // 256, 31))
        path.write_bytes(raw)
        manifest.loc[index, "sha256"] = hashlib.sha256(raw).hexdigest()
    report = study.validate_manifest(manifest, root=tmp_path, verify_files=True)
    assert report["valid"], report["errors"]
    changed = tmp_path / manifest.loc[0, "source_path"]
    changed.write_bytes(_png_bytes("red"))
    report = study.validate_manifest(manifest, root=tmp_path, verify_files=True)
    assert not report["valid"]
    assert report["errors"]


def test_feature_extraction_has_finite_declared_dimension(manifest, tmp_path):
    # A tall image with a horizontal mark exercises explicit rotation without
    # requiring camera files or assuming portrait means sideways handwriting.
    array = np.full((90, 45, 3), 250, dtype=np.uint8)
    array[20:25, 6:39] = 20
    array[55:60, 10:35] = 50
    path = tmp_path / "paragraph.png"
    Image.fromarray(array).save(path)
    rows = manifest.iloc[:2].copy()
    rows["source_path"] = "paragraph.png"
    rows["archive_member"] = ""
    rows["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows["rotation_deg"] = [0, 90]
    values = study.extract_features(rows, tmp_path)
    assert values.shape == (2, 183)
    assert np.isfinite(values).all()
    assert not np.allclose(values[0], values[1])


def test_background_control_statistics_never_use_masked_handwriting_pixels():
    gray = np.full((40, 50), 180.0)
    ink = np.zeros_like(gray, dtype=bool)
    ink[10:20, 15:25] = True
    gray[ink] = 0
    first = study.masked_background_features(gray, ink, dilation_iterations=3)
    changed = gray.copy()
    changed[ink] = 255
    second = study.masked_background_features(changed, ink, dilation_iterations=3)
    assert first["features"].shape == (7,)
    assert np.isfinite(first["features"]).all()
    np.testing.assert_allclose(first["features"], second["features"])
    assert not first["mask"][ink].any()
    assert first["valid_pixel_fraction"] == pytest.approx(first["mask"].mean())
    assert 0 < first["valid_pixel_fraction"] < 1


@pytest.fixture
def fitted_study(manifest, features, monkeypatch):
    from sklearn.pipeline import Pipeline

    fit_rows = []
    original = Pipeline.fit

    def record_fit(self, values, labels, *args, **kwargs):
        fit_rows.append(frozenset(np.asarray(values)[:, 0].astype(int)))
        return original(self, values, labels, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", record_fit)
    result = study.run_classical_study(
        manifest, features, bootstrap_resamples=20,
    )
    return result, fit_rows


def test_fits_use_only_declared_training_writers_and_condition(manifest, fitted_study):
    result, fit_rows = fitted_study
    folds = result["fold_assignments"]
    assert len(folds) == 40 and folds.writer_id.is_unique
    expected = []
    for fold in range(5):
        held_out = set(folds.loc[folds.fold == fold, "writer_id"])
        training = ~manifest.writer_id.isin(held_out)
        expected.append(frozenset(manifest.index[training & manifest.condition.eq("shared")]))
        expected.append(frozenset(manifest.index[training & manifest.condition.eq("different")]))
        for pair in ["S1", "S2", "S3"]:
            expected.append(frozenset(manifest.index[training & manifest.pair_id.eq(pair)]))
    assert len(fit_rows) == 25, "Transfer and pair scoring must reuse A rather than fit again"
    assert set(fit_rows) == set(expected)
    selection = result["model_selection"]
    assert len(selection) == 25
    for _, row in selection.iterrows():
        train = set(row.train_writer_ids.split("|"))
        test = set(row.test_writer_ids.split("|"))
        assert len(train) == 32 and len(test) == 8
        assert train.isdisjoint(test)


def test_transfer_reuses_fitted_model_and_preserves_writer_unit(fitted_study):
    result, _ = fitted_study
    writers = result["writer_predictions"]
    expected_analyses = {"A", "B", "C", "D_S1", "D_S2", "D_S3", "E_S1", "E_S2", "E_S3"}
    assert set(writers.analysis_id) == expected_analyses
    assert not writers.duplicated(["analysis_id", "model", "writer_id"]).any()
    for analysis, rows in writers.groupby("analysis_id"):
        assert len(rows) == 40
        assert rows.n_paragraphs.eq(6 if analysis == "A" else 2).all()
        assert rows.physical_pages.eq(3 if analysis == "A" else 1).all()
        assert rows.threshold.eq(0).all()
        np.testing.assert_array_equal(rows.prediction, (rows.score >= 0).astype(int))
    ids = writers.pivot(index="writer_id", columns="analysis_id", values="fitted_model_id")
    for analysis in ["C", "D_S1", "D_S2", "D_S3"]:
        assert ids[analysis].eq(ids["A"]).all()
    for analysis in ["B", "E_S1", "E_S2", "E_S3"]:
        assert ids[analysis].ne(ids["A"]).all()
    paragraphs = result["paragraph_predictions"]
    assert paragraphs.groupby(["analysis_id", "image_id"]).size().eq(1).all()
    assert paragraphs[paragraphs.analysis_id.eq("A")].shape[0] == 240
    assert paragraphs[paragraphs.analysis_id.eq("C")].shape[0] == 80
    assert np.isfinite(paragraphs.score).all()


def test_primary_permutation_requires_explicit_exchangeability_confirmation(manifest, features):
    with pytest.raises(ValueError, match="(?i)exchangeability"):
        study.run_primary_permutation(manifest, features, n_permutations=3)


def test_primary_permutation_rejects_returning_development_writer(manifest, features):
    mask = manifest.writer_id.eq("person_00")
    manifest.loc[mask, "pilot_status"] = "returning"
    manifest.loc[mask, "pilot_writer_id"] = "pilot_person_04"
    with pytest.raises(ValueError, match="(?i)(pilot|returning|development)"):
        study.run_primary_permutation(
            manifest, features, n_permutations=3, exchangeability_confirmed=True,
        )


def _paired_prediction_fixture():
    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    correct = {
        "C": [1, 0, 1, 0, 1, 1, 0],
        "D_S1": [1, 1, 0, 1, 0, 0, 1],
        "D_S2": [0, 1, 1, 1, 1, 0, 0],
        "D_S3": [1, 0, 0, 0, 1, 1, 1],
    }
    rows = []
    for analysis, correctness in correct.items():
        for wi, (label, ok) in enumerate(zip(labels, correctness)):
            prediction = int(label if ok else 1 - label)
            rows.append({
                "writer_id": f"paired_person_{wi}", "label": label,
                "analysis_id": analysis, "prediction": prediction,
                "model": "svm", "feature_set": "handcrafted183",
                "score": 1.0 if prediction else -1.0, "threshold": 0.0,
                "fold": wi % 2,
            })
    return pd.DataFrame(rows), labels, {k: np.array(v) for k, v in correct.items()}


def _independent_paired_interval(labels, correct, resamples=20, seed=2028):
    # Generate writer indices once per class, then compute all four condition
    # statistics using that same draw. The shared-pair triplet stays intact.
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == hand) for hand in [0, 1]]
    sampled = [indices[rng.integers(0, len(indices), size=(resamples, len(indices)))]
               for indices in class_indices]
    differences = []
    for draw in range(resamples):
        def ba(analysis):
            return np.mean([correct[analysis][indices[draw]].mean() for indices in sampled])
        differences.append(ba("C") - np.mean([ba(pair) for pair in ["D_S1", "D_S2", "D_S3"]]))
    return np.quantile(differences, [.025, .975])


def test_paired_bootstrap_resamples_the_same_writers_and_shared_pair_triplet():
    predictions, labels, correct = _paired_prediction_fixture()
    result = study.paired_comparisons(predictions, bootstrap_resamples=20, seed=2028)
    row = result.set_index("comparison").loc["transfer_vs_shared2"]
    ba = {key: np.mean([values[labels == hand].mean() for hand in [0, 1]])
          for key, values in correct.items()}
    assert row.n_writers == 7
    assert row.estimate == pytest.approx(ba["C"] - np.mean([ba[k] for k in ["D_S1", "D_S2", "D_S3"]]))
    np.testing.assert_allclose(
        [row.ci95_low, row.ci95_high], _independent_paired_interval(labels, correct), atol=1e-12,
    )
    assert "conditional" in row.scope.lower()


def test_paired_comparison_uses_only_the_complete_common_writer_cohort():
    predictions, labels, correct = _paired_prediction_fixture()
    predictions = predictions[~(predictions.writer_id.eq("paired_person_0") & predictions.analysis_id.eq("D_S2"))]
    result = study.paired_comparisons(predictions, bootstrap_resamples=20, seed=2028)
    row = result.iloc[0]
    keep = np.arange(1, len(labels))
    assert row.n_writers == 6
    expected = _independent_paired_interval(labels[keep], {k: v[keep] for k, v in correct.items()})
    np.testing.assert_allclose([row.ci95_low, row.ci95_high], expected, atol=1e-12)


def test_paired_comparison_rejects_conflicting_true_labels():
    predictions, _, _ = _paired_prediction_fixture()
    predictions.loc[predictions.writer_id.eq("paired_person_0") & predictions.analysis_id.eq("C"), "label"] = 1
    with pytest.raises((ValueError, AssertionError), match="(?i)label"):
        study.paired_comparisons(predictions, bootstrap_resamples=20)


def test_writer_summary_distinguishes_balanced_accuracy_and_counts():
    rows = pd.DataFrame({
        "writer_id": [f"w{i}" for i in range(5)], "label": [0, 0, 0, 1, 1],
        "prediction": [0] * 5, "score": [-1.0] * 5, "fold": [0, 1, 0, 1, 0],
        "model": "svm", "feature_set": "handcrafted183", "analysis_id": "A",
    })
    summary = study.summarize_writer_predictions(rows, bootstrap_resamples=20).iloc[0]
    assert summary.n_writers == 5
    assert summary.n_correct == 3
    assert summary.balanced_accuracy == pytest.approx(.5)
    assert summary.accuracy == pytest.approx(.6)
    assert (summary.tn, summary.fp, summary.fn, summary.tp) == (3, 0, 2, 0)


def test_margin_auc_is_computed_within_folds_before_averaging():
    rows = pd.DataFrame({
        "writer_id": ["r0", "l0", "r1", "l1"], "label": [0, 1, 0, 1],
        "prediction": [0, 0, 1, 1], "score": [-100.0, -99.0, 100.0, 101.0],
        "fold": [0, 0, 1, 1], "model": "svm", "feature_set": "handcrafted183", "analysis_id": "A",
    })
    summary = study.summarize_writer_predictions(rows, bootstrap_resamples=20).iloc[0]
    assert summary.fold_auc_n == 2
    assert summary.fold_auc_mean == pytest.approx(1.0)


def test_primary_null_broadcasts_whole_writer_labels_and_recreates_folds(manifest, features, monkeypatch):
    from sklearn.pipeline import Pipeline

    assignments = []
    fits = []
    original_folds = study.make_writer_folds
    original_fit = Pipeline.fit

    def record_folds(frame, n_splits=5, seed=2026, writer_labels=None):
        assert writer_labels is not None
        assignments.append(dict(writer_labels))
        return original_folds(frame, n_splits, seed, writer_labels=writer_labels)

    def record_fit(self, values, labels, *args, **kwargs):
        indices = np.asarray(values)[:, 0].astype(int)
        rows = manifest.iloc[indices][["writer_id", "condition"]].copy()
        rows["permuted_label"] = labels
        assert rows.condition.eq("shared").all()
        assert rows.groupby("writer_id").permuted_label.nunique().eq(1).all()
        assert rows.writer_id.nunique() == 32
        fits.append(rows.groupby("writer_id").permuted_label.first().to_dict())
        return original_fit(self, values, labels, *args, **kwargs)

    monkeypatch.setattr(study, "make_writer_folds", record_folds)
    monkeypatch.setattr(Pipeline, "fit", record_fit)
    result = study.run_primary_permutation(
        manifest, features, n_permutations=3, exchangeability_confirmed=True,
    )
    assert len(assignments) == 4
    assert len(fits) == 20
    actual = manifest.groupby("writer_id").label.first().to_dict()
    assert assignments[0] == actual
    assert any(assigned != actual for assigned in assignments[1:])
    for assignment_index, assignment in enumerate(assignments):
        assert len(assignment) == 40
        assert sum(assignment.values()) == 20
        for fitted in fits[assignment_index * 5:(assignment_index + 1) * 5]:
            assert fitted == {writer: assignment[writer] for writer in fitted}
    null = np.asarray(result["null_scores"])
    assert null.shape == (3,)
    assert result["p_value"] == pytest.approx((np.count_nonzero(null >= result["observed"] - 1e-12) + 1) / 4)


def test_restricted_permutation_preserves_each_declared_writer_block(manifest, features, monkeypatch):
    actual = manifest.groupby("writer_id").label.first().to_dict()
    blocks = {writer: "capture_a" if int(writer.rsplit("_", 1)[1]) % 20 < 10 else "capture_b"
              for writer in actual}
    recorded = []
    original = study.make_writer_folds

    def record_folds(frame, n_splits=5, seed=2026, writer_labels=None):
        recorded.append(dict(writer_labels))
        return original(frame, n_splits, seed, writer_labels=writer_labels)

    monkeypatch.setattr(study, "make_writer_folds", record_folds)
    result = study.run_primary_permutation(
        manifest, features, n_permutations=3, exchangeability_confirmed=True, writer_blocks=blocks,
    )
    assert len(recorded) == 4
    for assignment in recorded:
        for block in ["capture_a", "capture_b"]:
            writers = [writer for writer in actual if blocks[writer] == block]
            assert sum(assignment[writer] for writer in writers) == sum(actual[writer] for writer in writers) == 10
    assert "block" in str(result["parameters"]).lower()


def test_pure_capture_blocks_cannot_produce_a_label_null(manifest, features):
    blocks = {writer: f"hand_pure_session_{label}" for writer, label in manifest.groupby("writer_id").label.first().items()}
    with pytest.raises(ValueError, match="(?i)(block|variation|exchangeability)"):
        study.run_primary_permutation(
            manifest, features, n_permutations=3, exchangeability_confirmed=True, writer_blocks=blocks,
        )


def test_tuned_conditions_share_the_same_inner_writer_assignments(manifest, features, monkeypatch):
    class InexpensiveProbabilityEstimator:
        @property
        def steps(self):
            return [("inexpensive_probability_stub", self)]

        def get_params(self):
            return {"test_stub": True}

        def fit(self, values, labels):
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, values):
            p = 1 / (1 + np.exp(-np.asarray(values)[:, 1]))
            return np.column_stack([1 - p, p])

    calls = []
    original = study.make_writer_folds

    def record_folds(frame, n_splits=5, seed=2026, writer_labels=None):
        result = original(frame, n_splits, seed, writer_labels=writer_labels)
        if n_splits == 4:
            calls.append(result.set_index("writer_id").fold.to_dict())
        return result

    monkeypatch.setattr(study, "_estimator", lambda name, parameters=None: InexpensiveProbabilityEstimator())
    monkeypatch.setattr(study, "make_writer_folds", record_folds)
    result = study.run_classical_study(
        manifest, features, models=("logistic",), bootstrap_resamples=20,
    )
    assert len(calls) == 25
    outer = result["fold_assignments"]
    for fold in range(5):
        training = set(outer.loc[outer.fold.ne(fold), "writer_id"])
        applicable = [mapping for mapping in calls if set(mapping) == training]
        assert len(applicable) == 5
        assert all(mapping == applicable[0] for mapping in applicable)
        assert set(applicable[0].values()) == {0, 1, 2, 3}
        assert set(applicable[0]).isdisjoint(outer.loc[outer.fold.eq(fold), "writer_id"])


def test_mapping_template_refuses_to_overwrite_an_existing_file(tmp_path):
    for filename, group, color in [
        ("LeftHandedNew.zip", "LeftHandedNew", "white"),
        ("RightHandNew.zip", "Right-hand-new", "gray"),
    ]:
        with zipfile.ZipFile(tmp_path / filename, "w") as archive:
            archive.writestr(f"{group}/P1/capture.png", _png_bytes(color))
    output = tmp_path / "annotated_manifest.csv"
    output.write_text("existing annotation\n")
    with pytest.raises(FileExistsError):
        study.make_inventory(tmp_path, output_csv=output)
    assert output.read_text() == "existing annotation\n"


def test_d_only_still_includes_the_additional_pair_for_its_declared_comparison(manifest, features):
    result = study.run_classical_study(manifest, features, analyses=("D",), bootstrap_resamples=20)
    assert set(result["writer_predictions"].analysis_id) == {"C", "D_S1", "D_S2", "D_S3"}
    assert len(result["model_selection"]) == 5
    assert set(result["paired_comparisons"].comparison) == {"transfer_vs_shared2"}


def test_fitted_model_identity_changes_when_actual_training_assignment_changes(manifest, features):
    first = study.run_classical_study(manifest, features, analyses=("A",), seed=2026, bootstrap_resamples=20)
    second = study.run_classical_study(manifest, features, analyses=("A",), seed=2027, bootstrap_resamples=20)
    fits1 = first["model_selection"].set_index("fold")
    fits2 = second["model_selection"].set_index("fold")
    different_training = fits1.train_writer_ids.ne(fits2.train_writer_ids)
    assert different_training.any()
    assert fits1.loc[different_training, "fitted_model_id"].ne(fits2.loc[different_training, "fitted_model_id"]).all()




def test_paragraph_summaries_keep_writer_counts_and_descriptive_scope():
    paragraphs = pd.DataFrame({
        'model': ['svm'] * 4, 'feature_set': ['handcrafted183'] * 4,
        'analysis_id': ['A'] * 4, 'prompt_id': ['common_0', 'common_1'] * 2,
        'image_id': ['a0', 'a1', 'b0', 'b1'], 'writer_id': ['a', 'a', 'b', 'b'],
        'label': [0, 0, 1, 1], 'score': [-1., 1., 1., 2.], 'threshold': [0.] * 4,
    })
    summary = study.summarize_paragraph_predictions(paragraphs).iloc[0]
    assert summary.n_images == 4 and summary.n_writers == 2
    assert summary.balanced_accuracy == pytest.approx(.75)
    assert summary.accuracy == pytest.approx(.75)
    assert (summary.tn, summary.fp, summary.fn, summary.tp) == (1, 1, 0, 2)
    assert 'correlated images' in summary.scope
    prompts = study.summarize_paragraph_predictions(paragraphs, by_prompt=True)
    assert set(prompts.prompt_id) == {'common_0', 'common_1'}
    assert prompts.n_writers.eq(2).all()


def test_single_writer_prompt_does_not_manufacture_balanced_accuracy():
    row = pd.DataFrame({'model': ['svm'], 'feature_set': ['handcrafted183'],
        'analysis_id': ['B'], 'prompt_id': ['unique_text'], 'image_id': ['a0'],
        'writer_id': ['a'], 'label': [0], 'score': [-1.], 'threshold': [0.]})
    summary = study.summarize_paragraph_predictions(row, by_prompt=True).iloc[0]
    assert summary.accuracy == 1 and summary.n_writers == 1
    assert np.isnan(summary.balanced_accuracy) and np.isnan(summary.macro_f1)
    assert np.isnan(summary.recall_left)
    assert 'not estimable' in summary.status


def test_repeated_image_in_paragraph_summary_is_rejected():
    row = pd.DataFrame({'model': ['svm'] * 2, 'feature_set': ['handcrafted183'] * 2,
        'analysis_id': ['B'] * 2, 'prompt_id': ['unique_text'] * 2,
        'image_id': ['a0'] * 2, 'writer_id': ['a'] * 2,
        'label': [0] * 2, 'score': [-1.] * 2, 'threshold': [0.] * 2})
    with pytest.raises(AssertionError, match='repeated OOF'):
        study.summarize_paragraph_predictions(row)
