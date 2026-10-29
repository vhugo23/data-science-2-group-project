"""Verify the diagnostic handoff against primary references and exported scores."""
import json
from pathlib import Path
import hashlib

import nbformat
import numpy as np
import pandas as pd

from src.text.diagnostics import score_metrics


def main():
    output = Path("outputs/text/diagnostics")
    summary = json.loads((output / "diagnostic_summary.json").read_text())
    manifest = pd.read_csv("notebooks/text/split_manifest.csv")
    train_ids = set(manifest.loc[manifest.included & manifest.split.eq("train"), "dialogue_id"])
    validation = manifest[manifest.included & manifest.split.eq("validation")].set_index("dialogue_id")
    folds = pd.read_csv("outputs/text/baseline/oof_folds.csv")
    trace = json.loads((output / "fit_call_trace.json").read_text())
    assert len(trace) == 11
    for record in trace[:6]:
        assert set(record["fit_ids"]) == train_ids
        assert record["dialogue_only"] and record["target_alignment_verified"]
    for fold, record in enumerate(trace[6:]):
        assert record["heldout_fold"] == fold
        assert set(record["fit_ids"]) == set(folds.loc[folds.fold.ne(fold), "dialogue_id"])
        assert not set(record["fit_ids"]) & set(validation.index)
    decisions = pd.read_csv(output / "overlap_decisions.csv")
    removals = pd.read_csv(output / "removed_training_ids.csv")
    strict_ids = set(decisions.loc[decisions.strict_remove, "train_id"])
    broad_ids = set(decisions.train_id)
    assert strict_ids.issubset(broad_ids) and broad_ids.issubset(train_ids)
    for scenario, expected in [("reviewed_near_duplicates_removed", strict_ids), ("all_flagged_validation_counterparts_removed", broad_ids)]:
        assert set(removals.loc[removals.scenario.eq(scenario), "dialogue_id"]) == expected
    predictions = pd.read_csv(output / "overlap_sensitivity_predictions.csv")
    metrics = pd.read_csv(output / "overlap_sensitivity_metrics.csv").set_index("scenario")
    for scenario, group in predictions.groupby("scenario"):
        assert group.dialogue_id.is_unique and set(group.dialogue_id) == set(validation.index)
        np.testing.assert_array_equal(group.label, validation.loc[group.dialogue_id, "label"])
        assert group.scam_score.between(0, 1).all()
        actual = score_metrics(group.label, group.scam_score, float(metrics.loc[scenario, "threshold"]))
        for key in ["precision", "recall", "f1", "accuracy", "roc_auc"]:
            np.testing.assert_allclose(actual[key], metrics.loc[scenario, key], atol=1e-12)
    controls = pd.read_csv(output / "label_shuffle_metrics.csv").set_index("seed")
    control_predictions = pd.read_csv(output / "label_shuffle_predictions.csv")
    assert set(controls.index) == set(range(20))
    for seed, group in control_predictions.groupby("seed"):
        assert group.dialogue_id.is_unique and set(group.dialogue_id) == set(validation.index)
        np.testing.assert_array_equal(group.label, validation.loc[group.dialogue_id, "label"])
        actual = score_metrics(group.label, group.scam_score, float(controls.loc[seed, "threshold"]))
        for key in ["precision", "recall", "f1", "accuracy", "roc_auc"]:
            np.testing.assert_allclose(actual[key], controls.loc[seed, key], atol=1e-12)
    assert summary["test_evaluated"] is False
    assert summary["review_scope"]["original_reference_artifacts_unchanged"]
    # Durable checksums preserve input provenance after disposable transfer ZIPs are removed.
    provenance = json.loads(Path("outputs/text/input_provenance.json").read_text())["diagnostic_inputs.zip"]
    for name, expected in provenance["entries_sha256"].items():
        if name.startswith("reference/"):
            actual = Path("outputs/text/baseline", Path(name).name).read_bytes()
            assert hashlib.sha256(actual).hexdigest() == expected
    notebook = nbformat.read("notebooks/text/02_baseline_diagnostics.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata.scam_intent_execution.colab_runtime
    assert "A100" in notebook.metadata.scam_intent_execution.gpu
    assert notebook.metadata.input_archive_sha256 == provenance["archive_sha256"]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert not any(o.output_type == "error" for o in cell.outputs)
    print("PASS: recorded fit isolation, fixed validation IDs, review/removal alignment, recomputed control and sensitivity metrics, original artifacts preserved, completed Colab A100 diagnostic notebook.")


if __name__ == "__main__":
    main()
