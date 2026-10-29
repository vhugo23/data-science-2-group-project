"""Check the actual baseline handoff, saved model, and fixed split metadata."""
import json
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd

from src.text.baseline import sha256


def main():
    output = Path("outputs/text/baseline")
    fixed = pd.read_csv("notebooks/text/split_manifest.csv", keep_default_na=False)
    manifest = pd.read_csv(output / "split_manifest.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(fixed, manifest)
    assert manifest.dialogue_id.is_unique
    assert manifest.split.value_counts().to_dict() == {"train": 1088, "test": 320, "validation": 192}
    used = manifest[manifest.included]
    assert used.split.value_counts().to_dict() == {"train": 1086, "test": 320, "validation": 192}
    assert manifest.loc[manifest.source_split.eq("test"), "split"].eq("test").all()
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert set(used.loc[used.split.eq(left), "normalized_sha256"]).isdisjoint(
            used.loc[used.split.eq(right), "normalized_sha256"])
    scores = pd.read_csv(output / "fusion_scores_train_validation.csv")
    folds = pd.read_csv(output / "oof_folds.csv")
    assert scores.dialogue_id.is_unique and folds.dialogue_id.is_unique
    assert set(scores.dialogue_id) == set(used.loc[used.split.ne("test"), "dialogue_id"])
    assert scores.scam_score.between(0, 1).all()
    assert set(scores.loc[scores.split.eq("train"), "fold"]) == set(range(5))
    assert scores.loc[scores.split.eq("validation"), "fold"].eq(-1).all()
    joined = scores.merge(used, on="dialogue_id", suffixes=("_score", "_manifest"), validate="one_to_one")
    assert joined.label_score.eq(joined.label_manifest).all()
    assert joined.split_score.eq(joined.split_manifest).all()
    train_scores = scores[scores.split.eq("train")].merge(folds, on="dialogue_id", validate="one_to_one")
    assert len(train_scores) == 1086 and train_scores.fold_x.eq(train_scores.fold_y).all()
    config = json.loads((output / "frozen_config.json").read_text())
    assert sha256((output / "tfidf_logreg.joblib").read_bytes()) == config["saved_model_sha256"]
    assert config["test_evaluated"] is False
    bundle = joblib.load(output / "tfidf_logreg.joblib")
    # Real held-out round trip across platforms: reproduce saved Colab validation scores.
    raw = pd.read_csv("data/scam_dialogue/" + config["dataset_revision"] + "/scam-dialogue_train.csv", keep_default_na=False)
    validation = used[used.split.eq("validation")]
    restored = bundle["pipeline"].predict_proba(raw.iloc[validation.source_row].dialogue)[:, 1]
    expected = scores.set_index("dialogue_id").loc[validation.dialogue_id, "scam_score"].to_numpy()
    np.testing.assert_allclose(restored, expected, rtol=1e-12, atol=1e-12)
    notebook = nbformat.read("notebooks/text/01_scam_intent_baseline.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata.scam_intent_execution.colab_runtime
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert not any(o.output_type == "error" for o in cell.outputs)
    print("PASS: fixed IDs/splits, duplicate separation, OOF coverage/folds, scores/labels, saved-model hash and score round trip, completed Colab A100 notebook.")


if __name__ == "__main__":
    main()
