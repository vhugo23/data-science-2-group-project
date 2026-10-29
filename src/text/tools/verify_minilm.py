"""Verify saved MiniLM artifacts without needing a local GPU or PyTorch install.

Uses cached frozen embeddings to reproduce held-out and OOF classifier scores.
"""
import hashlib
import json
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.text.baseline import metrics, sha256
from src.text.challenge import read_csv, require_locked


def main():
    root = Path("outputs/text/minilm")
    summary = json.loads((root / "minilm_summary.json").read_text())
    config = json.loads((root / "frozen_config.json").read_text())
    freeze = json.loads((root / "comparison_freeze.json").read_text())
    assert summary["encoder_frozen"] and summary["trainable_encoder_parameters"] == 0
    assert summary["embedding_cache_reuse_verified"]
    assert summary["challenge_scored"] is False and summary["test_evaluated"] is False
    assert sha256((root / "minilm_logreg.joblib").read_bytes()) == config["saved_classifier_sha256"] == freeze["minilm_classifier_sha256"]
    assert sha256((root / "frozen_config.json").read_bytes()) == freeze["minilm_config_sha256"]
    assert sha256(Path("src/text/minilm.py").read_text(encoding="utf-8")) == freeze["minilm_source_sha256"]
    for name, expected in freeze["encoder_files_sha256"].items():
        assert sha256((root / "encoder" / name).read_bytes()) == expected
    key = summary["embedding_cache"]["cache_key"]
    cache_path = root / "embedding_cache" / f"{key}.npz"
    assert sha256(cache_path.read_bytes()) == summary["embedding_cache"]["npz_sha256"]
    with np.load(cache_path, allow_pickle=False) as cache:
        ids, embeddings = cache["ids"].tolist(), cache["embeddings"].copy()
        lengths, chunks = cache["content_wordpieces"].copy(), cache["chunks"].copy()
    assert embeddings.shape == (1278, 384) and len(set(ids)) == 1278
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=2e-6)
    np.testing.assert_array_equal(chunks, np.maximum(1, np.ceil(lengths / 254).astype(int)))
    fixed = pd.read_csv("notebooks/text/split_manifest.csv").set_index("dialogue_id")
    data = fixed.loc[ids].assign(position=np.arange(len(ids)))
    assert data.included.all() and set(data.split) == {"train", "validation"}
    scores = pd.read_csv(root / "fusion_scores_train_validation.csv").set_index("dialogue_id")
    assert scores.index.is_unique and set(scores.index) == set(ids)
    assert scores.scam_score.between(0, 1).all()
    np.testing.assert_array_equal(scores.loc[ids, "label"], data.label)
    classifier = joblib.load(root / "minilm_logreg.joblib")["classifier"]
    val = data[data.split.eq("validation")]
    reproduced = classifier.predict_proba(embeddings[val.position])[:, 1]
    np.testing.assert_allclose(reproduced, scores.loc[val.index, "scam_score"], atol=1e-12, rtol=1e-12)
    assert metrics(val.label, reproduced, config["threshold"]) == config["validation"]
    folds = pd.read_csv("outputs/text/baseline/oof_folds.csv").set_index("dialogue_id")
    for fold in range(5):
        fit = folds[folds.fold.ne(fold)].index
        heldout = folds[folds.fold.eq(fold)].index
        estimator = clone(classifier).fit(embeddings[data.loc[fit, "position"]], data.loc[fit, "label"])
        expected = estimator.predict_proba(embeddings[data.loc[heldout, "position"]])[:, 1]
        np.testing.assert_allclose(expected, scores.loc[heldout, "scam_score"], atol=1e-12, rtol=1e-12)
        assert scores.loc[heldout, "fold"].eq(fold).all()
    prefixes = pd.read_csv(root / "validation_prefix_scores.csv")
    assert len(prefixes) == 192 * 2 * 4
    for (name, fraction), group in prefixes.groupby(["model", "turn_fraction"]):
        assert group.dialogue_id.is_unique and set(group.dialogue_id) == set(val.index)
        assert group.scam_score.between(0, 1).all()
    notebook = nbformat.read("notebooks/text/03_minilm_scam_intent.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata.scam_intent_execution.colab_runtime
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None and not any(o.output_type == "error" for o in cell.outputs)
    challenge = Path("notebooks/text/challenge_v1")
    assert len(read_csv(challenge / "review_packet/dialogues.csv")) == 48
    if not (challenge / "challenge_lock.json").exists():
        try:
            require_locked(challenge)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Unreviewed challenge gate did not block")
    provenance = json.loads(Path("outputs/text/input_provenance.json").read_text())["minilm_inputs.zip"]
    assert notebook.metadata.input_archive_sha256 == provenance["archive_sha256"]
    assert not any("challenge" in name.lower() for name in provenance["entries_sha256"])
    print("PASS: frozen encoder/cache, saved preprocessing hashes, fixed IDs/folds, validation and all OOF scores reproduced, prefix exports, completed Colab A100 notebook, unreviewed challenge remains gated.")


if __name__ == "__main__":
    main()
