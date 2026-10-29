"""Evaluate fixed classifiers only after independent challenge review and locking.

Official test evaluation additionally requires a completed challenge-evaluation
receipt matching the locked data and unchanged model freeze. No fitting occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.text.baseline import DATASET, REVISION, SOURCE_SHA256, metrics, prefix, sha256, write_json
from src.text.challenge import require_locked


def verify_freeze(baseline: Path, minilm: Path) -> dict:
    freeze = json.loads((minilm / "comparison_freeze.json").read_text())
    checks = [(baseline / "tfidf_logreg.joblib", freeze["tfidf_model_sha256"]),
              (baseline / "frozen_config.json", freeze["tfidf_config_sha256"]),
              (minilm / "minilm_logreg.joblib", freeze["minilm_classifier_sha256"]),
              (minilm / "frozen_config.json", freeze["minilm_config_sha256"])]
    checks.extend((minilm / "encoder" / name, digest) for name, digest in freeze["encoder_files_sha256"].items())
    for path, expected in checks:
        if sha256(path.read_bytes()) != expected:
            raise ValueError(f"A validation-selected model or preprocessing artifact changed: {path}")
    source = Path(__file__).with_name("minilm.py").read_text(encoding="utf-8")
    if sha256(source) != freeze["minilm_source_sha256"]:
        raise ValueError("MiniLM preprocessing implementation changed after freeze")
    return freeze


def evaluate(frame: pd.DataFrame, split: str, baseline: Path, minilm: Path, output: Path, freeze: dict):
    # Imported only after callers have checked the review/lock gates.
    from src.text.minilm import FrozenMiniLM
    encoder = FrozenMiniLM(minilm / "encoder")
    assert encoder.fingerprint() == freeze["minilm_encoder_fingerprint"]
    tfidf = joblib.load(baseline / "tfidf_logreg.joblib")["pipeline"]
    classifier = joblib.load(minilm / "minilm_logreg.joblib")["classifier"]
    output.mkdir(parents=True, exist_ok=True)
    rows, summary, by_type, errors = [], [], [], []
    for fraction in [0.25, 0.5, 0.75, 1.0]:
        texts = [text if fraction == 1 else prefix(text, fraction) for text in frame.dialogue]
        embeddings, lengths, _ = encoder.encode(frame.dialogue_id.tolist(), texts, output / "embedding_cache")
        if fraction == 1:
            lengths.to_csv(output / "wordpiece_lengths.csv", index=False)
        for name, scores, threshold in [
            ("tfidf_logreg", tfidf.predict_proba(texts)[:, 1], freeze["tfidf_threshold"]),
            ("minilm_logreg", classifier.predict_proba(embeddings)[:, 1], freeze["minilm_threshold"]),
        ]:
            summary.append({"model": name, "split": split, "turn_fraction": fraction,
                            "threshold": threshold, **metrics(frame.label, scores, threshold)})
            rows.extend({"dialogue_id": ident, "split": split, "label": int(label), "model": name,
                         "scam_score": float(score), "fold": -1, "turn_fraction": fraction}
                        for ident, label, score in zip(frame.dialogue_id, frame.label, scores))
            if fraction == 1:
                for kind in sorted(frame.type.unique()):
                    mask = frame.type.eq(kind)
                    result = metrics(frame.loc[mask, "label"], scores[mask], threshold)
                    if not result["scam_support"]:
                        result.update(precision=None, recall=None, f1=None)
                    by_type.append({"model": name, "type": kind, **result})
                for row, score in zip(frame.itertuples(), scores):
                    prediction = int(score >= threshold)
                    if prediction != row.label:
                        errors.append({"dialogue_id": row.dialogue_id, "model": name, "type": row.type,
                                       "label": row.label, "scam_score": float(score),
                                       "error_kind": "false_positive" if prediction else "false_negative", "dialogue": row.dialogue})
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "prefix_scores.csv", index=False)
    scores[scores.turn_fraction.eq(1)].drop(columns="turn_fraction").to_csv(output / "full_scores.csv", index=False)
    table = pd.DataFrame(summary).drop(columns="confusion_matrix")
    table.to_csv(output / "prefix_metrics.csv", index=False)
    table[table.turn_fraction.eq(1)].to_csv(output / "full_metrics.csv", index=False)
    pd.DataFrame(by_type).drop(columns="confusion_matrix").to_csv(output / "by_type.csv", index=False)
    pd.DataFrame(errors, columns=["dialogue_id", "model", "type", "label", "scam_score", "error_kind", "dialogue"]).to_csv(output / "errors.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name in zip(axes, ["tfidf_logreg", "minilm_logreg"]):
        result = next(r for r in summary if r["model"] == name and r["turn_fraction"] == 1)
        ConfusionMatrixDisplay(np.asarray(result["confusion_matrix"]), display_labels=["Benign", "Scam"]).plot(ax=ax, colorbar=False)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=160)
    plt.close(fig)
    assert encoder.fingerprint() == freeze["minilm_encoder_fingerprint"]
    return {"split": split, "dialogues": len(frame), "metrics": summary,
            "models_unchanged": True, "no_fitting_or_threshold_selection": True,
            "constructed_diagnostic": split == "challenge"}


def score_challenge(challenge: Path, baseline: Path, minilm: Path, output: Path):
    # Check lock before loading classifiers, embeddings, or challenge model inputs.
    rows = require_locked(challenge)
    if (output / "challenge_receipt.json").exists():
        raise FileExistsError("Challenge was already scored; preserve its evaluation receipt")
    freeze = verify_freeze(baseline, minilm)
    data = pd.DataFrame(rows).rename(columns={"topic": "type"})
    data["label"] = data.label.astype(int)
    result = evaluate(data, "challenge", baseline, minilm, output, freeze)
    result.update(challenge_lock_sha256=sha256((challenge / "challenge_lock.json").read_bytes()),
                  comparison_freeze_sha256=sha256((minilm / "comparison_freeze.json").read_bytes()),
                  full_scores_sha256=sha256((output / "full_scores.csv").read_bytes()),
                  note="Constructed paired diagnostic; no tuning against these results; 24 paired scenarios, not 48 independent calls")
    write_json(output / "challenge_receipt.json", result)
    return result


def official_test(challenge: Path, challenge_results: Path, baseline: Path, minilm: Path,
                  manifest: Path, data_dir: Path, output: Path):
    require_locked(challenge)
    receipt_path = challenge_results / "challenge_receipt.json"
    if not receipt_path.exists():
        raise RuntimeError("Official test blocked: complete the locked challenge evaluation first")
    receipt = json.loads(receipt_path.read_text())
    assert receipt["challenge_lock_sha256"] == sha256((challenge / "challenge_lock.json").read_bytes())
    assert receipt["comparison_freeze_sha256"] == sha256((minilm / "comparison_freeze.json").read_bytes())
    assert receipt["full_scores_sha256"] == sha256((challenge_results / "full_scores.csv").read_bytes())
    if (output / "test_receipt.json").exists():
        raise FileExistsError("Official test already evaluated; preserve the frozen comparison")
    freeze = verify_freeze(baseline, minilm)
    # No official test source is read before all review and evaluation gates pass.
    name = "scam-dialogue_test.csv"
    path = data_dir / REVISION / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}", timeout=120) as response:
            path.write_bytes(response.read())
    assert sha256(path.read_bytes()) == SOURCE_SHA256[name]
    raw = pd.read_csv(path, keep_default_na=False).assign(source_row=lambda x: np.arange(len(x)))
    fixed = pd.read_csv(manifest)
    meta = fixed[fixed.source_split.eq("test")]
    assert len(meta) == 320 and meta.included.all() and meta.split.eq("test").all()
    data = raw.merge(meta, on="source_row", suffixes=("_source", ""), validate="one_to_one")
    assert data.label.eq(data.label_source).all() and data.dialogue.map(sha256).eq(data.text_sha256).all()
    result = evaluate(data, "test", baseline, minilm, output, freeze)
    result.update(comparison_freeze_sha256=sha256((minilm / "comparison_freeze.json").read_bytes()),
                  challenge_receipt_sha256=sha256(receipt_path.read_bytes()))
    test_scores = pd.read_csv(output / "full_scores.csv")
    all_scores = pd.concat([pd.read_csv(baseline / "fusion_scores_train_validation.csv"),
                            pd.read_csv(minilm / "fusion_scores_train_validation.csv"), test_scores], ignore_index=True)
    assert not all_scores.duplicated(["dialogue_id", "model"]).any()
    all_scores.to_csv(output / "fusion_scores_all_splits.csv", index=False)
    write_json(output / "test_receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["challenge", "test"])
    parser.add_argument("--challenge-dir", type=Path, default=Path("notebooks/text/challenge_v1"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/text/baseline"))
    parser.add_argument("--minilm-dir", type=Path, default=Path("outputs/text/minilm"))
    parser.add_argument("--challenge-results", type=Path, default=Path("outputs/text/challenge_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("notebooks/text/split_manifest.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/scam_dialogue"))
    parser.add_argument("--test-output", type=Path, default=Path("outputs/text/official_comparison"))
    args = parser.parse_args()
    if args.action == "challenge":
        result = score_challenge(args.challenge_dir, args.baseline_dir, args.minilm_dir, args.challenge_results)
    else:
        result = official_test(args.challenge_dir, args.challenge_results, args.baseline_dir, args.minilm_dir,
                               args.manifest, args.data_dir, args.test_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
