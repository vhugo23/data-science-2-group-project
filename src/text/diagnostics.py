"""Leakage, shuffled-label controls, and validation-overlap sensitivity.

Never scores the official test set and never overwrites primary artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, roc_auc_score

from src.text.baseline import (
    DATASET, REVISION, SOURCE_SHA256, export_oof, metrics, runtime_info, sha256,
    train_baseline, write_json,
)

FIXED_MANIFEST_SHA256 = "1fe25e23bff220b8c26efc211e5f2cc40b18b4d42135af4a80555a49a96e7559"
PERMUTATION_SEEDS = list(range(20))


def load_development(data_dir: Path, manifest_path: Path) -> pd.DataFrame:
    """Read only the published training CSV; honor the immutable split manifest."""
    if sha256(manifest_path.read_bytes()) != FIXED_MANIFEST_SHA256:
        raise ValueError("Fixed manifest changed")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    metadata = manifest[manifest.source_split.eq("train")].copy()
    path = data_dir / REVISION / "scam-dialogue_train.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{path.name}"
        with urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    if sha256(path.read_bytes()) != SOURCE_SHA256[path.name]:
        raise ValueError("Training source checksum changed")
    raw = pd.read_csv(path, keep_default_na=False).assign(source_row=lambda x: np.arange(len(x)))
    data = raw.merge(metadata, on="source_row", suffixes=("_source", ""), validate="one_to_one")
    assert len(data) == 1280 and data.label.eq(data.label_source).all()
    assert data.dialogue.map(sha256).eq(data.text_sha256).all()
    return data.drop(columns="label_source")


def score_metrics(labels, scores, threshold: float) -> dict:
    return {**metrics(labels, scores, threshold),
            "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= threshold)),
            "roc_auc": float(roc_auc_score(labels, scores))}


def overlap_decisions(review: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Use prior text review, not validation model errors, to define removals."""
    included = data[data.included].set_index("dialogue_id")
    records = []
    for row in review.itertuples():
        if not row.included_a or not row.included_b or {row.split_a, row.split_b} != {"train", "validation"}:
            continue
        train_id, val_id = (row.id_a, row.id_b) if row.split_a == "train" else (row.id_b, row.id_a)
        assert included.loc[train_id, "split"] == "train"
        assert included.loc[val_id, "split"] == "validation"
        confirmed = row.assessment == "near_verbatim_variant_retained"
        records.append({
            "train_id": train_id, "validation_id": val_id, "similarity": row.similarity,
            "prior_assessment": row.assessment,
            "review_decision": "confirmed_near_verbatim_variant" if confirmed else "shared_template_or_uncertain_source_identity",
            "strict_remove": confirmed, "broad_remove": True,
            "reason": row.reason,
        })
    return pd.DataFrame(records)


def audit_implementation(data: pd.DataFrame, reference: Path, output: Path, bundle: dict) -> dict:
    train = data[data.included & data.split.eq("train")]
    validation = data[data.included & data.split.eq("validation")]
    manifest = pd.read_csv(reference / "split_manifest.csv", keep_default_na=False)
    assert manifest.dialogue_id.is_unique
    assert set(train.dialogue_id).isdisjoint(validation.dialogue_id)
    assert set(train.normalized_sha256).isdisjoint(validation.normalized_sha256)
    assert set(data.split) == {"train", "validation"}
    lookup = data[data.included].set_index("text_sha256")
    assert lookup.index.is_unique
    fit_log = []
    original_tfidf = TfidfVectorizer.fit_transform
    original_lr = LogisticRegression.fit

    def observe_tfidf(vectorizer, raw_documents, y=None):
        documents = list(raw_documents)
        assert all(isinstance(item, str) for item in documents)
        hashes = [sha256(item) for item in documents]
        ids = lookup.loc[hashes, "dialogue_id"].tolist()
        assert set(ids).issubset(set(train.dialogue_id))
        fit_log.append({"fit_ids": ids, "n": len(ids), "dialogue_only": True})
        return original_tfidf(vectorizer, documents, y)

    def observe_lr(classifier, X, y, *args, **kwargs):
        record = fit_log[-1]
        expected = train.set_index("dialogue_id").loc[record["fit_ids"], "label"].to_numpy()
        np.testing.assert_array_equal(np.asarray(y), expected)
        assert X.shape[0] == len(expected)
        record["target_alignment_verified"] = True
        return original_lr(classifier, X, y, *args, **kwargs)

    replay = output / "implementation_replay"
    replay.mkdir(parents=True, exist_ok=True)
    with patch.object(TfidfVectorizer, "fit_transform", observe_tfidf), patch.object(LogisticRegression, "fit", observe_lr):
        replay_model, config = train_baseline(data, replay)
        replay_scores = export_oof(data, replay_model, replay)
    assert len(fit_log) == 11  # six grid fits plus five OOF fits
    for row in fit_log[:6]:
        assert set(row["fit_ids"]) == set(train.dialogue_id)
    folds = pd.read_csv(reference / "oof_folds.csv")
    for fold, row in enumerate(fit_log[6:]):
        expected = set(folds.loc[folds.fold.ne(fold), "dialogue_id"])
        heldout = set(folds.loc[folds.fold.eq(fold), "dialogue_id"])
        assert set(row["fit_ids"]) == expected and set(row["fit_ids"]).isdisjoint(heldout)
        row["heldout_fold"] = fold
    reference_scores = pd.read_csv(reference / "fusion_scores_train_validation.csv")
    joined = reference_scores.merge(replay_scores, on="dialogue_id", suffixes=("_reference", "_replay"), validate="one_to_one")
    assert len(joined) == len(reference_scores) == 1278
    np.testing.assert_allclose(joined.scam_score_reference, joined.scam_score_replay, rtol=1e-12, atol=1e-12)
    for key in ["ngram_range", "C", "threshold", "fit_n", "validation"]:
        assert config[key] == bundle["config"][key]
    # Independently reconstruct vocabulary and smoothed IDF from training documents.
    vectorizer = bundle["pipeline"]["tfidf"]
    document_frequency = Counter()
    analyzer = vectorizer.build_analyzer()
    for text in train.dialogue:
        document_frequency.update(set(analyzer(text)))
    expected_terms = {term for term, count in document_frequency.items() if count >= 2}
    terms = vectorizer.get_feature_names_out()
    assert set(terms) == expected_terms
    expected_idf = np.array([np.log((1 + len(train)) / (1 + document_frequency[t])) + 1 for t in terms])
    np.testing.assert_allclose(vectorizer.idf_, expected_idf, rtol=1e-12, atol=1e-12)
    write_json(output / "fit_call_trace.json", fit_log)
    result = {
        "train_validation_ids_disjoint": True, "normalized_exact_text_disjoint": True,
        "dialogue_only_fit_inputs_verified": True, "target_alignment_verified": True,
        "grid_pipeline_fits_verified": 6, "oof_pipeline_fits_verified": 5,
        "oof_heldout_preprocessing_isolation_verified": True,
        "training_only_vocabulary_and_idf_verified": True, "vocabulary_size": len(terms),
        "primary_validation_and_oof_scores_reproduced": True,
        "test_text_loaded_by_diagnostics": False, "test_scored": False,
        "limits": "Reproducing the implementation does not resolve shared templates or topic-label confounding.",
    }
    write_json(output / "implementation_checks.json", result)
    return result


def run_diagnostics(data: pd.DataFrame, review: pd.DataFrame, reference: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    reference_hashes = {p.name: sha256(p.read_bytes()) for p in reference.iterdir() if p.is_file()}
    config = json.loads((reference / "frozen_config.json").read_text())
    assert sha256((reference / "tfidf_logreg.joblib").read_bytes()) == config["saved_model_sha256"]
    bundle = joblib.load(reference / "tfidf_logreg.joblib")
    pipeline, threshold = bundle["pipeline"], config["threshold"]
    train = data[data.included & data.split.eq("train")].copy()
    validation = data[data.included & data.split.eq("validation")].copy()
    checks = audit_implementation(data, reference, output, bundle)
    decisions = overlap_decisions(review, data)
    decisions.to_csv(output / "overlap_decisions.csv", index=False, lineterminator="\n")
    permutations, permutation_predictions = [], []
    for seed in PERMUTATION_SEEDS:
        shuffled = np.random.default_rng(seed).permutation(train.label.to_numpy())
        control = clone(pipeline).fit(train.dialogue, shuffled)
        scores = control.predict_proba(validation.dialogue)[:, 1]
        row = {"seed": seed, "threshold": threshold,
               "changed_label_fraction": float(np.mean(shuffled != train.label)),
               **score_metrics(validation.label, scores, threshold)}
        permutations.append(row)
        permutation_predictions.extend({"seed": seed, "dialogue_id": ident, "label": int(label), "scam_score": float(score)}
                                       for ident, label, score in zip(validation.dialogue_id, validation.label, scores))
    controls = pd.DataFrame(permutations).drop(columns="confusion_matrix")
    controls.to_csv(output / "label_shuffle_metrics.csv", index=False)
    pd.DataFrame(permutation_predictions).to_csv(output / "label_shuffle_predictions.csv", index=False)
    scenarios = [
        ("primary_reference", set()),
        ("reviewed_near_duplicates_removed", set(decisions.loc[decisions.strict_remove, "train_id"])),
        ("all_flagged_validation_counterparts_removed", set(decisions.train_id)),
    ]
    rows, predictions, type_results, removed_rows = [], [], [], []
    reference_scores = pipeline.predict_proba(validation.dialogue)[:, 1]
    for name, removed in scenarios:
        fit = train[~train.dialogue_id.isin(removed)]
        if name == "primary_reference":
            fitted = pipeline
        else:
            fitted = clone(pipeline).fit(fit.dialogue, fit.label)
            joblib.dump({"pipeline": fitted, "primary_frozen_config": config, "scenario": name}, output / f"{name}.joblib")
        scores = fitted.predict_proba(validation.dialogue)[:, 1]
        row = {"scenario": name, "train_n": len(fit), "train_benign": int(fit.label.eq(0).sum()),
               "train_scam": int(fit.label.eq(1).sum()), "removed_train_n": len(removed),
               "threshold": threshold, "changed_predictions_vs_primary": int(np.sum((scores >= threshold) != (reference_scores >= threshold))),
               "mean_absolute_score_change": float(np.mean(np.abs(scores - reference_scores))),
               **score_metrics(validation.label, scores, threshold)}
        rows.append(row)
        for kind in sorted(validation.type.unique()):
            mask = validation.type.eq(kind)
            per_type = metrics(validation.loc[mask, "label"], scores[mask], threshold)
            if not per_type["scam_support"]:
                per_type.update(precision=None, recall=None, f1=None)
            type_results.append({"scenario": name, "type": kind, **per_type})
        predictions.extend({"scenario": name, "dialogue_id": ident, "label": int(label), "scam_score": float(score)}
                           for ident, label, score in zip(validation.dialogue_id, validation.label, scores))
        removed_rows.extend({"scenario": name, "dialogue_id": ident} for ident in sorted(removed))
    pd.DataFrame(rows).drop(columns="confusion_matrix").to_csv(output / "overlap_sensitivity_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "overlap_sensitivity_predictions.csv", index=False)
    pd.DataFrame(type_results).drop(columns="confusion_matrix").to_csv(output / "overlap_sensitivity_by_type.csv", index=False)
    pd.DataFrame(removed_rows).to_csv(output / "removed_training_ids.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, column in zip(axes, ["balanced_accuracy", "roc_auc"]):
        ax.scatter(controls.seed, controls[column], label="Shuffled training labels")
        ax.axhline(0.5, color="gray", linestyle="--", label="Chance expectation")
        ax.axhline(1.0, color="black", linestyle=":", label="Primary validation")
        ax.set(xlabel="Permutation seed", ylabel=column.replace("_", " ").upper(), ylim=(-0.03, 1.03))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "label_shuffle_controls.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, row, title in zip(axes, rows, ["Primary reference", "Reviewed near duplicates removed", "All flagged counterparts removed"]):
        ConfusionMatrixDisplay(np.asarray(row["confusion_matrix"]), display_labels=["Benign", "Scam"]).plot(ax=ax, colorbar=False)
        ax.set_title(title + f"\nfit n={row['train_n']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "overlap_sensitivity_confusion_matrices.png", dpi=160)
    plt.close(fig)
    for filename, digest in reference_hashes.items():
        assert sha256((reference / filename).read_bytes()) == digest
    summary = {
        "protocol": "Fixed primary unigram/C/threshold settings; no retuning; same 192 validation dialogues; official test not read or scored.",
        "implementation_checks": checks,
        "label_shuffle": {
            "seeds": PERMUTATION_SEEDS,
            "summary": controls[["accuracy", "balanced_accuracy", "roc_auc", "precision", "recall", "f1"]].agg(["mean", "std", "min", "max"]).to_dict(),
            "interpretation": "A negative control for implementation leakage; F1 has no universal chance value. Topic-correlated validation dialogues can produce substantial between-permutation variation.",
        },
        "overlap_sensitivity": rows,
        "review_scope": {"train_validation_flagged_pairs": len(decisions),
                         "confirmed_near_verbatim_pairs": int(decisions.strict_remove.sum()),
                         "validation_membership_unchanged": True,
                         "original_reference_artifacts_unchanged": True},
        "test_evaluated": False,
        "limits": "Even perfect sensitivity performance does not establish same-topic intent discrimination or real-world generalization. No independently reviewed challenge set has been evaluated.",
    }
    write_json(output / "diagnostic_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scam_dialogue"))
    parser.add_argument("--reference-dir", type=Path, default=Path("outputs/text/baseline"))
    parser.add_argument("--manifest", type=Path, default=Path("notebooks/text/split_manifest.csv"))
    parser.add_argument("--review", type=Path, default=Path("notebooks/text/audit/similarity_review.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/text/diagnostics"))
    parser.add_argument("--require-a100", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "runtime.json", runtime_info(args.require_a100))
    data = load_development(args.data_dir, args.manifest)
    summary = run_diagnostics(data, pd.read_csv(args.review), args.reference_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
