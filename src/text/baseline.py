"""Reproducible scam-text audit and train-only TF-IDF baseline.

Run from the repository root: python -m src.text.baseline
Only validation metrics are computed at this milestone; test remains sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from urllib.request import urlopen

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline

REVISION = "321b961b5ae353e19ed479b960658dcd223d5e06"
DATASET = "BothBosu/scam-dialogue"
SOURCE_SHA256 = {
    "scam-dialogue_train.csv": "cda5d42d15fd341208a25466d9ea279de905a640bd153f67b5c29743520f96f8",
    "scam-dialogue_test.csv": "17a9e52e90d893541d76dbb4588cb75b1a66ef2474ef2619d56feff5b8d5aec0",
}
SEED = 42
NEAR_THRESHOLD = 0.90
TURN = re.compile(r"\b(?:caller|receiver)\s*:", re.IGNORECASE)


def sha256(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def turns(text: str) -> list[str]:
    starts = [m.start() for m in TURN.finditer(text)]
    if not starts:
        return [text]
    starts[0] = 0  # Preserve any preamble.
    return [text[a:b].strip() for a, b in zip(starts, starts[1:] + [len(text)])]


def prefix(text: str, fraction: float) -> str:
    items = turns(text)
    return " ".join(items[:max(1, int(np.ceil(len(items) * fraction)))])


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def runtime_info(require_a100: bool = False) -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        gpu = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "unavailable"
    # This detects the remote runtime, including a VS Code Colab kernel.
    import os
    colab = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"))
    info = {
        "python": sys.version, "platform": platform.platform(), "gpu": gpu,
        "colab_runtime": colab,
        "classifier_compute": "CPU (scikit-learn TF-IDF/logistic regression)",
        "versions": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib"]},
    }
    if require_a100 and not (colab and "A100" in gpu):
        raise RuntimeError(f"A Colab A100 runtime is required; observed: {info}")
    return info


def load_data(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    data_dir.mkdir(parents=True, exist_ok=True)
    frames, files = [], {}
    for source, count in [("train", 1280), ("test", 320)]:
        name = f"scam-dialogue_{source}.csv"
        # Cache names are revision-scoped so a different snapshot cannot be reused silently.
        path = data_dir / REVISION / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}"
            with urlopen(url, timeout=120) as response:
                payload = response.read()
            path.write_bytes(payload)
        files[name] = sha256(path.read_bytes())
        if files[name] != SOURCE_SHA256[name]:
            raise ValueError(f"Dataset checksum mismatch: {path}; investigate cache/source before proceeding")
        frame = pd.read_csv(path, keep_default_na=False)
        if set(frame.columns) != {"dialogue", "type", "label"} or len(frame) != count:
            raise ValueError(f"Unexpected schema/count in {name}: {frame.shape}, {frame.columns}")
        if not frame.label.isin([0, 1]).all():
            raise ValueError("Labels must be 0/1")
        frame["source_split"] = source
        frame["source_row"] = np.arange(len(frame))  # zero-based CSV data-row ordinal
        frame["text_sha256"] = frame.dialogue.map(sha256)
        frame["normalized_sha256"] = frame.dialogue.map(lambda s: sha256(normalized(s)))
        frame["dialogue_id"] = [f"scam-{source}-{i:04d}-{digest[:16]}"
                                for i, digest in zip(frame.source_row, frame.text_sha256)]
        frame["split"] = source
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    train_indices = data.index[data.source_split.eq("train")]
    _, validation = train_test_split(
        train_indices, test_size=192, random_state=SEED,
        stratify=data.loc[train_indices, "label"],
    )
    data.loc[validation, "split"] = "validation"
    data["included"] = True
    data["exclusion_reason"] = ""
    empty = data.dialogue.str.strip().eq("")
    if empty[data.split.eq("test")].any():
        raise ValueError("Missing test text: preserve test and investigate before fitting")
    data.loc[empty, ["included", "exclusion_reason"]] = [False, "missing_text"]
    # Preserve test, then validation; remove exact/whitespace/case duplicates from lower-priority pools.
    # Similarity matches alone NEVER cause exclusions.
    priority = {"test": 0, "validation": 1, "train": 2}
    ordered = data.assign(priority=data.split.map(priority)).sort_values(["priority", "source_row"])
    for _, group in ordered.loc[~empty.reindex(ordered.index)].groupby("normalized_sha256", sort=False):
        if len(group) < 2:
            continue
        if group.label.nunique() != 1:
            raise ValueError("Conflicting labels in duplicate group; manual review required")
        keep = group.iloc[0]
        for index, row in group.iloc[1:].iterrows():
            if row.split != "test":
                data.loc[index, ["included", "exclusion_reason"]] = [
                    False, f"normalized_duplicate_of:{keep.dialogue_id}"]
    if not data.dialogue_id.is_unique:
        raise ValueError("ID collision")
    manifest_cols = ["dialogue_id", "source_split", "source_row", "text_sha256",
                     "normalized_sha256", "split", "label", "included", "exclusion_reason"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_text = data[manifest_cols].to_csv(index=False, lineterminator="\n")
    (output_dir / "split_manifest.csv").write_text(manifest_text, encoding="utf-8", newline="")
    write_json(output_dir / "dataset_provenance.json", {
        "dataset": DATASET, "revision": REVISION, "file_sha256": files,
        "seed": SEED, "validation_size": 192, "stratification": "label",
        "id_rule": "scam-{published_split}-{zero_based_row:04d}-{raw_text_sha256[:16]}",
    })
    return data


def audit(data: pd.DataFrame, output_dir: Path) -> dict:
    lengths = data[["dialogue_id", "split", "label", "type", "included"]].copy()
    lengths["characters"] = data.dialogue.str.len()
    lengths["words"] = data.dialogue.str.split().map(len)
    lengths["turns"] = data.dialogue.map(lambda s: len(turns(s)))
    lengths.to_csv(output_dir / "dialogue_lengths.csv", index=False)
    balance = data.groupby(["split", "included", "label", "type"]).size().rename("count").reset_index()
    balance.to_csv(output_dir / "class_type_counts.csv", index=False)
    duplicate_rows = data[data.duplicated("normalized_sha256", keep=False)]
    duplicate_rows.drop(columns="dialogue").to_csv(output_dir / "exact_duplicate_rows.csv", index=False)
    # Stateless character hashing is an audit instrument only, never a model input.
    hasher = HashingVectorizer(analyzer="char", ngram_range=(5, 5), n_features=2**18,
                               alternate_sign=False, lowercase=True, norm="l2")
    vectors = hasher.transform(data.dialogue.map(normalized))
    similarity = (vectors @ vectors.T).toarray()
    upper = np.triu(np.ones(similarity.shape, dtype=bool), 1)
    cross = data.split.to_numpy()[:, None] != data.split.to_numpy()[None, :]
    indices = np.argwhere(upper & cross & (similarity >= NEAR_THRESHOLD))
    records = []
    for i, j in indices:
        a, b = data.iloc[i], data.iloc[j]
        records.append({"id_a": a.dialogue_id, "id_b": b.dialogue_id,
                        "split_a": a.split, "split_b": b.split,
                        "similarity": float(similarity[i, j]),
                        "normalized_exact": a.normalized_sha256 == b.normalized_sha256,
                        "included_a": bool(a.included), "included_b": bool(b.included)})
    columns = ["id_a", "id_b", "split_a", "split_b", "similarity", "normalized_exact", "included_a", "included_b"]
    pairs = pd.DataFrame(records, columns=columns).sort_values("similarity", ascending=False)
    pairs.to_csv(output_dir / "near_duplicate_candidates.csv", index=False)
    # Keep full matching text locally for actual human inspection, including the most similar
    # pair for each split boundary even if no pair clears the candidate threshold.
    review = pairs.to_dict("records")
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        a = data.index[data.split.eq(left)].to_numpy()
        b = data.index[data.split.eq(right)].to_numpy()
        local_i, local_j = np.unravel_index(np.argmax(similarity[np.ix_(a, b)]), (len(a), len(b)))
        i, j = a[local_i], b[local_j]
        review.append({"id_a": data.iloc[i].dialogue_id, "id_b": data.iloc[j].dialogue_id,
                       "similarity": float(similarity[i, j]), "boundary_maximum": f"{left}/{right}"})
    lookup = data.set_index("dialogue_id")
    for row in review:
        row["dialogue_a"] = lookup.loc[row["id_a"], "dialogue"]
        row["dialogue_b"] = lookup.loc[row["id_b"], "dialogue"]
    write_json(output_dir / "similarity_review_text.json", review)
    summary = {
        "raw_rows": len(data), "missing_text": int(data.dialogue.str.strip().eq("").sum()),
        "planned_counts": data.split.value_counts().to_dict(),
        "included_counts": data[data.included].split.value_counts().to_dict(),
        "excluded_count": int((~data.included).sum()),
        "raw_exact_duplicate_groups": int(data.groupby("text_sha256").size().gt(1).sum()),
        "normalized_duplicate_groups": int(data.groupby("normalized_sha256").size().gt(1).sum()),
        "cross_split_candidates": len(pairs), "candidate_threshold": NEAR_THRESHOLD,
        "audit_similarity": "cosine of stateless 2^18-bin character 5-gram hashes; collisions possible",
        "length_summary": lengths[["characters", "words", "turns"]].describe().to_dict(),
        "exclusion_policy": "Missing text and normalized exact duplicates only; test always retained. No similarity-only exclusions.",
        "wordpiece_lengths": "Deferred to the MiniLM milestone; word counts are not wordpiece counts.",
    }
    write_json(output_dir / "audit_summary.json", summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for split, group in lengths.groupby("split"):
        axes[0].hist(group.words, bins=25, histtype="step", label=split)
        axes[1].hist(group.turns, bins=20, histtype="step", label=split)
    axes[0].set(xlabel="Whitespace-separated words", ylabel="Dialogues")
    axes[1].set(xlabel="Speaker turns", ylabel="Dialogues")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "dialogue_lengths.png", dpi=160)
    plt.close(fig)
    return summary


def metrics(labels, scores, threshold: float) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    predictions = scores >= threshold
    return {"n": len(labels), "scam_support": int(labels.sum()),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "accuracy": float(np.mean(labels == predictions)),
            "false_positive_rate": float(np.mean(predictions[labels == 0])) if np.any(labels == 0) else None,
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}


def train_baseline(data: pd.DataFrame, output_dir: Path) -> tuple[Pipeline, dict]:
    train = data[data.included & data.split.eq("train")].copy()
    validation = data[data.included & data.split.eq("validation")].copy()
    candidates, fitted = [], []
    for ngram in [(1, 1), (1, 2)]:
        for c in [0.1, 1.0, 10.0]:
            model = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=ngram, min_df=2, sublinear_tf=True)),
                ("classifier", LogisticRegression(C=c, max_iter=2000, solver="liblinear", random_state=SEED)),
            ])
            # dialogue is the ONLY feature; no type, ID, split, or target enters the pipeline.
            model.fit(train.dialogue, train.label)
            score = model.predict_proba(validation.dialogue)[:, 1]
            result = {"ngram_max": ngram[1], "C": c, **metrics(validation.label, score, 0.5)}
            candidates.append(result)
            fitted.append((model, score))
    table = pd.DataFrame(candidates).drop(columns="confusion_matrix")
    table.to_csv(output_dir / "validation_search.csv", index=False)
    # Deterministic ties: simpler ngrams, then stronger regularization (loop order).
    best = max(range(len(candidates)), key=lambda i: candidates[i]["f1"])
    model, scores = fitted[best]
    thresholds = sorted(set([0.5, 0.0, 1.0, *map(float, scores)]))
    threshold = max(thresholds, key=lambda t: (
        f1_score(validation.label, scores >= t, zero_division=0), -abs(t - 0.5), t))
    result = {"model": "tfidf_logreg", "seed": SEED,
              "ngram_range": [1, candidates[best]["ngram_max"]], "C": candidates[best]["C"],
              "threshold": threshold, "threshold_rule": "score >= threshold",
              "selection": "validation F1 at 0.5, ties simplest ngrams then smallest C; threshold then maximizes validation F1 with ties closest to 0.5",
              "fit_n": len(train), "validation_at_0_5": metrics(validation.label, scores, 0.5),
              "validation": metrics(validation.label, scores, threshold),
              "test_evaluated": False, "dataset_revision": REVISION}
    model_path = output_dir / "tfidf_logreg.joblib"
    joblib.dump({"pipeline": model, "config": result}, model_path)
    result["saved_model_sha256"] = sha256(model_path.read_bytes())
    write_json(output_dir / "frozen_config.json", result)
    validation = validation.assign(scam_score=scores, prediction=(scores >= threshold).astype(int))
    validation.drop(columns="dialogue").to_csv(output_dir / "validation_predictions.csv", index=False)
    by_type = []
    for kind, group in validation.groupby("type"):
        # Types are single-class; precision/F1 are undefined for benign-only groups.
        row = {"type": kind, **metrics(group.label, group.scam_score, threshold)}
        if row["scam_support"] == 0:
            row.update(precision=None, recall=None, f1=None)
        by_type.append(row)
    pd.DataFrame(by_type).drop(columns="confusion_matrix").to_csv(output_dir / "validation_by_type.csv", index=False)
    errors = validation[validation.label.ne(validation.prediction)].copy()
    errors["error_kind"] = np.where(errors.label.eq(0), "false_positive", "false_negative")
    errors.to_csv(output_dir / "validation_errors.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(validation.label, validation.prediction,
        labels=[0, 1], display_labels=["Benign", "Scam"], ax=ax, colorbar=False)
    ax.set_title(f"Validation, threshold={threshold:.3f}")
    fig.tight_layout()
    fig.savefig(output_dir / "validation_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return model, result


def export_oof(data: pd.DataFrame, model: Pipeline, output_dir: Path) -> pd.DataFrame:
    """Fixed selected settings; refit vocabulary, IDF, classifier inside each fold."""
    train = data[data.included & data.split.eq("train")].copy().reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train["fold"] = -1
    train["scam_score"] = np.nan
    for fold, (fit, heldout) in enumerate(splitter.split(train.dialogue, train.label, train.normalized_sha256)):
        assert set(train.iloc[fit].normalized_sha256).isdisjoint(train.iloc[heldout].normalized_sha256)
        fold_model = clone(model).fit(train.iloc[fit].dialogue, train.iloc[fit].label)
        train.loc[heldout, "scam_score"] = fold_model.predict_proba(train.iloc[heldout].dialogue)[:, 1]
        train.loc[heldout, "fold"] = fold
    validation = data[data.included & data.split.eq("validation")].copy()
    validation["fold"] = -1  # -1 means classifier fitted on the complete eligible training partition
    validation["scam_score"] = model.predict_proba(validation.dialogue)[:, 1]
    scores = pd.concat([train, validation], ignore_index=True).assign(model="tfidf_logreg")
    scores = scores[["dialogue_id", "split", "label", "model", "scam_score", "fold"]]
    assert scores.dialogue_id.is_unique and scores.scam_score.between(0, 1).all()
    scores.to_csv(output_dir / "fusion_scores_train_validation.csv", index=False)
    train[["dialogue_id", "fold"]].to_csv(output_dir / "oof_folds.csv", index=False)
    return scores


def score_transcripts(bundle_path: str | Path, transcripts: list[str]) -> np.ndarray:
    """Return uncalibrated scam-intent ranking scores; load only trusted joblib files."""
    bundle = joblib.load(bundle_path)
    return bundle["pipeline"].predict_proba(transcripts)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scam_dialogue"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/text/baseline"))
    parser.add_argument("--require-a100", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "runtime.json", runtime_info(args.require_a100))
    data = load_data(args.data_dir, args.output_dir)
    print(json.dumps(audit(data, args.output_dir), indent=2))
    model, config = train_baseline(data, args.output_dir)
    export_oof(data, model, args.output_dir)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
