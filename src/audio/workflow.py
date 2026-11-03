"""Shared split manifests and out-of-fold score exports for audio notebooks."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .evaluate import evaluate_scores, new_classifier, select_threshold
from .features import SETTINGS, load_protocol


def prepare_frames(la_dir, output_dir, full=False, final=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "split_manifest.csv"
    frames = {}
    for official, split in [("train", "train"), ("dev", "validation"), ("eval", "test")]:
        if split == "test" and not final:
            continue
        frame, _ = load_protocol(la_dir, official)
        if not full:
            frame = frame.groupby("key").sample(n=20, random_state=42).copy()
        frames[split] = frame.reset_index(drop=True)
    folds = StratifiedKFold(n_splits=5 if full else 3, shuffle=True, random_state=42)
    train = frames["train"]
    train["fold"] = -1
    for fold, (_, held_out) in enumerate(folds.split(train, train["label"])):
        train.loc[held_out, "fold"] = fold
    manifest = pd.concat([frame[["sample_id", "label"]].assign(split=split,
                          fold=frame["fold"] if split == "train" else -1)
                          for split, frame in frames.items()], ignore_index=True)
    if manifest_path.exists():
        saved = pd.read_csv(manifest_path)
        shared = saved["split"].isin(frames)
        pd.testing.assert_frame_equal(saved.loc[shared].reset_index(drop=True), manifest)
    else:
        manifest.to_csv(manifest_path, index=False)
    return frames


def fit_export(name, features, frames, output_dir):
    targets = {split: frame["label"].to_numpy() for split, frame in frames.items()}
    model = new_classifier(42)
    model.fit(features["train"], targets["train"])
    scores = {split: model.predict_proba(values)[:, 1] for split, values in features.items()}
    threshold = select_threshold(targets["validation"], scores["validation"])
    assignments = frames["train"]["fold"].to_numpy()
    folds = [(np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold))
             for fold in np.unique(assignments)]
    scores["train"] = cross_val_predict(new_classifier(42), features["train"], targets["train"],
                                        cv=folds, method="predict_proba")[:, 1]
    tables = []
    for split, frame in frames.items():
        table = frame[["sample_id", "label"]].copy()
        table["split"] = split
        table["score"] = scores[split]
        table["fold"] = assignments if split == "train" else -1
        tables.append(table[["sample_id", "split", "label", "score", "fold"]])
    pd.concat(tables, ignore_index=True).to_csv(output_dir / f"{name}_scores.csv", index=False)
    metrics = pd.DataFrame([{"model": name, "dataset": split, "condition": "clean",
                            **evaluate_scores(targets[split], scores[split], threshold)}
                           for split in frames if split != "train"])
    metrics.to_csv(output_dir / f"{name}_metrics.csv", index=False)
    joblib.dump(model, output_dir / f"{name}_classifier.joblib")
    (output_dir / f"{name}_settings.json").write_text(json.dumps(
        {**SETTINGS, "threshold": float(threshold), "seed": 42,
         "train_scores": "stratified_out_of_fold", "test_evaluated": "test" in frames}, indent=2))
    return model, threshold, metrics
