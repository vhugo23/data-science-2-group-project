"""Validation-only layer/pooling sweep, followed by one frozen test evaluation.

Run: python -m src.audio.tune [--full] [--stress]

--stress reuses the saved frozen selection (no new tuning) and scores the test split
under white noise, mu-law, and an untrimmed retrain, mirroring the original ablations.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.datasets import dataset_path

from .encoder import cache_path, encoder_features, load_encoder
from .evaluate import evaluate_scores, new_classifier, select_threshold
from .features import SETTINGS
from .workflow import prepare_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    output_dir = Path("outputs/audio") / ("full" if args.full else "smoke") / "layer_sweep"
    if args.stress:
        return stress(output_dir, args.full)
    frames = prepare_frames(dataset_path("asvspoof19_la"), output_dir, full=args.full)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, encoder = load_encoder(device)
    features = {}
    for split, frame in frames.items():
        print(f"Extracting {split}: {len(frame)} clips on {device}", flush=True)
        features[split] = encoder_features(frame, processor, encoder, device,
                                           cache=cache_path(output_dir, split, all_layers=True), all_layers=True)
    results = []
    for layer in range(features["train"].shape[1]):
        for pooling in ["mean", "mean_std"]:
            width = features["train"].shape[-1] // (2 if pooling == "mean" else 1)
            for C in [0.01, 0.1, 1, 10]:
                model = new_classifier(42, C=C).fit(features["train"][:, layer, :width], frames["train"]["label"])
                scores = model.predict_proba(features["validation"][:, layer, :width])[:, 1]
                threshold = select_threshold(frames["validation"]["label"], scores)
                results.append({"layer": layer, "pooling": pooling, "C": C,
                                **evaluate_scores(frames["validation"]["label"], scores, threshold)})
                pd.DataFrame(results).to_csv(output_dir / "validation_sweep.csv", index=False)
        print(f"Layer {layer} complete", flush=True)
    best = min(results, key=lambda row: (row["eer"], -row["roc_auc"]))
    (output_dir / "frozen_selection.json").write_text(json.dumps({**SETTINGS, **best}, indent=2))
    print("Frozen validation choice:", best, flush=True)
    width = features["train"].shape[-1] // (2 if best["pooling"] == "mean" else 1)
    model = new_classifier(42, C=best["C"]).fit(features["train"][:, best["layer"], :width], frames["train"]["label"])
    final_frames = prepare_frames(dataset_path("asvspoof19_la"), output_dir, full=args.full, final=True)
    test = encoder_features(final_frames["test"], processor, encoder, device,
                            cache=cache_path(output_dir, "test", all_layers=True), all_layers=True)
    scores = model.predict_proba(test[:, best["layer"], :width])[:, 1]
    metrics = evaluate_scores(final_frames["test"]["label"], scores, best["threshold"])
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Frozen test result:", metrics, flush=True)


def stress(output_dir, full):
    """Score the frozen selection on degraded and untrimmed test audio; never re-tunes."""
    best = json.loads((output_dir / "frozen_selection.json").read_text())
    layer, width_div = best["layer"], 2 if best["pooling"] == "mean" else 1
    frames = prepare_frames(dataset_path("asvspoof19_la"), output_dir, full=full, final=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, encoder = load_encoder(device)
    targets = {split: frame["label"].to_numpy() for split, frame in frames.items()}
    results_path = output_dir / "stress_metrics.csv"

    def features(split, condition="clean", trim=True):
        name = split if condition == "clean" else f"{split}_{condition}"
        values = encoder_features(frames[split], processor, encoder, device, condition=condition, seed=42,
                                  cache=cache_path(output_dir, name, all_layers=True, trim=trim),
                                  all_layers=True, trim=trim)
        return values[:, layer, :values.shape[-1] // width_div]

    def record(condition, scores, threshold):
        row = {"model": "wav2vec2_tuned", "dataset": "test", "condition": condition,
               **evaluate_scores(targets["test"], scores, threshold)}
        print(row, flush=True)
        saved = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
        saved = saved[saved["condition"] != condition] if len(saved) else saved
        pd.concat([saved, pd.DataFrame([row])], ignore_index=True).to_csv(results_path, index=False)

    model = new_classifier(42, C=best["C"]).fit(features("train"), targets["train"])
    record("clean", model.predict_proba(features("test"))[:, 1], best["threshold"])
    for condition in ["mulaw", "noise"]:
        print(f"Extracting test ({condition})", flush=True)
        record(condition, model.predict_proba(features("test", condition))[:, 1], best["threshold"])
    print("Extracting untrimmed train/validation/test", flush=True)
    untrimmed = new_classifier(42, C=best["C"]).fit(features("train", trim=False), targets["train"])
    threshold = select_threshold(targets["validation"],
                                 untrimmed.predict_proba(features("validation", trim=False))[:, 1])
    record("untrimmed", untrimmed.predict_proba(features("test", trim=False))[:, 1], threshold)


if __name__ == "__main__":
    main()
