"""Frozen MiniLM chunk embeddings and train-only logistic regression.

Uses the official Transformers mean-pooling recipe for all-MiniLM-L6-v2.
No challenge examples or official test text are loaded during model selection.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from transformers import AutoModel, AutoTokenizer

from src.text.baseline import SEED, metrics, prefix, runtime_info, sha256, write_json
from src.text.diagnostics import load_development

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RECIPE = {
    "model_id": MODEL_ID, "revision": MODEL_REVISION, "max_sequence_tokens": 256,
    "chunking": "nonoverlapping slices of original content token IDs; reserve tokenizer special tokens",
    "chunk_pooling": "attention-mask mean of final hidden states, then L2 normalization",
    "dialogue_pooling": "equal mean of chunk vectors, then L2 normalization",
    "dtype": "float32", "recipe_version": 1,
}


class FrozenMiniLM:
    def __init__(self, model_path: str | Path = MODEL_ID, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the requested embedding run")
        self.device = torch.device(device)
        kwargs = {"revision": MODEL_REVISION} if str(model_path) == MODEL_ID else {}
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False, token=False, **kwargs)
        self.encoder = AutoModel.from_pretrained(str(model_path), trust_remote_code=False, token=False, use_safetensors=True, **kwargs)
        self.encoder.requires_grad_(False)
        self.encoder.eval().to(self.device)
        torch.manual_seed(SEED)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.content_budget = 256 - self.tokenizer.num_special_tokens_to_add(pair=False)
        assert self.content_budget == 254
        assert not self.encoder.training and not any(p.requires_grad for p in self.encoder.parameters())

    def fingerprint(self) -> str:
        import hashlib
        digest = hashlib.sha256()
        for name, tensor in self.encoder.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def encode(self, ids: list[str], texts: list[str], cache_dir: Path, batch_size: int = 64) -> tuple[np.ndarray, pd.DataFrame, dict]:
        assert len(ids) == len(texts) and len(ids) == len(set(ids))
        text_hashes = [sha256(text) for text in texts]
        key = sha256(json.dumps({"recipe": RECIPE, "ids": ids, "text_sha256": text_hashes}, sort_keys=True))
        cache_dir.mkdir(parents=True, exist_ok=True)
        path, metadata_path = cache_dir / f"{key}.npz", cache_dir / f"{key}.json"
        if path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            assert metadata["npz_sha256"] == sha256(path.read_bytes()), "Embedding cache checksum mismatch"
            with np.load(path, allow_pickle=False) as cached:
                assert cached["ids"].tolist() == ids and cached["text_sha256"].tolist() == text_hashes
                embeddings = cached["embeddings"].copy()
                lengths = cached["content_wordpieces"].copy()
                chunk_counts = cached["chunks"].copy()
            metadata = dict(metadata, cache_hit=True)
        else:
            features, counts, lengths = [], [], []
            for text in texts:
                token_ids = self.tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
                lengths.append(len(token_ids))
                chunks = [token_ids[start:start + self.content_budget] for start in range(0, len(token_ids), self.content_budget)] or [[]]
                assert sum(map(len, chunks)) == len(token_ids)
                counts.append(len(chunks))
                for chunk in chunks:
                    item = self.tokenizer.prepare_for_model(chunk, add_special_tokens=True, truncation=False, return_attention_mask=True)
                    assert len(item["input_ids"]) <= 256
                    features.append(item)
            vectors = []
            with torch.inference_mode():
                for start in range(0, len(features), batch_size):
                    batch = self.tokenizer.pad(features[start:start + batch_size], padding=True, return_tensors="pt")
                    batch = {key: value.to(self.device) for key, value in batch.items()}
                    hidden = self.encoder(**batch).last_hidden_state
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    vectors.append(F.normalize(pooled, p=2, dim=1).cpu().numpy())
            chunk_vectors = np.concatenate(vectors).astype(np.float32)
            rows, offset = [], 0
            for count in counts:
                mean = chunk_vectors[offset:offset + count].mean(axis=0)
                rows.append(mean / max(float(np.linalg.norm(mean)), 1e-12))
                offset += count
            assert offset == len(chunk_vectors)
            embeddings = np.asarray(rows, dtype=np.float32)
            lengths, chunk_counts = np.asarray(lengths), np.asarray(counts)
            np.savez_compressed(path, ids=np.asarray(ids), text_sha256=np.asarray(text_hashes),
                                embeddings=embeddings, content_wordpieces=lengths, chunks=chunk_counts)
            metadata = {"cache_key": key, "npz_sha256": sha256(path.read_bytes()), "recipe": RECIPE,
                        "dialogues": len(ids), "chunks": int(chunk_counts.sum()), "content_budget": self.content_budget,
                        "cache_hit": False, "labels_used": False,
                        "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "tokenizers", "safetensors"]}}
            write_json(metadata_path, metadata)
        assert embeddings.shape == (len(ids), 384) and np.isfinite(embeddings).all()
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=2e-6)
        stats = pd.DataFrame({"dialogue_id": ids, "text_sha256": text_hashes,
                              "content_wordpieces": lengths, "wordpieces_with_special_tokens": lengths + 2,
                              "chunks": chunk_counts, "would_truncate_at_256": lengths > self.content_budget})
        return embeddings, stats, metadata

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        write_json(path / "embedding_recipe.json", RECIPE)


def train_classifier(data: pd.DataFrame, embeddings: np.ndarray, output: Path) -> tuple[LogisticRegression, dict]:
    train_mask, val_mask = data.split.eq("train").to_numpy(), data.split.eq("validation").to_numpy()
    labels = data.label.to_numpy()
    candidates, fitted = [], []
    for c in [0.1, 1.0, 10.0]:
        model = LogisticRegression(C=c, solver="liblinear", max_iter=2000, random_state=SEED)
        model.fit(embeddings[train_mask], labels[train_mask])
        scores = model.predict_proba(embeddings[val_mask])[:, 1]
        candidates.append({"C": c, **metrics(labels[val_mask], scores, 0.5)})
        fitted.append((model, scores))
    best = max(range(len(candidates)), key=lambda i: candidates[i]["f1"])
    classifier, scores = fitted[best]
    threshold = max(sorted(set([0.0, 0.5, 1.0, *map(float, scores)])), key=lambda t: (
        f1_score(labels[val_mask], scores >= t, zero_division=0), -abs(t - 0.5), t))
    config = {"model": "minilm_logreg", "C": candidates[best]["C"], "threshold": threshold,
              "threshold_rule": "score >= threshold", "fit_n": int(train_mask.sum()), "seed": SEED,
              "embedding_recipe": RECIPE, "encoder_frozen": True,
              "selection": "Original validation F1 at 0.5; ties smallest C; then threshold maximizes validation F1 with ties closest to 0.5",
              "validation_at_0_5": metrics(labels[val_mask], scores, 0.5),
              "validation": metrics(labels[val_mask], scores, threshold), "challenge_scored": False, "test_evaluated": False}
    pd.DataFrame(candidates).drop(columns="confusion_matrix").to_csv(output / "validation_search.csv", index=False)
    bundle_path = output / "minilm_logreg.joblib"
    joblib.dump({"classifier": classifier, "config": config}, bundle_path)
    config["saved_classifier_sha256"] = sha256(bundle_path.read_bytes())
    write_json(output / "frozen_config.json", config)
    val = data.loc[val_mask, ["dialogue_id", "type", "label", "dialogue"]].assign(scam_score=scores, prediction=(scores >= threshold).astype(int))
    val.drop(columns="dialogue").to_csv(output / "validation_predictions.csv", index=False)
    errors = val[val.label.ne(val.prediction)].copy()
    errors["error_kind"] = np.where(errors.label.eq(0), "false_positive", "false_negative")
    errors.to_csv(output / "validation_errors.csv", index=False)
    per_type = []
    for kind, group in val.groupby("type"):
        result = metrics(group.label, group.scam_score, threshold)
        if not result["scam_support"]:
            result.update(precision=None, recall=None, f1=None)
        per_type.append({"type": kind, **result})
    pd.DataFrame(per_type).drop(columns="confusion_matrix").to_csv(output / "validation_by_type.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(val.label, val.prediction, labels=[0, 1], display_labels=["Benign", "Scam"], colorbar=False, ax=ax)
    ax.set_title(f"MiniLM validation, threshold={threshold:.3f}")
    fig.tight_layout()
    fig.savefig(output / "validation_confusion_matrix.png", dpi=160)
    plt.close(fig)
    return classifier, config


def export_scores(data: pd.DataFrame, embeddings: np.ndarray, classifier, folds: pd.DataFrame, output: Path):
    index = data.set_index("dialogue_id").assign(position=np.arange(len(data)))
    train = data[data.split.eq("train")].merge(folds, on="dialogue_id", validate="one_to_one")
    assert len(train) == 1086 and set(train.fold) == set(range(5))
    rows = []
    for fold in range(5):
        fit, heldout = train[train.fold.ne(fold)], train[train.fold.eq(fold)]
        assert set(fit.dialogue_id).isdisjoint(heldout.dialogue_id)
        model = clone(classifier).fit(embeddings[index.loc[fit.dialogue_id, "position"]], fit.label)
        scores = model.predict_proba(embeddings[index.loc[heldout.dialogue_id, "position"]])[:, 1]
        rows.extend({"dialogue_id": ident, "split": "train", "label": int(label), "model": "minilm_logreg", "scam_score": float(score), "fold": fold}
                    for ident, label, score in zip(heldout.dialogue_id, heldout.label, scores))
    val = data[data.split.eq("validation")]
    scores = classifier.predict_proba(embeddings[index.loc[val.dialogue_id, "position"]])[:, 1]
    rows.extend({"dialogue_id": ident, "split": "validation", "label": int(label), "model": "minilm_logreg", "scam_score": float(score), "fold": -1}
                for ident, label, score in zip(val.dialogue_id, val.label, scores))
    frame = pd.DataFrame(rows)
    assert frame.dialogue_id.is_unique and frame.scam_score.between(0, 1).all()
    frame.to_csv(output / "fusion_scores_train_validation.csv", index=False)
    folds.to_csv(output / "oof_folds.csv", index=False)
    return frame


def validation_prefixes(data, encoder, classifier, config, baseline_reference: Path, output: Path):
    baseline = joblib.load(baseline_reference / "tfidf_logreg.joblib")
    val = data[data.split.eq("validation")]
    rows, reports = [], []
    for fraction in [0.25, 0.5, 0.75, 1.0]:
        texts = [text if fraction == 1 else prefix(text, fraction) for text in val.dialogue]
        vectors, _, _ = encoder.encode(val.dialogue_id.tolist(), texts, output / "embedding_cache")
        for name, scores, threshold in [
            ("minilm_logreg", classifier.predict_proba(vectors)[:, 1], config["threshold"]),
            ("tfidf_logreg", baseline["pipeline"].predict_proba(texts)[:, 1], baseline["config"]["threshold"]),
        ]:
            rows.extend({"dialogue_id": ident, "split": "validation", "label": int(label), "model": name,
                         "scam_score": float(score), "fold": -1, "turn_fraction": fraction}
                        for ident, label, score in zip(val.dialogue_id, val.label, scores))
            reports.append({"model": name, "turn_fraction": fraction, "threshold": threshold, **metrics(val.label, scores, threshold)})
    pd.DataFrame(rows).to_csv(output / "validation_prefix_scores.csv", index=False)
    table = pd.DataFrame(reports).drop(columns="confusion_matrix")
    table.to_csv(output / "validation_prefix_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, group in table.groupby("model"):
        ax.plot(group.turn_fraction * 100, group.f1, marker="o", label=name)
    ax.set(xlabel="First percentage of conversation turns", ylabel="Scam F1", ylim=(0, 1.03), xticks=[25, 50, 75, 100])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "validation_prefix_performance.png", dpi=160)
    plt.close(fig)


def run_minilm(data_dir: Path, manifest: Path, reference: Path, output: Path, require_a100=True):
    output.mkdir(parents=True, exist_ok=True)
    runtime = runtime_info(require_a100)
    runtime.update(torch_version=torch.__version__, transformers_version=importlib.metadata.version("transformers"))
    write_json(output / "runtime.json", runtime)
    data = load_development(data_dir, manifest)
    data = data[data.included].reset_index(drop=True)
    encoder = FrozenMiniLM()
    before = encoder.fingerprint()
    vectors, stats, cache = encoder.encode(data.dialogue_id.tolist(), data.dialogue.tolist(), output / "embedding_cache")
    stats = stats.merge(data[["dialogue_id", "split", "label", "type"]], on="dialogue_id", validate="one_to_one")
    stats.to_csv(output / "wordpiece_lengths.csv", index=False)
    cached_vectors, _, cache_again = encoder.encode(data.dialogue_id.tolist(), data.dialogue.tolist(), output / "embedding_cache")
    np.testing.assert_array_equal(vectors, cached_vectors)
    assert cache_again["cache_hit"]
    classifier, config = train_classifier(data, vectors, output)
    folds = pd.read_csv(reference / "oof_folds.csv")
    scores = export_scores(data, vectors, classifier, folds, output)
    encoder.save(output / "encoder")
    assert before == encoder.fingerprint()
    baseline_config = json.loads((reference / "frozen_config.json").read_text())
    assert sha256((reference / "tfidf_logreg.joblib").read_bytes()) == baseline_config["saved_model_sha256"]
    freeze = {"selection_data": "Original fixed validation split only", "no_challenge_or_test_tuning": True,
              "tfidf_model_sha256": baseline_config["saved_model_sha256"],
              "tfidf_config_sha256": sha256((reference / "frozen_config.json").read_bytes()),
              "minilm_classifier_sha256": config["saved_classifier_sha256"],
              "minilm_config_sha256": sha256((output / "frozen_config.json").read_bytes()),
              "minilm_encoder_fingerprint": before, "embedding_recipe": RECIPE,
              "encoder_files_sha256": {p.name: sha256(p.read_bytes()) for p in sorted((output / "encoder").iterdir()) if p.is_file()},
              "minilm_source_sha256": sha256(Path(__file__).read_text(encoding="utf-8")),
              "tfidf_threshold": baseline_config["threshold"], "minilm_threshold": config["threshold"],
              "challenge_state": "Awaiting independent review and locked final labels", "official_test_evaluated": False}
    write_json(output / "comparison_freeze.json", freeze)
    validation_prefixes(data, encoder, classifier, config, reference, output)
    assert before == encoder.fingerprint()
    summary = {"config": config, "encoder_frozen": True, "trainable_encoder_parameters": 0,
               "encoder_fingerprint_before_and_after": before,
               "embedding_cache_reuse_verified": True, "embedding_cache": cache,
               "included_development_dialogues": len(data), "longer_than_256_with_special_tokens": int(stats.would_truncate_at_256.sum()),
               "wordpiece_summary": stats.content_wordpieces.describe().to_dict(),
               "length_scope": "Included original training/validation only; test wordpiece lengths deferred until final evaluation",
               "fusion_score_rows": len(scores), "challenge_scored": False, "test_evaluated": False}
    write_json(output / "minilm_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/scam_dialogue"))
    parser.add_argument("--manifest", type=Path, default=Path("notebooks/text/split_manifest.csv"))
    parser.add_argument("--baseline-reference", type=Path, default=Path("outputs/text/baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/text/minilm"))
    args = parser.parse_args()
    print(json.dumps(run_minilm(args.data_dir, args.manifest, args.baseline_reference, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
