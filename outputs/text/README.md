# Shared text outputs

These files let teammates reuse the trained text models and existing scores
after cloning/pulling the repository. No classifier training is needed.

| Folder / file | Purpose |
| --- | --- |
| `baseline/tfidf_logreg.joblib` | Complete fitted TF-IDF vocabulary, IDF, and logistic regression |
| `minilm/minilm_logreg.joblib` | Fitted logistic regression on frozen MiniLM vectors |
| `minilm/encoder/` | Saved pretrained encoder, tokenizer, pooling/chunking recipe, and license |
| Each model's `frozen_config.json` | Selected settings and decision threshold |
| `minilm/comparison_freeze.json` | Hashes that keep model/preprocessing selection fixed during evaluation |
| Each model's `fusion_scores_train_validation.csv` | OOF training and validation scores for Hugo |
| `minilm/validation_prefix_scores.csv` | Fixed-model validation scores at 25/50/75/100% of turns |
| `baseline/split_manifest.csv` and `oof_folds.csv` | Original IDs, split membership, exclusions, and OOF fold assignments |
| `minilm/embedding_cache/` | Main frozen development vectors; retained for reuse and artifact verification |
| `diagnostics/` | Metrics, predictions, fit traces, and figures supporting the completed audit |
| `input_provenance.json` | Checksums of prior Colab input archives; archives themselves are disposable |

Remaining CSV/JSON/PNG files record validation results, lengths, runtime versions,
and checkpoint verification. The selected models are unchanged. Temporary prefix
embedding caches, nonselected diagnostic checkpoints, transfer ZIPs, raw notebook
copies, and copied dataset dialogue text are excluded from the shared outputs.

The original 192-dialogue validation split selected both models. Challenge-set
review/locking and official-test comparison remain pending. The challenge answer
key is not part of this folder and stays local during independent review.

Use the existing inference/evaluation code in `src/text/` with the pinned
dependencies there. MiniLM's encoder computes embeddings for new inputs; that
is inference, not training. The current evaluation scripts use CUDA. The
`FrozenMiniLM` class also accepts `device="cpu"` when called directly.

The encoder/tokenizer are from
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41),
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Pretrained weights were not
fine-tuned. The model card declares Apache-2.0; a copy is included in
`minilm/encoder/LICENSE`.
