# Text / scam-intent branch

Start with the MiniLM notebook for the current model, or the baseline notebook
for the beginning of the pipeline. The detailed run commands live in
[tools/README.md](tools/README.md).

| What you need | Where to look |
| --- | --- |
| TF-IDF baseline notebook | [01_scam_intent_baseline.ipynb](../../notebooks/text/01_scam_intent_baseline.ipynb) |
| Why the perfect baseline was investigated | [02_baseline_diagnostics.ipynb](../../notebooks/text/02_baseline_diagnostics.ipynb) |
| Frozen MiniLM notebook | [03_minilm_scam_intent.ipynb](../../notebooks/text/03_minilm_scam_intent.ipynb) |
| Current results and limitations | [MINILM_RESULTS.md](../../notebooks/text/MINILM_RESULTS.md) |
| File to share with Justin and Hugo | [review_packet.zip](../../notebooks/text/challenge_v1/review_packet.zip) |
| Review / lock instructions | [challenge_v1/README.md](../../notebooks/text/challenge_v1/README.md) |
| Fixed dialogue IDs and splits | [split_manifest.csv](../../notebooks/text/split_manifest.csv) |
| Saved TF-IDF model and scores | `outputs/text/baseline/` |
| Saved MiniLM model, embeddings, and scores | `outputs/text/minilm/` |
| Completed diagnostic evidence | `outputs/text/diagnostics/` |

## Source files

- `baseline.py`: dataset audit and TF-IDF training.
- `minilm.py`: frozen chunk embeddings, classifier, OOF and prefix scores.
- `diagnostics.py`: completed leakage/shuffle/overlap experiments.
- `challenge.py`: create review files and lock actual reviewed labels.
- `challenge_content.py`: local author draft specifications; excluded from Git during blind review.
- `comparison.py`: evaluate locked challenge, then official test, with fixed models.
- `tools/`: notebook builders, Colab helpers, artifact verification, and full commands.
- `tests/`: structural tests for the review/scoring gates.
- `requirements-*.txt`: isolated baseline and MiniLM environments.

## Current state

Both models achieved validation F1 1.00. The 48-dialogue challenge is drafted but
awaits Justin and Hugo's independent labels. Challenge and official-test scoring
remain gated until review, adjudication, and lock are complete.

The review packet ZIP is the only ZIP you need to share. Keep `author_only/`
separate during initial review. The generated transfer ZIPs and raw notebook
copies were moved to a recoverable folder outside the project after verifying
the extracted artifacts and final notebooks.
Builders can recreate transfer files when another Colab run is needed.

Hugo's score files are named `fusion_scores_train_validation.csv` in each model
output folder. Join on `dialogue_id` and `split`; training folds 0?4 are OOF,
validation fold -1 uses the full eligible training partition. The MiniLM output
folder also contains `validation_prefix_scores.csv` for both models.
