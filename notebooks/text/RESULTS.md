# Scam-intent baseline: methods and validation results

Completed 2026-10-29 using Google's Colab CLI from the VS Code workspace.
The notebook verified a **Colab NVIDIA A100-SXM4-40GB**, Python 3.13.16, and pinned
baseline dependencies. TF-IDF/logistic regression runs on CPU inside that runtime.
See `outputs/text/baseline/runtime.json` and notebook execution metadata.

Follow-up: [implementation, label-shuffle, and overlap diagnostics](DIAGNOSTICS.md)
preserve this original result and report separate sensitivity experiments.

## Data and audit

Loaded `BothBosu/scam-dialogue` at commit
`321b961b5ae353e19ed479b960658dcd223d5e06`; file SHA-256 values are checked before
loading. The provided training/test files contain 1,280/320 dialogues with equal
scam/benign classes and eight types. No missing text or conflicting duplicate
labels were found. `dialogue` is the sole feature; `label` is the target and
`type` is audit/evaluation metadata. The `all.csv` file is not appended.

Selected 192 validation dialogues from published training using label-stratified
sampling, seed 42. Source-row ordinals and raw-text hashes define stable IDs.
[The fixed manifest](split_manifest.csv) records all assignments and exclusions;
its SHA-256 is
`1fe25e23bff220b8c26efc211e5f2cc40b18b4d42135af4a80555a49a96e7559`.

| Split | Assigned | Included | Benign included | Scam included |
| --- | ---: | ---: | ---: | ---: |
| Training | 1,088 | 1,086 | 543 | 543 |
| Validation | 192 | 192 | 96 | 96 |
| Test | 320 | 320 | 160 | 160 |

Two raw-exact duplicate groups also match after Unicode normalization,
case-folding, and whitespace normalization. Excluded these training copies:

- `scam-train-0614-2ba5858a6a871fcc`: identical to published test row 125, a reward scam.
- `scam-train-1143-42298682e83aee87`: identical to training-source row 1138, assigned to validation, a wrong-number call.

All original test rows and labels remain unchanged. Assignment counts remain
1,088/192/320; fitting counts differ because exclusions are explicit.

Lengths range from 55–413 whitespace-separated words (median 250), 304–2,382
characters (median 1,443.5), and 6–28 speaker turns (median 13). These are not
MiniLM wordpiece counts; token-length measurement follows in that milestone.

Cross-split cosine search on stateless hashed character 5-grams found 56 pairs
at similarity ≥0.90, including the two exact pairs. After exact exclusions, 53
flagged pairs have both members included. We inspected word-level differences
for all 56 pairs. [The review table](audit/similarity_review.csv) records decisions;
full matching texts are saved locally in `similarity_review_text.json`.

Examples show why similarity needs interpretation:

- Train row 1164 versus test row 318 (0.9845) differs only in punctuation/casing and “thank you” versus “thanks”: a near-verbatim variant retained under the primary exact-only exclusion policy.
- Validation-source row 1138 versus training-source row 1183 (0.9712) changes the greeting and “trying to reach” to “looking for”: another near-verbatim variant.
- Train-source rows 1150/1240 (0.9027, assigned to validation/train) reuse a script for pizza versus auto repair: different entities/scenarios with a common template.

All non-exact candidates are variants of short synthetic wrong-number scripts.
No additional rows were excluded solely by similarity. **Residual near-duplicate
and template overlap remains**; this result is not a template-disjoint estimate.
A later sensitivity analysis can remove/group adjudicated variants without
moving or shrinking the published test set. See [audit summary](audit/summary.json)
and [source provenance](audit/provenance.json).

## Model selection and results

Each training-only Pipeline fits TF-IDF (`min_df=2`, sublinear term frequency,
default lowercasing/L2 normalization) then logistic regression (`liblinear`,
`max_iter=2000`, seed 42). Compared unigrams versus unigrams+bigrams and
`C ∈ {0.1, 1, 10}`. All six obtained validation F1 1.00 at threshold 0.5.
The deterministic tie-break selected **unigrams, C=0.1**. Validation F1 threshold
search selected **0.5**, preferring the threshold closest to 0.5 when tied.
The saved classifier uses all 1,086 eligible training dialogues; validation was
not added to fitting.

| Validation metric | Result |
| --- | ---: |
| Scam precision | 1.000 |
| Scam recall | 1.000 |
| Scam F1 | 1.000 |
| True negatives / false positives | 96 / 0 |
| False negatives / true positives | 0 / 96 |

The validation confusion matrix and length distributions are saved as PNGs in
`outputs/text/baseline/` and embedded in the executed notebook.

| Type | Validation n | Appropriate result |
| --- | ---: | --- |
| delivery | 24 | 0 false positives; accuracy 1.00 |
| insurance | 23 | 0 false positives; accuracy 1.00 |
| telemarketing | 28 | 0 false positives; accuracy 1.00 |
| wrong | 21 | 0 false positives; accuracy 1.00 |
| refund | 21 | Scam precision/recall/F1 1.00 |
| reward | 19 | Scam precision/recall/F1 1.00 |
| ssn | 30 | Scam precision/recall/F1 1.00 |
| support | 26 | Scam precision/recall/F1 1.00 |

Types are single-class. Scam recall/F1 are undefined for benign-only types;
accuracy and false-positive rate are reported there. Precision within a scam-only
type cannot measure confusion with benign examples in another type.
There are no validation false positives or false negatives to interpret, and no
error examples have been invented. `validation_errors.csv` has its header and no
rows. Error analysis will be revisited for the final comparison and prefixes.

This exceeds the exploratory F1 >0.85 target on validation. It does not establish
real-world reliability: the dialogues were generated with Meta-Llama-3-70B-Instruct,
topic vocabulary separates labels strongly, templates overlap, and validation was
used for selection. Scores are uncalibrated rankings. There is no final test claim.

## Fusion exports and verification

`fusion_scores_train_validation.csv` contains 1,086 training scores from five-fold
stratified group OOF fitting and 192 validation scores from the training-fitted
classifier. Every fold refits the complete TF-IDF/classifier Pipeline; normalized
text hashes keep exact copies together. Selected settings are fixed across folds.
Similar template variants may cross folds. See `src/text/README.md` for the schema
and ID contract. Hugo should pair only included IDs within their fixed split.

Verified 14 executed Colab notebook cells with no errors, original test membership,
fixed manifest identity, exact-duplicate separation, OOF coverage and fold labels,
score/target alignment, saved-model checksum, and reproduction of Colab validation
scores after reloading the model locally. Raw data, models, figures, and score
exports remain in ignored artifact folders.

The [MiniLM milestone and frozen validation-prefix comparison](MINILM_RESULTS.md)
are now complete. Final challenge and official-test comparisons await independent
challenge review and lock. No optional external dataset has been treated as an
independent test.

Sources: [pinned dataset card](https://huggingface.co/datasets/BothBosu/scam-dialogue/blob/321b961b5ae353e19ed479b960658dcd223d5e06/README.md),
[MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
