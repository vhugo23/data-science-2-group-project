# Investigating perfect validation performance

Executed 2026-10-29 on a verified Colab NVIDIA A100-SXM4-40GB using the CLI.
See [the executed diagnostic notebook](02_baseline_diagnostics.ipynb) and
[machine-readable summary](audit/diagnostic_summary.json). The original baseline,
split manifest, model, and fusion scores are preserved. This experiment reads
only the published training CSV; official test text is not loaded or scored.

## Implementation audit

Replayed all six baseline candidate fits and all five OOF fits with instrumentation
at `TfidfVectorizer.fit_transform` and `LogisticRegression.fit`. Every fit received
only known training dialogue strings, and every classifier target matched the
corresponding dialogue label. All training/validation IDs and included normalized
text hashes were disjoint. Each OOF vocabulary/IDF/classifier fit excluded the
held-out fold and all validation rows.

Independently reconstructed the 2,290-term saved vocabulary and smoothed IDF from
the 1,086 eligible training dialogues. These matched the saved preprocessing.
The replay reproduced original selected settings, threshold, validation scores,
and OOF scores. Full fit-ID traces are saved in
`outputs/text/diagnostics/fit_call_trace.json`.

**These checks found no implementation leakage in the paths tested.** They do
not resolve common templates or label-associated topic vocabulary.

## Shuffled-label negative controls

For seeds 0–19, permuted training labels, refitted the full Pipeline, and scored
the unchanged 192 validation dialogues against their original true labels.
Used the primary unigram/C=0.1 settings and threshold 0.5 throughout, without
retuning or selecting a favorable permutation.

| Metric | Mean | Standard deviation | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Accuracy / balanced accuracy | 0.521 | 0.172 | 0.229 | 0.755 |
| ROC-AUC | 0.522 | 0.221 | 0.142 | 0.804 |
| Scam F1 | 0.517 | 0.174 | 0.253 | 0.754 |

Mean accuracy/AUC are near their chance expectation of 0.5. F1 has no universal
chance value. Individual controls vary substantially; they should not be read
as 20 tightly concentrated estimates from independent validation examples.
Shared topics/templates could contribute to the variability; this is an
interpretation, not a diagnosed cause. The controls do not establish independent
same-topic or real-world performance. No shuffled-label run achieved the
primary perfect result.

## Training/validation overlap sensitivity

There are 30 eligible flagged training/validation pairs. Before fitting these
experiments, used prior text review to identify four near-verbatim pairs sharing
the same scenario, participants, and turn structure with only small phrase
changes. The coding agent reviewed these; this is not independent teammate review.
Removed their training counterparts only:

- source row 1183, paired with validation-source row 1138;
- source row 1185, paired with validation-source row 1200;
- source row 1142, paired with validation-source row 1236;
- source row 1218, paired with validation-source row 1201.

A second, deliberately broader stress test removed all 28 unique training
counterparts flagged against validation at character-hash cosine ≥0.90,
regardless of whether source identity could be established. This broad policy
does not designate every flagged pair as a duplicate. See
[pair decisions](audit/diagnostic_overlap_decisions.csv).

Every experiment retained all 192 validation dialogues, the same unigram/C=0.1
settings, and threshold 0.5. No validation examples were moved or dropped, and no
hyperparameters or thresholds were retuned.

| Experiment | Fitting rows | Additional removals | Scam precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original reference | 1,086 | 0 | 1.000 | 1.000 | 1.000 |
| Reviewed near-verbatim counterparts removed | 1,082 | 4 | 1.000 | 1.000 | 1.000 |
| All flagged validation counterparts removed | 1,058 | 28 | 1.000 | 1.000 | 1.000 |

Each confusion matrix remains `[[96, 0], [0, 96]]`; no validation prediction
changed. Mean absolute score changes were 0.000717 for strict removals and
0.005317 for broad removals. All removed examples were benign wrong-number
dialogues; scam training examples remained unchanged. Each type retained perfect
validation accuracy. Scores, per-type tables, removed IDs, figures, and both
sensitivity classifiers are saved under `outputs/text/diagnostics/`.

These specific overlaps do not explain the perfect validation result. This does
not demonstrate that all paraphrase/template overlap has been removed, especially
below the search threshold or within OOF folds. Topic-label confounding remains
untested by these removals.

## Interpretation and fusion handoff

The evidence supports retaining the reported perfect baseline result while
limiting its interpretation to this synthetic benchmark. Topic shortcuts remain
a leading hypothesis, not an established diagnosis. The next informative
diagnostic is a fixed supplemental set with both labels represented within the
same topics: legitimate refunds/support and fraudulent delivery/insurance calls.
Disclose construction, obtain independent teammate labels before scoring, and
keep it separate from the original benchmark. That set has not yet been evaluated.

Hugo's handoff caveat: text scores separate these synthetic categories very well;
the fusion branch could inherit that ease without establishing real-call intent
detection. Compare component and fused performance under explicit stress tests.
No messages have been sent to teammates. MiniLM and incomplete-transcript
comparisons should use the same fixed conditions, with official test evaluation
reserved until settings and protocols are frozen.
