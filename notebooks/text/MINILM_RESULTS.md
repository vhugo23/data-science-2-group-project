# Frozen MiniLM and the pending topic-controlled challenge

Executed 2026-10-29 on Colab NVIDIA A100-SXM4-40GB through the CLI. The
[executed notebook](03_minilm_scam_intent.ipynb) embeds the source and records the
runtime. Model selection used the original 1,086 eligible training and 192
validation dialogues. No draft challenge transcripts or official test text were
loaded by this run. The original TF-IDF artifacts remain unchanged.

## Frozen representation and reproducibility

Loaded `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` using Transformers. Applied the
[official attention-mask mean-pooling recipe](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
with L2 normalization. Every encoder parameter has `requires_grad=False`, the
encoder stays in evaluation mode, and inference runs on CUDA in float32. No
optimizer or encoder fine-tuning is used.

To address the 256-wordpiece input limit, tokenize each full dialogue without
truncation and partition its original token IDs into nonoverlapping chunks of
at most 254 content tokens, reserving two special tokens. Encode chunks, average
their normalized vectors equally, then L2-normalize the dialogue mean. Chunks
remain attached to their dialogue and never become independently split examples.
This fixed chunking/pooling policy was chosen before challenge scoring.

Among 1,278 included development dialogues, content lengths range from 80 to
563 wordpieces, with median 339. **963 dialogues (75.4%) require chunking**:
808 training and 155 validation dialogues. They produce 2,330 chunks in total.
These counts cover training/validation; official test wordpiece measurement is
reserved for final evaluation.

Cached vectors include ordered dialogue IDs, raw-text hashes, model revision,
recipe, token counts, chunk counts, and a checksum. Cache reuse was checked
without re-encoding. Saved artifacts include the actual encoder weights,
tokenizer, embedding recipe, classifier, and model/threshold freeze hashes.
Encoder fingerprints were identical before and after all classifier/prefix work.
A fresh saved-encoder load reproduced 1-, 2-, and 3-chunk dialogue vectors with
maximum absolute difference `1.49e-7` from cached vectors; classifier scores also
matched within the declared numerical tolerance.

The run used Python 3.13.16, PyTorch 2.11.0+cu130, Transformers 4.57.1, tokenizers
0.22.1, safetensors 0.6.2, and the pinned baseline numerical dependencies. See
`outputs/text/minilm/runtime.json`, [summary](audit/minilm_summary.json), and
[checkpoint verification](audit/minilm_checkpoint_verification.json).

## Validation selection and comparison

Fit logistic regression on frozen training vectors only (`liblinear`, seed 42,
`max_iter=2000`) with C in {0.1, 1, 10}. Select C by validation F1 at 0.5, preferring
smaller C when tied. Then choose the threshold by validation F1, preferring the
threshold closest to 0.5 when tied. No validation refit is performed.

| MiniLM C | Validation precision at 0.5 | Recall | F1 |
| --- | ---: | ---: | ---: |
| 0.1 | 0.9897 | 1.0000 | 0.9948 |
| 1 | 1.0000 | 1.0000 | 1.0000 |
| 10 | 1.0000 | 1.0000 | 1.0000 |

Selected **C=1, threshold=0.5**. Both selected models have validation precision,
recall, and F1 of 1.00, with confusion matrix `[[96, 0], [0, 96]]`. Every dialogue
type has validation accuracy 1.00. There are no selected-model full-dialogue
validation errors to interpret. The original validation set does not distinguish
these selected models. This is a tuning-set result on synthetic dialogues, not
an independent generalization claim.

MiniLM exports 1,086 OOF training scores plus 192 validation scores using the
same IDs and five fold assignments as TF-IDF. Each OOF logistic regression fits
only the other four folds. Cached encoder outputs can be reused across folds
because the encoder and normalization do not learn from this dataset. No scaler
or other data-fitted embedding preprocessing is applied.

## Incomplete-information results, fixed models and thresholds

Evaluated the first ceil(fraction × turn count) speaker turns on the same 192
validation examples, retaining each full dialogue's label. The 100% condition
uses the original unmodified text. Settings were fixed before these runs and
were not retuned using prefix results.

| First portion of turns | TF-IDF scam F1 | MiniLM scam F1 | MiniLM false negatives |
| --- | ---: | ---: | ---: |
| 25% | 1.000 | 0.939 | 11 |
| 50% | 1.000 | 0.995 | 1 |
| 75% | 1.000 | 1.000 | 0 |
| 100% | 1.000 | 1.000 | 0 |

Neither selected model produced a false positive in these validation-prefix
conditions. TF-IDF has the stronger early-prefix result on this dataset; no
MiniLM advantage is claimed.

Several MiniLM false negatives illustrate the incomplete-information limit:

- `scam-train-0000-ffd548fb813511af`, 25%, score 0.354: the prefix contains greetings
  and a claimed Social Security Administration affiliation, but ends before an
  SSN request or threat. The full-conversation scam label is not established by
  the visible behavior alone.
- `scam-train-0206-563d70b6dce6a5f0`, 25%, score 0.173: the caller claims to be customer
  service and mentions a delivery problem and refund. The prefix does not yet
  include the later request that establishes the scam.
- `scam-train-0539-e28e01b913863330`, 25%, score 0.234: the prefix ends after a customer
  service introduction, before the reward offer or associated request appears.
- `scam-train-0315-7bcb1e4969958244`, 50%, score 0.488: the caller asks for an account
  email and claims a laptop purchase the recipient does not recognize. This is
  suspicious context, but the visible prefix still lacks the later decisive scam
  request. The score is just below the unchanged threshold.

These are errors relative to the full-dialogue label, not proof that fraud was
identifiable at the early observation point. TF-IDF's perfect 25% result is
consistent with strong early category cues; it does not diagnose the exact
mechanism. Fusion-ready validation prefix scores are saved separately with
`turn_fraction`, so Hugo can preserve each condition rather than mix observations.

## Constructed challenge and next evaluation gate

The fixed draft [blind packet](challenge_v1/review_packet.zip) contains 48 fictional
dialogues, targeting six legitimate and six scam examples per refund, technical
support, delivery, and insurance topic. It uses 24 paired scenarios with shared
opening context and different service requests. All texts have 12 turns and
283–322 whitespace-separated words, with intended-label means 305.25 and 303.875.
Within-pair word-length ratios are at most 1.021. Author targets, topic/pair IDs,
and comparability tables are separated from the blind packet.

Justin and Hugo's actual review sheets and attestations are blank. No independent
labels, consensus, locked challenge, challenge scores, or official test results
are claimed. See [review instructions](challenge_v1/README.md). Structural gate
tests use temporary synthetic fixtures and never fill the real review forms.

After actual independent labels and adjudication, lock the text and final labels,
evaluate both unchanged models, then evaluate the official test without tuning
against either result. The [comparison freeze](audit/comparison_freeze.json)
records selected classifier, encoder/tokenizer, preprocessing, and threshold
hashes. The evaluation entry points enforce the review/lock and evaluation order.
The constructed paired set is a supplemental diagnostic; even a successful result
would leave broader real-world generalization untested.
