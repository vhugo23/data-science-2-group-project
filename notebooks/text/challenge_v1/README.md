# Topic-controlled challenge set — awaiting independent review

Share **only [review_packet.zip](review_packet.zip)** with Justin and Hugo.
It contains a blind HTML reading copy, transcripts, two separate blank label
sheets, reviewer instructions, a text-hash manifest, and blank attestations.
Keep `author_only/` and the construction source out of the initial review.
The author key and label-bearing construction source are intentionally excluded
from Git until independent review finishes. The project owner performs initial
adjudication/locking locally with that key. Review and subsequent scoring do not
require publishing it; public tests use unrelated synthetic fixtures.

The draft contains 48 fictional dialogues: six intended legitimate and six
intended scam examples for each of refund, technical support, delivery, and
insurance. These are construction targets, not reviewed ground truth. Each of
24 scenarios has a paired continuation with shared context but different
requests/behavior. Every dialogue has 12 turns and 283–322 whitespace-separated
words. Intended-label mean lengths are 305.25 and 303.875 words; the largest
within-pair length ratio is 1.021. No model scores were consulted during drafting.

This is a constructed diagnostic set, not an independent real-call benchmark.
Paired examples are related observations. Their first six turns are identical
within a pair, so the differing intent is absent from first-25%/50% prefixes by
construction. Results should be reported by topic and paired scenario, with
the construction and this prefix limitation disclosed.

## Review and lock

1. Justin completes `review_packet/justin_labels.csv` and Hugo completes
   `review_packet/hugo_labels.csv` independently. Each supplies label 0/1 or
   `uncertain`, confidence 1–5, a rationale, and an exact supporting quotation.
2. Each reviewer personally completes their own attestation. Neither initial
   review should consult author targets, the other review, or model scores.
3. After both initial reviews, discuss disagreements, uncertainty, and conflicts
   with the author's intended label. Record resolutions in `adjudication.csv`,
   including final label, resolver identity, and reason. Consensus rows agreeing
   with author intent need no separate adjudication entry.
4. If wording is ambiguous or incoherent, create a new unscored version and have
   changed examples reviewed again. Do not force a label to preserve balance.
   The final set must retain six examples of each label per topic and both labels
   within each pair. The gate records attestations; it cannot establish the
   authenticity or quality of human review by itself.
5. From the repository root, using the baseline Python environment:

```bash
python -m src.text.challenge lock --directory notebooks/text/challenge_v1
python -m src.text.challenge check --directory notebooks/text/challenge_v1
```

The lock writes `locked_dialogues.csv` and `challenge_lock.json`, containing
checksums of the text, final labels, initial reviews, attestations, and
adjudication. Completed labels are required; no reviews have been fabricated.
The current real review sheets and attestations remain blank.

## Scoring order

Both classifiers are selected only on the original validation split. MiniLM's
`comparison_freeze.json` records classifier, encoder/tokenizer, preprocessing,
and threshold hashes before any challenge scoring. In a CUDA environment with
the saved artifacts available, use:

```bash
python -m src.text.comparison challenge
python -m src.text.comparison test
```

The first command fails before loading model inputs unless review and locking
are complete. The second additionally requires a challenge-evaluation receipt
matching the unchanged lock and frozen models. It does not read official test
text before those gates pass. It then exports final metrics, per-type results,
errors, confusion matrices, full scores, and 25/50/75/100% prefix scores for both
models. Preserve results and do not retune using either evaluation set.

For Colab execution, upload the **locked** challenge folder, saved baseline and
MiniLM artifact directories, fixed manifest, and the text source modules to a
runtime; run these commands from their repository-relative layout. The existing
MiniLM training notebook intentionally contains no draft challenge transcripts.
The local packet is ready to share; evaluation is pending actual independent
review, adjudication, and lock.
