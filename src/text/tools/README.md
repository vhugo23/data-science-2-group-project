# Text track: reproduction commands and helper tools

Current assignment: this branch owns text scam intent; Justin owns audio and Hugo
owns pairing/fusion. This supersedes the older ownership table in the root README.

For the short folder guide, see [../README.md](../README.md).
Helper scripts now live in this folder. Executed notebooks retain their original
cell text as a historical record; use the current commands below to rerun them.
Transfer ZIPs are disposable after extraction. Input checksums are retained in
`outputs/text/input_provenance.json` so verification does not depend on those ZIPs.

Start with [the executed notebook](../../../notebooks/text/01_scam_intent_baseline.ipynb)
and [methods/results](../../../notebooks/text/RESULTS.md). The first milestone is
complete: dataset audit, fixed split IDs, train-only TF-IDF + logistic regression,
validation selection, saved preprocessing/classifier, and fusion OOF exports.
The [MiniLM notebook](../../../notebooks/text/03_minilm_scam_intent.ipynb) is now
executed, with [methods/results](../../../notebooks/text/MINILM_RESULTS.md), cached
chunked embeddings, saved preprocessing/classifier, OOF scores, and fixed-model
validation prefixes. Challenge and official-test comparisons await independent
review and lock of the [48-dialogue packet](../../../notebooks/text/challenge_v1/README.md).

Follow-up audit: [executed diagnostics](../../../notebooks/text/02_baseline_diagnostics.ipynb)
and [diagnostic interpretation](../../../notebooks/text/DIAGNOSTICS.md). These preserve
the primary artifacts, run fit-isolation checks and 20 label shuffles, and compare
strict/broad training-overlap removals on the same validation examples.

## Reproduce locally

From the repository root, using Python 3.12 and an isolated environment:

```powershell
uv venv "$HOME/.venvs/scam-text-baseline" --python 3.12
uv pip install --python "$HOME/.venvs/scam-text-baseline/Scripts/python.exe" -r src/text/requirements-baseline.txt
& "$HOME/.venvs/scam-text-baseline/Scripts/python.exe" -m src.text.baseline
```

This records a local runtime. The notebook defaults to requiring a Colab A100;
a local notebook smoke run requires `SCAM_REQUIRE_COLAB_A100=0`. The shared
`pyproject.toml` and `uv.lock` are unchanged. This isolated environment uses
`requirements-baseline.txt`; do not mix it with another project's environment.

## Reproduce on Colab A100 through the CLI

The verified run used Google's official `google-colab-cli` 0.6.0 from Ubuntu in
WSL. The official CLI supports Linux/macOS. From a Linux/WSL terminal in the
repository root, with CLI authentication already configured:

```bash
colab --auth=oauth2 new -s scam-text-a100 --gpu A100
colab --auth=oauth2 exec -s scam-text-a100 -f src/text/tools/colab_runtime_check.py --timeout 60
colab --auth=oauth2 install -s scam-text-a100 -r src/text/requirements-baseline.txt
colab --auth=oauth2 restart-kernel -s scam-text-a100
colab --auth=oauth2 exec -s scam-text-a100 -f notebooks/text/01_scam_intent_baseline.ipynb --timeout 600
colab --auth=oauth2 exec -s scam-text-a100 -f src/text/tools/collect_colab_provenance.py --timeout 60
colab --auth=oauth2 download -s scam-text-a100 /content/text_baseline_artifacts.zip outputs/text/colab_baseline_artifacts.zip
```

Initial CLI sign-in, if required, must be completed by the account holder.
Reuse an existing suitable session when available. The notebook can also use the
installed VS Code Colab extension: Select Kernel → Colab → New Colab Server →
GPU → A100. The notebook verifies actual hardware instead of relying on its
metadata accelerator hint. Scikit-learn fits on CPU within the A100 runtime.

Extract the archive to `outputs/text/baseline/`. The CLI writes an
`01_scam_intent_baseline_output.ipynb` beside the input notebook. Using the baseline
Python environment from the repository root:

```bash
python src/text/tools/finalize_colab_notebook.py --input notebooks/text/01_scam_intent_baseline_output.ipynb --output notebooks/text/01_scam_intent_baseline.ipynb
python -m src.text.tools.verify_baseline
```

The finalizer rejects cell errors and restores execution counts from actual
remote IPython history; CLI 0.6.0 omits those counts. That CLI can exit successfully
even when a notebook cell failed, so verify before accepting a run. After
retrieving and verifying artifacts, release the session:

```bash
colab --auth=oauth2 stop -s scam-text-a100
```

To update the notebook implementation, run `python src/text/tools/build_notebook.py`
and execute it again. The generator embeds the implementation for fresh remote
kernels; no workspace synchronization or GitHub push is needed. Installing then
restarting through the CLI avoids mixing already-loaded library versions.

Official sources: [CLI](https://github.com/googlecolab/google-colab-cli),
[VS Code extension](https://github.com/googlecolab/colab-vscode).

## Artifacts and inference

`outputs/text/baseline/` contains the downloaded Colab artifact set:

- `tfidf_logreg.joblib`: training-fitted TF-IDF vocabulary/IDF and classifier.
- `frozen_config.json`: settings, threshold, validation results, model hash.
- `runtime.json`, `notebook_execution_counts.json`: execution provenance.
- `split_manifest.csv`, `oof_folds.csv`: stable identities, inclusion flags, folds.
- `fusion_scores_train_validation.csv`: 1,086 OOF training and 192 validation scores.
- Audit tables, matching texts, validation predictions/errors, and PNG figures.

Data, models, and generated score files stay under ignored `data/` and `outputs/`.
Fixed split IDs, aggregate audit results, and manual similarity review are retained
under `notebooks/text/`. Do not change the fixed manifest silently.

```python
from src.text.baseline import score_transcripts
scores = score_transcripts("outputs/text/baseline/tfidf_logreg.joblib", [transcript])
```

Only load trusted joblib files. Scores are uncalibrated ranking values, with
larger values indicating more scam-like text in this dataset.

## Hugo's handoff contract

Join using `dialogue_id` in `notebooks/text/split_manifest.csv`, not CSV position.
IDs encode published source split, zero-based source row, and a raw-text SHA-256
prefix. Preserve `split`; exclude `included=False` rows from pairing. All 320
published test rows remain; there are no test predictions yet.

```text
dialogue_id,split,label,model,scam_score,fold
```

`label=1` means scam; `model=tfidf_logreg`; training `fold=0..4` identifies the
held-out fold; validation `fold=-1` means fitted on all eligible training rows.
Each OOF pipeline learns vocabulary, IDF, and classifier inside the other four
folds. Selected settings come from validation; OOF is for fusion fitting, not an
independent estimate of tuned-model quality. Similar templates can still cross
folds. Test scores and frozen turn-prefix exports follow the MiniLM milestone.

## Reproduce the follow-up diagnostics

Local run, from the root using the isolated baseline Python environment:

```bash
python -m src.text.diagnostics --output-dir outputs/text/diagnostics-local
```

For the remote run, first generate the notebook and frozen-reference input ZIP
using that environment:

```bash
python src/text/tools/build_diagnostic_notebook.py
```

Then from Linux/WSL in the repository root, with Colab CLI authentication ready:

```bash
colab --auth=oauth2 new -s scam-text-audit-a100 --gpu A100
colab --auth=oauth2 upload -s scam-text-audit-a100 outputs/text/diagnostic_inputs.zip /content/diagnostic_inputs.zip
colab --auth=oauth2 install -s scam-text-audit-a100 -r src/text/requirements-baseline.txt
colab --auth=oauth2 restart-kernel -s scam-text-audit-a100
colab --auth=oauth2 exec -s scam-text-audit-a100 -f notebooks/text/02_baseline_diagnostics.ipynb --timeout 600
colab --auth=oauth2 exec -s scam-text-audit-a100 -f src/text/tools/collect_diagnostic_provenance.py --timeout 60
colab --auth=oauth2 download -s scam-text-audit-a100 /content/text_diagnostic_artifacts.zip outputs/text/colab_diagnostic_artifacts.zip
```

Extract the archive into `outputs/text/diagnostics/`, then using the baseline
Python environment from the root:

```bash
python src/text/tools/finalize_colab_notebook.py --input notebooks/text/02_baseline_diagnostics_output.ipynb --output notebooks/text/02_baseline_diagnostics.ipynb --artifacts outputs/text/diagnostics
python -m src.text.tools.verify_diagnostics
python -m src.text.tools.verify_baseline
```

After verifying and recovering artifacts, release the runtime:

```bash
colab --auth=oauth2 stop -s scam-text-audit-a100
```

The diagnostic loader verifies the fixed manifest and downloads only the original
training CSV. It never scores official test examples. Overlap review is by the
coding agent; independent teammate review of a same-topic challenge set remains
outstanding. Sensitivity models are separate artifacts, not replacements for the
primary model or its fusion score exports.

## MiniLM and independent challenge review

Share only `notebooks/text/challenge_v1/review_packet.zip` with Justin and Hugo.
It includes a blind HTML reading copy, transcripts, separate blank label sheets,
instructions, and attestations. Keep `author_only/` separate until initial
independent review is complete. Do not score the draft set.

Prepare/reproduce MiniLM using the baseline environment to generate its notebook
and the frozen-reference archive. No local PyTorch installation is needed for
generation or artifact verification:

```bash
python src/text/tools/build_minilm_notebook.py
```

Then from Linux/WSL in the repository root:

```bash
colab --auth=oauth2 new -s scam-text-minilm-a100 --gpu A100
colab --auth=oauth2 upload -s scam-text-minilm-a100 outputs/text/minilm_inputs.zip /content/minilm_inputs.zip
colab --auth=oauth2 install -s scam-text-minilm-a100 -r src/text/requirements-minilm.txt
colab --auth=oauth2 restart-kernel -s scam-text-minilm-a100
colab --auth=oauth2 exec -s scam-text-minilm-a100 -f notebooks/text/03_minilm_scam_intent.ipynb --timeout 900
colab --auth=oauth2 exec -s scam-text-minilm-a100 -f src/text/tools/verify_minilm_checkpoint.py --timeout 120
colab --auth=oauth2 exec -s scam-text-minilm-a100 -f src/text/tools/collect_minilm_provenance.py --timeout 180
colab --auth=oauth2 download -s scam-text-minilm-a100 /content/text_minilm_artifacts.zip outputs/text/colab_minilm_artifacts.zip
```

Wait for each command to finish before issuing dependent commands. Use the
official CLI's current release; 0.7.5 completed artifact recovery after connection
losses with 0.6.0. The public model is loaded without authentication or optional
Xet transfers. The notebook uses the Colab image's CUDA-enabled PyTorch and records
its exact version (verified run: 2.11.0+cu130); use a matching runtime for exact
reproduction. The repository's shared torch/torchaudio setup is unchanged.

Extract the archive into `outputs/text/minilm/`, then use the baseline Python
environment locally:

```bash
python src/text/tools/finalize_colab_notebook.py --input notebooks/text/03_minilm_scam_intent_output.ipynb --output notebooks/text/03_minilm_scam_intent.ipynb --artifacts outputs/text/minilm
python -m src.text.tools.verify_minilm
python -m unittest src.text.tests.test_challenge_gate
```

Release the runtime after saving and verifying artifacts:

```bash
colab --auth=oauth2 stop -s scam-text-minilm-a100
```

MiniLM artifacts include `encoder/` (actual weights, tokenizer, and recipe),
`minilm_logreg.joblib`, `frozen_config.json`, `comparison_freeze.json`,
`embedding_cache/`, `wordpiece_lengths.csv`, validation metrics/errors/figures,
`fusion_scores_train_validation.csv`, and validation prefix scores/metrics.
The OOF rows use the exact TF-IDF folds. Keep each model's scores and each prefix
condition distinct; IDs and `split` remain the join keys.

The challenge-lock and final comparison commands are documented in
`notebooks/text/challenge_v1/README.md`. The challenge and official-test entry points
do no fitting. They verify frozen model/preprocessing hashes and enforce the
required review → challenge evaluation → official test order. No actual reviews
have been substituted with generated labels.
