"""Generate the portable, self-contained baseline notebook from the CLI implementation.

Run this after changing baseline.py. The notebook embeds the implementation so a
fresh Colab kernel does not need access to local VS Code workspace files.
"""
from pathlib import Path
import hashlib
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[3]
source = (ROOT / "src/text/baseline.py").read_text(encoding="utf-8")
requirements = (ROOT / "src/text/requirements-baseline.txt").read_text(encoding="utf-8")
source_hash = hashlib.sha256(source.encode()).hexdigest()
markdown = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [markdown("""# Scam intent: dataset audit and TF-IDF baseline

First text-track milestone. Input: **dialogue only**. Target: `label` (1 scam, 0 benign).
`type` is retained for audit/evaluation and never enters the classifier.

Run in VS Code with a Colab A100 kernel, or in Google Colab. This notebook is
self-contained; there is no need to clone/upload source modules. Runtime checks
fail unless a Colab A100 is actually present. Scikit-learn itself fits on CPU;
the A100 is reserved for the upcoming frozen MiniLM experiment.

The original 1,280/320 split is preserved, with 192 validation rows selected only
from published training rows (label stratification, seed 42). All 1,600 assignments
remain in the manifest. Two duplicate training rows are excluded from fitting,
leaving **1,086 train / 192 validation / 320 test**. Non-exact template matches are
retained and reported as a limitation; this is not a template-disjoint benchmark.

**Test predictions and test metrics are deliberately deferred** until the MiniLM
comparison settings have been frozen. This notebook exports training OOF and
validation scores only. No test outcomes are used for model or threshold choice.

Sources: [dataset card](https://huggingface.co/datasets/BothBosu/scam-dialogue/blob/321b961b5ae353e19ed479b960658dcd223d5e06/README.md),
[MiniLM card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
These synthetic dialogues do not establish real-world scam-call reliability.
"""), code('''import os
from pathlib import Path
import subprocess
import sys

# Keep the default for the requested remote run. Set the environment variable to 0
# only for an explicitly labelled local smoke run.
REQUIRE_COLAB_A100 = os.environ.get("SCAM_REQUIRE_COLAB_A100", "1") == "1"
try:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
    ).strip()
except (OSError, subprocess.SubprocessError):
    gpu = "unavailable"
is_colab = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"))
print({"Colab runtime": is_colab, "GPU": gpu, "Python": sys.version})
if REQUIRE_COLAB_A100:
    assert is_colab and "A100" in gpu, "Connect a Colab A100 kernel before running."
'''), markdown("""## Dependencies and paths

Pinned text-baseline dependencies are isolated from teammates' shared setup.
Installation happens before importing numerical libraries. If this kernel has
already imported another NumPy/pandas version, restart it before Run All.
Colab files are ephemeral: download the output archive at the end.
"""), code(f'''requirements = {requirements!r}
if is_colab:
    import importlib.metadata
    requested = [line for line in requirements.splitlines() if line and not line.startswith("#")]
    mismatches = []
    for item in requested:
        name, expected = item.split("==")
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            mismatches.append(item)
    if mismatches:
        loaded_before_install = [name for name in ("numpy", "pandas", "scipy", "sklearn", "matplotlib") if name in sys.modules]
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *requested], check=True)
        if loaded_before_install:
            raise RuntimeError("Dependencies installed. Restart the kernel and Run All before importing numerical libraries.")

ROOT = Path.cwd()
if not is_colab:
    ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()), ROOT)
DATA_DIR = ROOT / "data/scam_dialogue"
OUTPUT_DIR = ROOT / "outputs/text/baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Artifacts:", OUTPUT_DIR)
'''), markdown(f"""## Implementation and runtime provenance

Embedded from `src/text/baseline.py`; source SHA-256: `{source_hash}`.
Regenerate using `python src/text/tools/build_notebook.py` after code changes.
""")]

# Keep definitions in readable sections. No __main__ invocation inside the notebook.
body = source[source.index("from __future__"):source.index("\ndef main()")]
sections = ["\ndef load_data", "\ndef audit", "\ndef metrics", "\ndef train_baseline", "\ndef export_oof"]
starts = [0] + [body.index(marker) for marker in sections]
for start, end in zip(starts, starts[1:] + [len(body)]):
    cells.append(code(body[start:end].strip()))

cells.extend([
    code('''runtime = runtime_info(require_a100=REQUIRE_COLAB_A100)
write_json(OUTPUT_DIR / "runtime.json", runtime)
display(runtime)
'''), markdown("""## Load, fix splits, and audit

Source files and raw text hashes are verified against a pinned Hugging Face commit.
IDs include the original file split, zero-based row, and raw-text hash. Exclusions
retain their original ID and assignment. Exact duplicates prioritize test, then
validation, then training; no test rows are removed. No similarity-only exclusions.

Near-match search uses stateless character 5-gram hashing and cosine similarity
(threshold 0.90). This is audit-only and learns no vocabulary/IDF from held-out data.
Every candidate's full text is saved for review; high similarity is a flag, not
proof of a repeated source example. Word lengths below are not wordpiece lengths.
"""), code('''data = load_data(DATA_DIR, OUTPUT_DIR)
summary = audit(data, OUTPUT_DIR)
display(summary)
display(pd.read_csv(OUTPUT_DIR / "class_type_counts.csv"))
display(data.loc[~data.included, ["dialogue_id", "split", "label", "exclusion_reason"]])
from IPython.display import Image, display
display(Image(filename=str(OUTPUT_DIR / "dialogue_lengths.png")))
'''), code('''candidates = pd.read_csv(OUTPUT_DIR / "near_duplicate_candidates.csv")
display(candidates)
# The full dialogue pairs are available in similarity_review_text.json.
print("All candidate pairs:", len(candidates))
print("Human review conclusions are in notebooks/text/RESULTS.md and similarity_review.csv.")
'''), markdown("""## Train and select using validation F1

Six settings: word unigrams or unigrams+bigrams, C in {0.1, 1, 10}, min_df=2,
sublinear term frequency. Each full Pipeline fits vocabulary, IDF, and logistic
regression on eligible training rows only. First choose settings by validation F1
at 0.5 (ties: unigrams, then smaller C), then choose the threshold by validation F1
(ties closest to 0.5). Save the training-fitted Pipeline and frozen configuration.
"""), code('''model, config = train_baseline(data, OUTPUT_DIR)
display(pd.read_csv(OUTPUT_DIR / "validation_search.csv"))
display(config)
display(Image(filename=str(OUTPUT_DIR / "validation_confusion_matrix.png")))
display(pd.read_csv(OUTPUT_DIR / "validation_by_type.csv"))
errors = pd.read_csv(OUTPUT_DIR / "validation_errors.csv")
if errors.empty:
    print("No validation false positives or false negatives; no error examples to interpret.")
else:
    display(errors[["dialogue_id", "type", "label", "scam_score", "error_kind", "dialogue"]])
'''), markdown("""## Fusion handoff: OOF training scores and validation scores

Five stratified group folds (seed 42, group=normalized text hash). For each held-out
fold, refit the full Pipeline on the other four folds. OOF uses the settings selected
on validation; it is fusion input, not an independent estimate of tuned-model quality.
Each included training dialogue is scored once by a classifier that did not fit it.
Excluded rows are absent. Validation uses the classifier fitted on all eligible
training rows. Folds are 0–4 for training, -1 for validation. Similar template variants
can still cross folds. Join through `dialogue_id`, never by row order.
"""), code('''scores = export_oof(data, model, OUTPUT_DIR)
display(scores.groupby(["split", "fold"]).agg(n=("dialogue_id", "size")))
display(scores.head())
assert len(scores) == int((data.included & data.split.ne("test")).sum())
assert len(data[data.split.eq("test")]) == 320
assert len(data[data.split.eq("test") & data.included]) == 320
assert not set(data.loc[data.included & data.split.eq("train"), "normalized_sha256"]) & set(
    data.loc[data.included & data.split.ne("train"), "normalized_sha256"])

# Confirm that the serialized preprocessing/classifier reproduces validation scores.
validation = data[data.included & data.split.eq("validation")]
restored = score_transcripts(OUTPUT_DIR / "tfidf_logreg.joblib", validation.dialogue.tolist())
np.testing.assert_allclose(restored, model.predict_proba(validation.dialogue)[:, 1])
print("Verified: split membership, OOF coverage, exact-duplicate separation, model round trip.")
'''), markdown("""## Interpretation and next milestone

Scam precision/recall/F1 use label 1 as positive. In benign-only types, report
false-positive rate and accuracy; scam recall/F1 are undefined. The validation set
is used for tuning, so its result is not an unbiased final estimate. Scores are
uncalibrated rankings, not real-world fraud probabilities. F1 >0.85 is exploratory.

Next: measure actual MiniLM wordpieces; split beyond 254 content tokens (reserve
special tokens within 256), embed chunks with the frozen encoder, average and cache
dialogue vectors, then fit/select logistic regression. Freeze both branches before
one final test evaluation. Then export test scores and frozen 25/50/75/100% turn
prefix scores. Early prefixes may end before a scam request appears.
"""), code('''# Retrieve this archive with the Colab CLI or the notebook Files panel.
import shutil
assert (OUTPUT_DIR / "frozen_config.json").exists(), "Training must complete before export."
assert (OUTPUT_DIR / "fusion_scores_train_validation.csv").exists(), "OOF export must complete before archiving."
archive = shutil.make_archive(str(ROOT / "text_baseline_artifacts"), "zip", OUTPUT_DIR)
print("Artifact archive:", archive)
''')
])

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"name": "01_scam_intent_baseline.ipynb", "provenance": []},
    "accelerator": "GPU", "gpuClass": "premium", "source_sha256": source_hash,
})
path = ROOT / "notebooks/text/01_scam_intent_baseline.ipynb"
path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, path)
print(path)
