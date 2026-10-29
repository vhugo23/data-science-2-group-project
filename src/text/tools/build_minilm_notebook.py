"""Build the portable Colab MiniLM notebook and frozen baseline reference ZIP."""
import hashlib
import json
from pathlib import Path
import zipfile

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[3]
files = {name: (ROOT / f"src/text/{name}.py").read_text(encoding="utf-8") for name in ["baseline", "diagnostics", "minilm"]}
requirements = (ROOT / "src/text/requirements-minilm.txt").read_text()
reference = ROOT / "outputs/text/baseline"
archive = ROOT / "outputs/text/minilm_inputs.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
    for name in ["tfidf_logreg.joblib", "frozen_config.json", "oof_folds.csv", "fusion_scores_train_validation.csv"]:
        package.write(reference / name, f"reference/{name}")
    package.write(ROOT / "notebooks/text/split_manifest.csv", "fixed_manifest.csv")
archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
provenance_path = ROOT / "outputs/text/input_provenance.json"
provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
with zipfile.ZipFile(archive) as package:
    provenance[archive.name] = {"archive_sha256": archive_sha, "entries_sha256": {
        item.filename: hashlib.sha256(package.read(item)).hexdigest()
        for item in package.infolist() if not item.is_dir()}}
provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="")
primary = nbf.read(ROOT / "notebooks/text/01_scam_intent_baseline.ipynb", as_version=4)
runtime_cell = primary.cells[1].source + '''
# Public model: avoid optional transfer-backend and credential discovery paths.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
'''
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
setup = primary.cells[3].source
start, end = setup.index("requirements = "), setup.index("\nif is_colab:")
setup = "requirements = " + repr(requirements) + setup[end:]
setup = setup.replace('OUTPUT_DIR = ROOT / "outputs/text/baseline"', 'OUTPUT_DIR = ROOT / "outputs/text/minilm"')
cells = [md("""# Frozen MiniLM scam-intent classifier

Original fixed training/validation split, with the same exact-duplicate exclusions
as the TF-IDF baseline. MiniLM remains frozen; only logistic regression is fitted.
Select C and threshold using the original 192 validation examples. Challenge data
and official test text are not loaded or scored in this notebook.

Run on a Colab A100 using the CLI. Upload `outputs/text/minilm_inputs.zip` to
`/content/minilm_inputs.zip` first. The reference archive contains the frozen
TF-IDF model and split/fold metadata, **no challenge transcripts**. The encoder
actually computes on CUDA; logistic regression uses CPU.

Tokenizer/model revision is pinned. Inputs are sliced at 254 content wordpieces,
leaving two special tokens within a 256-token sequence. Encode each chunk with
attention-mask mean pooling and L2 normalization; average chunk vectors equally
and L2-normalize the dialogue mean. No chunks cross dialogue boundaries.

This follows the [official model-card mean-pooling recipe](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
extended with a fixed chunk-averaging policy. Chunking is not selected using
challenge performance. Embedding cache keys include ordered IDs, raw text hashes,
the model revision, and pooling/chunking recipe. Cache reuse is verified.
"""), code(runtime_cell), md("## Dependencies and remote paths"), code(setup),
md("## Frozen references and implementation"), code(f'''import hashlib
import zipfile

inputs_zip = Path("/content/minilm_inputs.zip") if is_colab else ROOT / "outputs/text/minilm_inputs.zip"
assert hashlib.sha256(inputs_zip.read_bytes()).hexdigest() == {archive_sha!r}
INPUT_DIR = ROOT / "outputs/text/minilm_inputs"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(inputs_zip) as package:
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in package.namelist())
    package.extractall(INPUT_DIR)

CODE_DIR = ROOT / "outputs/text/minilm_code"
MODULE_DIR = CODE_DIR / "src/text"
MODULE_DIR.mkdir(parents=True, exist_ok=True)
(MODULE_DIR.parent / "__init__.py").write_text("")
(MODULE_DIR / "__init__.py").write_text("")
sources = {files!r}
for name, source in sources.items():
    (MODULE_DIR / (name + ".py")).write_text(source, encoding="utf-8")
sys.path.insert(0, str(CODE_DIR))
from src.text.minilm import run_minilm
import pandas as pd
from IPython.display import Image, display
'''), md("""## Encode, cache, and fit validation-selected logistic regression

The frozen encoder uses no dataset fitting, including for OOF examples. All learned
classifier fitting stays within the training partition and within the other four
folds for OOF predictions. Reuse the TF-IDF fold IDs exactly. Save the encoder,
tokenizer, pooling recipe, classifier, thresholds, runtime versions, and a model
comparison freeze record before prefix evaluation. No optimizer is used for MiniLM.
"""), code('''summary = run_minilm(DATA_DIR, INPUT_DIR / "fixed_manifest.csv", INPUT_DIR / "reference", OUTPUT_DIR,
                     require_a100=REQUIRE_COLAB_A100)
display(summary)
display(pd.read_csv(OUTPUT_DIR / "wordpiece_lengths.csv").groupby("split").agg(
    n=("dialogue_id", "size"), median_wordpieces=("content_wordpieces", "median"),
    max_wordpieces=("content_wordpieces", "max"), chunked_dialogues=("would_truncate_at_256", "sum")))
display(pd.read_csv(OUTPUT_DIR / "validation_search.csv"))
display(pd.read_csv(OUTPUT_DIR / "validation_by_type.csv"))
display(Image(filename=str(OUTPUT_DIR / "validation_confusion_matrix.png")))
errors = pd.read_csv(OUTPUT_DIR / "validation_errors.csv")
if errors.empty:
    print("No validation errors to interpret.")
else:
    display(errors)
'''), md("""## Frozen incomplete-information evaluation on original validation

Score the first 25/50/75/100% of speaker turns using unchanged classifiers and
thresholds. An early prefix may end before the relevant scam request. These
validation robustness results are not used to retune either model. Challenge-set
prefixes are not scored before independent review and locking.
"""), code('''display(pd.read_csv(OUTPUT_DIR / "validation_prefix_metrics.csv"))
display(Image(filename=str(OUTPUT_DIR / "validation_prefix_performance.png")))
display(pd.read_csv(OUTPUT_DIR / "fusion_scores_train_validation.csv").groupby(["split", "fold"]).size())
print("Challenge scoring is pending independent labels and lock. Official test comparison follows that evaluation.")
assert not summary["challenge_scored"] and not summary["test_evaluated"]
'''), md("""## Recover artifacts

Download the archive, verify all notebook cells and saved hashes, and preserve the
embedding cache for repeated experiments. The independent review packet lives
locally; this runtime has not received its draft transcripts or author labels.
"""), code('''import shutil
assert (OUTPUT_DIR / "minilm_summary.json").exists()
assert (OUTPUT_DIR / "comparison_freeze.json").exists()
archive = shutil.make_archive(str(ROOT / "text_minilm_artifacts"), "zip", OUTPUT_DIR)
print("Artifact archive:", archive)
''')]
notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "accelerator": "GPU", "gpuClass": "premium",
    "source_file": "src/text/minilm.py", "source_sha256": hashlib.sha256(files["minilm"].encode()).hexdigest(),
    "baseline_source_sha256": hashlib.sha256(files["baseline"].encode()).hexdigest(),
    "input_archive_sha256": archive_sha,
    "colab": {"name": "03_minilm_scam_intent.ipynb", "provenance": []},
})
path = ROOT / "notebooks/text/03_minilm_scam_intent.ipynb"
nbf.write(notebook, path)
print(path)
print(archive)
