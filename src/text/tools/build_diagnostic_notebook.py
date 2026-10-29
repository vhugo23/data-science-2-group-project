"""Build a separate diagnostic notebook and its frozen-reference input archive."""
import hashlib
import json
from pathlib import Path
import zipfile

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[3]
baseline = (ROOT / "src/text/baseline.py").read_text(encoding="utf-8")
diagnostics = (ROOT / "src/text/diagnostics.py").read_text(encoding="utf-8")
reference = ROOT / "outputs/text/baseline"
archive = ROOT / "outputs/text/diagnostic_inputs.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
    for name in ["tfidf_logreg.joblib", "frozen_config.json", "split_manifest.csv", "oof_folds.csv", "fusion_scores_train_validation.csv"]:
        package.write(reference / name, f"reference/{name}")
    package.write(ROOT / "notebooks/text/split_manifest.csv", "fixed_manifest.csv")
    package.write(ROOT / "notebooks/text/audit/similarity_review.csv", "review.csv")
archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
provenance_path = ROOT / "outputs/text/input_provenance.json"
provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
with zipfile.ZipFile(archive) as package:
    provenance[archive.name] = {"archive_sha256": archive_sha, "entries_sha256": {
        item.filename: hashlib.sha256(package.read(item)).hexdigest()
        for item in package.infolist() if not item.is_dir()}}
provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="")
primary = nbf.read(ROOT / "notebooks/text/01_scam_intent_baseline.ipynb", as_version=4)
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
setup = primary.cells[3].source.replace('OUTPUT_DIR = ROOT / "outputs/text/baseline"', 'OUTPUT_DIR = ROOT / "outputs/text/diagnostics"')
cells = [md("""# Why is validation perfect? Baseline diagnostics

This supplementary experiment preserves the primary baseline and its fixed split.
It audits fit inputs, reconstructs training-only TF-IDF vocabulary/IDF, reproduces
the six model-selection fits and five OOF fits with fit-call tracing, runs 20
shuffled-label controls, and measures sensitivity to training/validation overlap.

The original settings and threshold are fixed. All 192 validation dialogues remain
in every experiment. **Official test text is neither loaded nor scored here.**

Two sensitivity policies are declared before fitting: remove four previously
reviewed near-verbatim training counterparts; and a broader stress test removing
all training counterparts to validation flagged at cosine ≥0.90. The latter is a
threshold stress test, not a claim that every flagged pair is a duplicate.

Run on a Colab A100. Upload `outputs/text/diagnostic_inputs.zip` to
`/content/diagnostic_inputs.zip` using the documented CLI workflow first. Source
implementation is embedded. This notebook uses synthetic data; it cannot establish
same-topic intent understanding or real-world reliability. An independently
reviewed challenge set has not yet been constructed/evaluated.
"""), code(primary.cells[1].source), md("## Pinned dependencies"), code(setup),
md("## Frozen reference and implementation"), code(f'''import hashlib
import zipfile

inputs_zip = Path("/content/diagnostic_inputs.zip") if is_colab else ROOT / "outputs/text/diagnostic_inputs.zip"
assert hashlib.sha256(inputs_zip.read_bytes()).hexdigest() == {archive_sha!r}, "Reference input archive changed"
INPUT_DIR = ROOT / "outputs/text/diagnostic_inputs"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(inputs_zip) as package:
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in package.namelist())
    package.extractall(INPUT_DIR)

# Local files are not automatically visible to a remote VS Code notebook kernel.
CODE_DIR = ROOT / "outputs/text/diagnostic_code"
MODULE_DIR = CODE_DIR / "src/text"
MODULE_DIR.mkdir(parents=True, exist_ok=True)
(MODULE_DIR.parent / "__init__.py").write_text("")
(MODULE_DIR / "__init__.py").write_text("")
(MODULE_DIR / "baseline.py").write_text({baseline!r}, encoding="utf-8")
(MODULE_DIR / "diagnostics.py").write_text({diagnostics!r}, encoding="utf-8")
sys.path.insert(0, str(CODE_DIR))
from src.text.baseline import runtime_info, write_json
from src.text.diagnostics import load_development, run_diagnostics
import pandas as pd
from IPython.display import Image, display

write_json(OUTPUT_DIR / "runtime.json", runtime_info(require_a100=REQUIRE_COLAB_A100))
display(runtime_info(require_a100=REQUIRE_COLAB_A100))
'''), md("""## Run the fixed audit protocol

Training data only is downloaded from the pinned revision. The saved manifest
governs split membership. Fit tracing checks that inputs are the original dialogue
strings and correctly aligned labels, and that every OOF TF-IDF/classifier fit
excludes its held-out fold. Primary validation and OOF scores must reproduce.

Label controls permute training targets for seeds 0–19, refit the complete Pipeline,
and evaluate against unchanged true validation labels. No settings or thresholds
are selected using control results. Chance expected balanced accuracy/AUC is 0.5;
F1 does not have a universal chance baseline. Wide variation can occur because
the synthetic validation dialogues share topics and templates.
"""), code('''development = load_development(DATA_DIR, INPUT_DIR / "fixed_manifest.csv")
review = pd.read_csv(INPUT_DIR / "review.csv")
summary = run_diagnostics(development, review, INPUT_DIR / "reference", OUTPUT_DIR)
display(summary["implementation_checks"])
display(pd.read_csv(OUTPUT_DIR / "label_shuffle_metrics.csv"))
display(pd.DataFrame(summary["label_shuffle"]["summary"]))
display(Image(filename=str(OUTPUT_DIR / "label_shuffle_controls.png")))
'''), md("""## Overlap sensitivity

The four strict removals were identified by prior text review: same participants,
scenario, turn structure, and content, with only small phrase changes. Review is
by the coding agent; it is not an independent teammate label review. Shared
templates with uncertain identity remain in the strict experiment. The broad
experiment removes every flagged training/validation counterpart regardless of
identity. No validation rows, official test rows, or original artifacts are changed.
"""), code('''display(pd.read_csv(OUTPUT_DIR / "overlap_decisions.csv"))
display(pd.read_csv(OUTPUT_DIR / "overlap_sensitivity_metrics.csv"))
display(pd.read_csv(OUTPUT_DIR / "overlap_sensitivity_by_type.csv"))
display(Image(filename=str(OUTPUT_DIR / "overlap_sensitivity_confusion_matrices.png")))
display(summary["review_scope"])
print(summary["limits"])
assert summary["test_evaluated"] is False
assert summary["review_scope"]["original_reference_artifacts_unchanged"]
'''), md("""## Interpretation and handoff

Passing implementation and shuffled-label controls reduces concern about a coding
leak. Passing overlap sensitivity shows whether these specific removals affect
this validation result. Neither demonstrates intent discrimination within a topic.
Next, preregister a fixed supplemental set of legitimate refunds/support and
fraudulent delivery/insurance calls; collect independent teammate labels before
scoring. Keep original benchmark results and diagnostics separate. Hugo should
account for topic/template shortcuts when interpreting fusion scores.

Save the artifacts before releasing the runtime. CLI 0.6.0 can continue after cell
errors; verify saved outputs and match cell source hashes to actual kernel history.
"""), code('''import json
import shutil

assert (OUTPUT_DIR / "diagnostic_summary.json").exists()
assert (OUTPUT_DIR / "label_shuffle_metrics.csv").exists()
assert (OUTPUT_DIR / "overlap_sensitivity_metrics.csv").exists()
archive = shutil.make_archive(str(ROOT / "text_diagnostic_artifacts"), "zip", OUTPUT_DIR)
print("Diagnostic archive:", archive)
''')]
notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "accelerator": "GPU", "gpuClass": "premium",
    "source_sha256": hashlib.sha256(diagnostics.encode()).hexdigest(),
    "source_file": "src/text/diagnostics.py",
    "baseline_source_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
    "input_archive_sha256": archive_sha,
    "colab": {"name": "02_baseline_diagnostics.ipynb", "provenance": []},
})
path = ROOT / "notebooks/text/02_baseline_diagnostics.ipynb"
nbf.write(notebook, path)
print(path)
print(archive)
