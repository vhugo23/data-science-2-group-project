"""Verify a Colab CLI notebook result and restore real execution counts.

Run after downloading the artifacts and collect_colab_provenance.py's history.
"""
import argparse
import hashlib
import json
from pathlib import Path

import nbformat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("outputs/text/baseline"))
    args = parser.parse_args()
    notebook = nbformat.read(args.input, as_version=4)
    history = json.loads((args.artifacts / "notebook_execution_counts.json").read_text())
    counts = {row["source_sha256"]: row["execution_count"] for row in history}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        digest = hashlib.sha256(cell.source.strip().encode()).hexdigest()
        if digest not in counts:
            raise ValueError("Notebook cell not found in the actual Colab kernel history")
        errors = [o for o in cell.outputs if o.output_type == "error"]
        if errors:
            raise ValueError(f"Notebook execution failed: {errors}")
        cell.execution_count = counts[digest]
    source_file = notebook.metadata.get("source_file", "src/text/baseline.py")
    actual_source = hashlib.sha256(Path(source_file).read_text(encoding="utf-8").encode()).hexdigest()
    if actual_source != notebook.metadata.source_sha256:
        raise ValueError(f"Notebook implementation differs from {source_file}; regenerate and rerun")
    if "baseline_source_sha256" in notebook.metadata:
        actual_baseline = hashlib.sha256(Path("src/text/baseline.py").read_text(encoding="utf-8").encode()).hexdigest()
        if actual_baseline != notebook.metadata.baseline_source_sha256:
            raise ValueError("Embedded baseline implementation differs from baseline.py")
    runtime = json.loads((args.artifacts / "runtime.json").read_text())
    if not runtime["colab_runtime"] or "A100" not in runtime["gpu"]:
        raise ValueError("A Colab A100 execution was not verified")
    notebook.metadata["scam_intent_execution"] = {
        "colab_runtime": True, "gpu": runtime["gpu"],
        "execution_counts_source": "Actual remote IPython history, matched by SHA-256 of each cell source",
        "execution_date": "2026-10-29",
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, args.output)
    print(f"Verified {sum(c.cell_type == 'code' for c in notebook.cells)} executed cells; wrote {args.output}")


if __name__ == "__main__":
    main()
