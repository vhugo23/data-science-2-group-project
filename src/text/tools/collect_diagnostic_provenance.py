"""Recover actual Colab kernel execution counts and package diagnostics."""
import hashlib
import json
from pathlib import Path
import shutil

history = [{"execution_count": count, "source_sha256": hashlib.sha256(source.strip().encode()).hexdigest()}
           for session, count, source in get_ipython().history_manager.get_range(session=0, raw=True)]
output = Path("/content/outputs/text/diagnostics")
(output / "notebook_execution_counts.json").write_text(json.dumps(history, indent=2) + "\n")
shutil.make_archive("/content/text_diagnostic_artifacts", "zip", output)
print(f"Captured {len(history)} kernel history entries; packaged diagnostic artifacts.")
