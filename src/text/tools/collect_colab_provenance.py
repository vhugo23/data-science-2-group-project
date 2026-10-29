"""Run after the notebook using colab exec; recover execution counts from IPython.

Colab CLI 0.6.0 saves cell outputs but leaves notebook execution_count null.
Export real kernel history counts, rather than guessing or manufacturing them.
"""
import hashlib
import json
from pathlib import Path
import shutil

history = [
    {"execution_count": count,
     "source_sha256": hashlib.sha256(source.strip().encode()).hexdigest()}
    for session, count, source in get_ipython().history_manager.get_range(session=0, raw=True)
]
output = Path("/content/outputs/text/baseline")
(output / "notebook_execution_counts.json").write_text(json.dumps(history, indent=2) + "\n")
shutil.make_archive("/content/text_baseline_artifacts", "zip", output)
print(json.dumps(history, indent=2))
