"""Small remote probe, invoked with colab exec -f before running the notebook."""
import json
import os
import platform
import subprocess

gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
).strip()
print(json.dumps({"python": platform.python_version(), "gpu": gpu,
                  "colab": bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_BACKEND_VERSION"))}, indent=2))
assert "A100" in gpu, gpu
