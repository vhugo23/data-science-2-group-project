"""Remote saved-encoder round trip on one 1-, 2-, and 3-chunk development dialogue."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.text.baseline import write_json
from src.text.diagnostics import load_development
from src.text.minilm import FrozenMiniLM

root = Path("/content/outputs/text/minilm")
summary = json.loads((root / "minilm_summary.json").read_text())
freeze = json.loads((root / "comparison_freeze.json").read_text())
key = summary["embedding_cache"]["cache_key"]
with np.load(root / "embedding_cache" / f"{key}.npz", allow_pickle=False) as cache:
    ids = cache["ids"].tolist()
    original = cache["embeddings"].copy()
    chunks = cache["chunks"].copy()
positions = [int(np.flatnonzero(chunks == count)[0]) for count in [1, 2, 3]]
selected = [ids[i] for i in positions]
data = load_development(Path("/content/data/scam_dialogue"), Path("/content/outputs/text/minilm_inputs/fixed_manifest.csv")).set_index("dialogue_id")
encoder = FrozenMiniLM(root / "encoder")
assert encoder.fingerprint() == freeze["minilm_encoder_fingerprint"]
restored, _, cache = encoder.encode(selected, data.loc[selected, "dialogue"].tolist(), root / "checkpoint_verification_cache")
np.testing.assert_allclose(restored, original[positions], rtol=3e-5, atol=3e-6)
classifier = joblib.load(root / "minilm_logreg.joblib")["classifier"]
np.testing.assert_allclose(classifier.predict_proba(restored), classifier.predict_proba(original[positions]), rtol=3e-5, atol=3e-6)
result = {"saved_encoder_and_tokenizer_round_trip_verified": True, "chunk_counts_tested": [1, 2, 3],
          "dialogue_ids": selected, "max_absolute_embedding_difference": float(np.max(np.abs(restored - original[positions]))),
          "challenge_or_test_inputs_used": False, "encoder_fingerprint_matches_freeze": True}
write_json(root / "checkpoint_verification.json", result)
print(json.dumps(result, indent=2))
