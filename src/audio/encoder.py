"""Frozen wav2vec 2.0 embeddings with an on-disk cache."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from .features import SETTINGS, degrade_audio, process_audio


def load_encoder(device):
    """Return (feature extractor, frozen encoder) at the pinned revision."""
    name, revision = SETTINGS["encoder"], SETTINGS["encoder_revision"]
    processor = AutoFeatureExtractor.from_pretrained(name, revision=revision)
    encoder = AutoModel.from_pretrained(name, revision=revision).to(device).eval()
    encoder.requires_grad_(False)
    if processor.sampling_rate != SETTINGS["sample_rate"]:
        raise ValueError(f"{name} expects {processor.sampling_rate} Hz, settings use {SETTINGS['sample_rate']} Hz")
    return processor, encoder


def cache_path(output_dir, name):
    """Cache file whose name records the encoder and preprocessing settings."""
    stem = (f"{SETTINGS['encoder'].split('/')[-1]}_{SETTINGS['sample_rate'] // 1000}k_"
            f"{SETTINGS['window_seconds']}s_trim{SETTINGS['trim_top_db']}")
    return Path(output_dir) / f"{stem}_{name}.npz"


def encoder_features(frame, processor, encoder, device, cache=None, condition="clean", seed=None):
    """Time-mean pooled last hidden states, one clip at a time to limit GPU memory.

    frame needs sample_id and path columns. A cache is reused only when its ids
    match exactly; degraded conditions should not pass a cache.
    """
    ids = frame["sample_id"].to_numpy(dtype=str)
    if cache is not None and Path(cache).exists():
        with np.load(cache) as saved:
            if np.array_equal(saved["sample_ids"], ids):
                return saved["features"]
    rng = np.random.default_rng(seed)
    values = []
    for path in frame["path"]:
        audio = degrade_audio(process_audio(path), condition, rng)
        inputs = processor(audio, sampling_rate=SETTINGS["sample_rate"], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = encoder(**inputs).last_hidden_state
        values.append(hidden.mean(dim=1).squeeze(0).cpu().numpy())
    features = np.stack(values)
    if cache is not None:
        np.savez_compressed(cache, sample_ids=ids, features=features)
    return features
