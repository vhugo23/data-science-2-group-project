"""ASVspoof protocol loading, waveform preprocessing, degradations, and MFCC features.

Every setting that changes features lives in SETTINGS, so the notebook, the
embedding-cache name, and the exported settings.json cannot drift apart.
"""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

SETTINGS = {
    "sample_rate": 16000,
    "window_seconds": 4,
    "trim_top_db": 30,
    "crop": "start",
    "padding": "right_zero",
    "normalization": "peak",
    "n_mfcc": 20,
    "n_fft": 400,
    "hop_length": 160,
    "n_mels": 40,
    "encoder": "facebook/wav2vec2-base",
    "encoder_revision": "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
    "pooling": "time_mean",
}
# Official ASVspoof 2019 LA split -> shared project split name.
SPLIT_NAMES = {"train": "train", "dev": "validation", "eval": "test"}
LABELS = {"bonafide": 0, "spoof": 1}
CONDITIONS = ("clean", "noise", "mulaw")
_PROTOCOL_SUFFIX = {"train": "trn", "dev": "trl", "eval": "trl"}


def load_protocol(la_dir, split):
    """Return (protocol rows with path and label columns, number of unlisted FLAC files)."""
    la_dir = Path(la_dir)
    protocol = la_dir / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{split}.{_PROTOCOL_SUFFIX[split]}.txt"
    frame = pd.read_csv(protocol, sep=r"\s+", names=["speaker", "sample_id", "unused", "attack", "key"])
    audio_dir = la_dir / f"ASVspoof2019_LA_{split}" / "flac"
    frame["path"] = [audio_dir / f"{name}.flac" for name in frame["sample_id"]]
    frame["label"] = frame["key"].map(LABELS)
    if frame["label"].isna().any():
        raise ValueError(f"{protocol.name} has keys other than {sorted(LABELS)}")
    if not frame["sample_id"].is_unique:
        raise ValueError(f"{protocol.name} lists a sample_id more than once")
    missing = [path for path in frame["path"] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} listed clips are missing, e.g. {missing[0]}")
    extra = len(set(audio_dir.glob("*.flac")) - set(frame["path"]))
    return frame, extra


def prepare_waveform(audio, trim=True):
    """Trim edge silence, peak-normalize, then crop or right-pad with zeros to the fixed window."""
    if trim:
        audio, _ = librosa.effects.trim(audio, top_db=SETTINGS["trim_top_db"])
    audio = librosa.util.normalize(audio)
    return librosa.util.fix_length(audio, size=SETTINGS["sample_rate"] * SETTINGS["window_seconds"])


def process_audio(path, trim=True):
    """Load a clip as 16 kHz mono and prepare it."""
    audio, _ = librosa.load(path, sr=SETTINGS["sample_rate"], mono=True)
    return prepare_waveform(audio, trim=trim)


def degrade_audio(audio, condition, rng):
    """Apply a test-time degradation: 'clean', 'noise' (white noise at 10 dB SNR), or 'mulaw'."""
    if condition == "clean":
        return audio
    if condition == "noise":
        # SNR is measured over the fixed window, padding included.
        noise = rng.standard_normal(len(audio))
        noise *= np.sqrt(np.mean(audio ** 2) / (10 * np.mean(noise ** 2)))
        return (audio + noise).astype(np.float32)
    if condition == "mulaw":
        # Simulated 8-bit mu-law quantization; not a full telephone codec.
        audio = np.clip(audio, -1, 1)
        encoded = np.sign(audio) * np.log1p(255 * np.abs(audio)) / np.log1p(255)
        encoded = np.round((encoded + 1) * 127.5) / 127.5 - 1
        return (np.sign(encoded) * np.expm1(np.abs(encoded) * np.log1p(255)) / 255).astype(np.float32)
    raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")


def mfcc_matrix(audio):
    """MFCCs with 25 ms windows and 10 ms hops: shape (n_mfcc, frames)."""
    return librosa.feature.mfcc(
        y=audio, sr=SETTINGS["sample_rate"], n_mfcc=SETTINGS["n_mfcc"],
        n_fft=SETTINGS["n_fft"], hop_length=SETTINGS["hop_length"], n_mels=SETTINGS["n_mels"],
    )


def summarize_mfcc(audio):
    """Mean and standard deviation of each coefficient over time (2 * n_mfcc features)."""
    matrix = mfcc_matrix(audio)
    return np.concatenate([matrix.mean(axis=1), matrix.std(axis=1)])


def mfcc_features(paths, trim=True, condition="clean", seed=None):
    """Feature matrix for clip paths; the seed makes degradations reproducible."""
    rng = np.random.default_rng(seed)
    return np.stack([summarize_mfcc(degrade_audio(process_audio(path, trim=trim), condition, rng))
                     for path in paths])
