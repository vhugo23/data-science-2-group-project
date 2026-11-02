"""Audio preprocessing and feature tests on synthetic signals; no dataset needed."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio.features import (
    SETTINGS,
    degrade_audio,
    load_protocol,
    mfcc_features,
    prepare_waveform,
    summarize_mfcc,
)

RATE = SETTINGS["sample_rate"]
WINDOW = RATE * SETTINGS["window_seconds"]


def tone(seconds, amplitude=0.3, frequency=440.0):
    time = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


class PrepareWaveformTests(unittest.TestCase):
    def test_short_clip_is_trimmed_normalized_and_right_padded(self):
        silence = np.zeros(RATE // 2, dtype=np.float32)
        audio = prepare_waveform(np.concatenate([silence, tone(1.0), silence]))
        self.assertEqual(len(audio), WINDOW)
        self.assertAlmostEqual(float(np.abs(audio).max()), 1.0, places=5)
        self.assertGreater(np.abs(audio[:RATE // 10]).max(), 0.1)  # leading silence removed
        self.assertTrue((audio[int(1.2 * RATE):] == 0).all())  # zero padding on the right

    def test_untrimmed_clip_keeps_leading_silence(self):
        silence = np.zeros(RATE // 2, dtype=np.float32)
        audio = prepare_waveform(np.concatenate([silence, tone(1.0)]), trim=False)
        self.assertTrue((audio[:RATE // 4] == 0).all())

    def test_long_clip_keeps_the_start(self):
        audio = np.linspace(-1, 1, 6 * RATE, dtype=np.float32)
        prepared = prepare_waveform(audio, trim=False)
        self.assertEqual(len(prepared), WINDOW)
        np.testing.assert_allclose(prepared, audio[:WINDOW])


class DegradeAudioTests(unittest.TestCase):
    def setUp(self):
        self.audio = prepare_waveform(tone(4.0), trim=False)

    def test_clean_is_unchanged(self):
        self.assertIs(degrade_audio(self.audio, "clean", np.random.default_rng(0)), self.audio)

    def test_noise_is_ten_db_snr_and_seeded(self):
        noisy = degrade_audio(self.audio, "noise", np.random.default_rng(0))
        snr = 10 * np.log10(np.mean(self.audio ** 2) / np.mean((noisy - self.audio) ** 2))
        self.assertAlmostEqual(snr, 10.0, places=3)
        again = degrade_audio(self.audio, "noise", np.random.default_rng(0))
        np.testing.assert_array_equal(noisy, again)

    def test_mulaw_is_bounded_quantized_and_close(self):
        coded = degrade_audio(self.audio, "mulaw", np.random.default_rng(0))
        self.assertLessEqual(np.abs(coded).max(), 1.0 + 1e-6)
        self.assertLessEqual(len(np.unique(coded)), 256)
        self.assertLess(np.abs(coded - self.audio).max(), 0.05)

    def test_unknown_condition_raises(self):
        with self.assertRaisesRegex(ValueError, "condition"):
            degrade_audio(self.audio, "reverb", np.random.default_rng(0))


class MfccTests(unittest.TestCase):
    def test_summary_has_mean_and_std_per_coefficient(self):
        features = summarize_mfcc(prepare_waveform(tone(2.0)))
        self.assertEqual(features.shape, (2 * SETTINGS["n_mfcc"],))
        self.assertTrue(np.isfinite(features).all())


class ProtocolTests(unittest.TestCase):
    def write_split(self, root, rows, extra_files=()):
        protocols = root / "ASVspoof2019_LA_cm_protocols"
        audio_dir = root / "ASVspoof2019_LA_train" / "flac"
        protocols.mkdir(parents=True)
        audio_dir.mkdir(parents=True)
        lines = [f"LA_0079 {sample_id} - {attack} {key}" for sample_id, attack, key in rows]
        (protocols / "ASVspoof2019.LA.cm.train.trn.txt").write_text("\n".join(lines) + "\n")
        for name in [row[0] for row in rows] + list(extra_files):
            sf.write(audio_dir / f"{name}.flac", tone(0.5), RATE)

    def test_rows_get_paths_labels_and_extra_count(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_split(root, [("LA_T_1", "-", "bonafide"), ("LA_T_2", "A01", "spoof")], ["LA_T_9"])
            frame, extra = load_protocol(root, "train")
            self.assertEqual(frame["label"].tolist(), [0, 1])
            self.assertTrue(all(path.exists() for path in frame["path"]))
            self.assertEqual(extra, 1)
            features = mfcc_features(frame["path"], condition="noise", seed=0)
            self.assertEqual(features.shape, (2, 2 * SETTINGS["n_mfcc"]))
            np.testing.assert_array_equal(features, mfcc_features(frame["path"], condition="noise", seed=0))

    def test_missing_clip_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_split(root, [("LA_T_1", "-", "bonafide")])
            (root / "ASVspoof2019_LA_train" / "flac" / "LA_T_1.flac").unlink()
            with self.assertRaises(FileNotFoundError):
                load_protocol(root, "train")

    def test_unknown_key_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_split(root, [("LA_T_1", "-", "genuine")])
            with self.assertRaisesRegex(ValueError, "keys"):
                load_protocol(root, "train")


if __name__ == "__main__":
    unittest.main()
