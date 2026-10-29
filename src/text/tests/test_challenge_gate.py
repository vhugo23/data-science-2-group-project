"""Structural gate tests using temporary synthetic reviewer fixtures only.

These fixtures are not real independent reviews and never touch the actual packet.
No model or embedding is invoked by these tests.
"""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.text.challenge import TOPICS, dump_csv, dump_json, lock, prepare, read_csv, require_locked


def synthetic_scenarios():
    """Public test fixtures are independent of the withheld real challenge key."""
    return [
        {"topic": topic, "scenario": f"TEST_FIXTURE_{topic}_{index}",
         "opening": [f"Synthetic structural fixture {topic} case {index} turn {turn}." for turn in range(6)],
         "legitimate": [f"Synthetic legitimate fixture request {turn}; this is not a real reviewer label." for turn in range(3)],
         "scam": [f"Synthetic fraudulent fixture request {turn}; this is not a real reviewer label." for turn in range(3)]}
        for topic in TOPICS for index in range(6)
    ]


class ChallengeGateTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.root = Path(self.workspace.name) / "fixture_challenge"
        prepare(self.root, scenarios=synthetic_scenarios())

    def tearDown(self):
        self.workspace.cleanup()

    def fixture_reviews(self):
        author = {r["dialogue_id"]: r for r in read_csv(self.root / "author_only/intended_labels.csv")}
        texts = {r["dialogue_id"]: r["dialogue"] for r in read_csv(self.root / "review_packet/dialogues.csv")}
        for reviewer in ["justin", "hugo"]:
            path = self.root / "review_packet" / f"{reviewer}_labels.csv"
            rows = read_csv(path)
            for row in rows:
                row.update(label=author[row["dialogue_id"]]["intended_label"], confidence_1_to_5="4",
                           rationale="Temporary structural test fixture, not a human review.",
                           evidence_quote=texts[row["dialogue_id"]][:80])
            dump_csv(path, rows, list(rows[0]))
        dump_json(self.root / "review_packet/review_attestations.json", {
            name: {"reviewer_name": f"SYNTHETIC_TEST_FIXTURE_{name}", "independent_review_completed": True,
                   "did_not_consult_author_key_or_other_reviewer_or_model_scores": True}
            for name in ["Justin", "Hugo"]})

    def test_unreviewed_packet_cannot_be_scored_or_locked(self):
        with self.assertRaisesRegex(RuntimeError, "scoring blocked"):
            require_locked(self.root)
        with self.assertRaisesRegex(ValueError, "attestation missing"):
            lock(self.root)
        self.assertFalse((self.root / "challenge_lock.json").exists())

    def test_disagreement_requires_adjudication(self):
        self.fixture_reviews()
        path = self.root / "review_packet/hugo_labels.csv"
        rows = read_csv(path)
        ident = rows[0]["dialogue_id"]
        rows[0]["label"] = "uncertain"
        dump_csv(path, rows, list(rows[0]))
        with self.assertRaisesRegex(ValueError, "Adjudication required"):
            lock(self.root)
        key = {r["dialogue_id"]: r for r in read_csv(self.root / "author_only/intended_labels.csv")}
        rows = read_csv(self.root / "adjudication.csv")
        for row in rows:
            if row["dialogue_id"] == ident:
                row.update(final_label=key[ident]["intended_label"], resolved_by="SYNTHETIC_TEST_FIXTURE",
                           reason="Temporary fixture resolution for testing the gate only.")
        dump_csv(self.root / "adjudication.csv", rows, list(rows[0]))
        lock(self.root)
        self.assertEqual(len(require_locked(self.root)), 48)

    def test_lock_rejects_changed_text_and_duplicate_lock(self):
        self.fixture_reviews()
        lock(self.root)
        self.assertEqual(len(require_locked(self.root)), 48)
        with self.assertRaises(FileExistsError):
            lock(self.root)
        with (self.root / "locked_dialogues.csv").open("a", encoding="utf-8") as stream:
            stream.write("tampered\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            require_locked(self.root)

    def test_independent_attestations_are_required(self):
        self.fixture_reviews()
        path = self.root / "review_packet/review_attestations.json"
        attestations = json.loads(path.read_text())
        attestations["Hugo"]["independent_review_completed"] = False
        dump_json(path, attestations)
        with self.assertRaisesRegex(ValueError, "attestation missing for Hugo"):
            lock(self.root)

    def test_scoring_entry_point_stops_before_models_are_loaded(self):
        from src.text.comparison import score_challenge
        with patch("src.text.comparison.verify_freeze") as model_check, patch("src.text.comparison.evaluate") as evaluate:
            with self.assertRaisesRegex(RuntimeError, "scoring blocked"):
                score_challenge(self.root, Path("unused_baseline"), Path("unused_minilm"), self.root / "never_scored")
            model_check.assert_not_called()
            evaluate.assert_not_called()
        self.assertFalse((self.root / "never_scored").exists())

    def test_official_test_stays_closed_without_challenge_results(self):
        from src.text.comparison import official_test
        self.fixture_reviews()
        lock(self.root)
        with patch("src.text.comparison.verify_freeze") as model_check:
            with self.assertRaisesRegex(RuntimeError, "Official test blocked"):
                official_test(self.root, self.root / "no_results", Path("unused_baseline"), Path("unused_minilm"),
                              Path("unused_manifest"), self.root / "no_test_source", self.root / "no_test_output")
            model_check.assert_not_called()
        self.assertFalse((self.root / "no_test_source").exists())


if __name__ == "__main__":
    unittest.main()
