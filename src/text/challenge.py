"""Prepare blind challenge review files and lock consensus labels before scoring.

No classifier is imported or invoked here. Author labels are construction targets,
not accepted ground truth until two independent teammate reviews are supplied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import random
import re
import zipfile

TOPICS = ["refund", "technical_support", "delivery", "insurance"]
REVIEWERS = ["Justin", "Hugo"]


def digest(payload: str | bytes) -> str:
    return hashlib.sha256(payload.encode() if isinstance(payload, str) else payload).hexdigest()


def dump_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="")


def dump_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def dataset_hash(rows: list[dict]) -> str:
    content = [{"dialogue_id": row["dialogue_id"], "dialogue": row["dialogue"]}
               for row in sorted(rows, key=lambda r: r["dialogue_id"])]
    return digest(json.dumps(content, sort_keys=True, ensure_ascii=False))


def reading_copy(packet: Path, rows: list[dict]):
    articles = []
    for index, row in enumerate(rows, 1):
        text = row["dialogue"]
        parts = re.split(r"\b(caller|receiver):\s*", text)[1:]
        paragraphs = "".join(f"<p><strong>{html.escape(role.title())}:</strong> {html.escape(words.strip())}</p>"
                             for role, words in zip(parts[::2], parts[1::2]))
        articles.append(f"<article><h2>Dialogue {index} of 48</h2><code>{html.escape(row['dialogue_id'])}</code>{paragraphs}</article>")
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Independent challenge review — blind reading copy</title><style>
body{max-width:850px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui,sans-serif;color:#17202a;background:white}
article{border-top:1px solid #bbc3cc;margin-top:3rem;padding-top:1rem}h1,h2{line-height:1.2}p{margin:.65rem 0}code{font-size:.9rem}
@media print{body{font-size:10pt;margin:0;max-width:none}article{break-before:page;border:0;margin:0;padding:0}p{margin:.45rem 0}}
</style><h1>Independent challenge review</h1><p>Constructed fictional conversations. Read the README before assigning labels.
Record labels, confidence, rationale, and supporting quotations in your own CSV. Do not consult another reviewer or model scores.</p>'''
    (packet / "dialogues.html").write_text(document + "".join(articles) + "</html>\n", encoding="utf-8", newline="")


def prepare(root: Path, scenarios: list[dict] | None = None):
    if scenarios is None:
        try:
            from src.text.challenge_content import SCENARIOS
        except ModuleNotFoundError as error:
            raise RuntimeError("The author construction source is held locally until independent review finishes. Use the published blind packet for review; preparation requires the author source.") from error
        scenarios = SCENARIOS
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Challenge directory already contains files; do not overwrite reviews or locked text. Create a new version directory if revision is required.")
    packet, author = root / "review_packet", root / "author_only"
    packet.mkdir(parents=True)
    author.mkdir()
    neutral = [
        "Before I act, how can I check that those steps relate to the query we have been discussing?",
        "What should I do if another page or message describes a different action from the one you are proposing?",
        "I understand the steps you described. I will review them against my records before doing anything further.",
    ]
    rows, key, pair_stats = [], [], []
    for item in scenarios:
        assert len(item["opening"]) == 6
        pair = []
        for label, variant in [(0, "legitimate"), (1, "scam")]:
            assert len(item[variant]) == 3
            turns = list(item["opening"])
            for request, response in zip(item[variant], neutral):
                turns.extend([request, response])
            dialogue = " ".join(f"{'caller' if i % 2 == 0 else 'receiver'}: {text}" for i, text in enumerate(turns))
            ident = "challenge-" + digest(dialogue)[:16]
            rows.append({"dialogue_id": ident, "dialogue": dialogue})
            word_count = len(dialogue.split())
            key.append({"dialogue_id": ident, "topic": item["topic"], "pair_id": item["scenario"],
                        "intended_label": label, "words": word_count, "turns": len(turns),
                        "text_sha256": digest(dialogue),
                        "construction_rationale": "Verified-account route and retained customer control; avoids unrelated credentials or payments" if label == 0 else "Requests control, redeemable payment, financial credentials, or unrelated transaction approval for the stated service"})
            pair.append((word_count, set(re.findall(r"\b\w+\b", dialogue.lower()))))
        ratio = max(pair[0][0], pair[1][0]) / min(pair[0][0], pair[1][0])
        pair_stats.append({"pair_id": item["scenario"], "topic": item["topic"],
                           "legitimate_words": pair[0][0], "scam_words": pair[1][0],
                           "word_length_ratio": ratio,
                           "word_set_jaccard": len(pair[0][1] & pair[1][1]) / len(pair[0][1] | pair[1][1])})
    assert len(rows) == 48 and len({r["dialogue_id"] for r in rows}) == 48
    for topic in TOPICS:
        for label in [0, 1]:
            assert sum(k["topic"] == topic and k["intended_label"] == label for k in key) == 6
    assert max(p["word_length_ratio"] for p in pair_stats) < 1.25
    random.Random(20261029).shuffle(rows)
    dump_csv(packet / "dialogues.csv", rows, ["dialogue_id", "dialogue"])
    reading_copy(packet, rows)
    dump_json(packet / "text_manifest.json", {"text_set_sha256": dataset_hash(rows),
                                               "dialogues": [{"dialogue_id": r["dialogue_id"], "text_sha256": digest(r["dialogue"])} for r in rows]})
    for index, reviewer in enumerate(REVIEWERS):
        order = list(rows)
        random.Random(20261029 + index + 1).shuffle(order)
        labels = [{"dialogue_id": r["dialogue_id"], "label": "", "confidence_1_to_5": "", "rationale": "", "evidence_quote": ""} for r in order]
        dump_csv(packet / f"{reviewer.lower()}_labels.csv", labels,
                 ["dialogue_id", "label", "confidence_1_to_5", "rationale", "evidence_quote"])
    dump_json(packet / "review_attestations.json", {
        reviewer: {"reviewer_name": "", "independent_review_completed": False,
                   "did_not_consult_author_key_or_other_reviewer_or_model_scores": False}
        for reviewer in REVIEWERS})
    instructions = """# Independent challenge review

This is a constructed diagnostic set of fictional service conversations, not real calls.
Review only this packet. Do not consult the author-only files, another reviewer's
answers, or model scores while assigning your initial labels. Topic words alone
are insufficient: decide from the requested behavior, access, payments, and the
conversation's evidence. Neither an unsolicited call nor an urgent deadline alone
establishes fraud. Treat an explicit request for unrelated credentials, deceptive
transaction approval, or diversion of funds as relevant evidence.

Justin: fill justin_labels.csv. Hugo: fill hugo_labels.csv. Keep all IDs unchanged.
Use dialogues.html as the blind reading/printing copy, or read dialogues.csv.
Label 0 = legitimate service behavior; 1 = scam behavior; uncertain = evidence is
ambiguous or the dialogue is incoherent. Supply confidence 1–5, a short rationale,
and an exact supporting quotation from the dialogue for every row. Do not change
the dialogue texts or try to achieve a particular label distribution.

Each reviewer must personally complete their own attestation in
review_attestations.json. After both initial reviews are complete, compare them
and resolve every disagreement, uncertainty, and conflict with author intent.
Record who resolved each case and why; never fabricate an independent review.
If wording must change, create a new unscored version and review changed examples
again. Return the two review CSVs and attestations to the project owner.

No model may score this set before the final consensus and text are locked.
All businesses, accounts, orders, and transactions in these examples are fictional.
"""
    (packet / "README.md").write_text(instructions, encoding="utf-8", newline="")
    dump_csv(author / "intended_labels.csv", key, list(key[0]))
    dump_csv(author / "pair_comparability.csv", pair_stats, list(pair_stats[0]))
    dump_csv(root / "adjudication.csv", [{"dialogue_id": r["dialogue_id"], "final_label": "", "resolved_by": "", "reason": ""} for r in rows],
             ["dialogue_id", "final_label", "resolved_by", "reason"])
    dump_json(root / "construction_manifest.json", {
        "version": 1, "status": "DRAFT_AWAITING_INDEPENDENT_REVIEW", "constructed_diagnostic": True,
        "author": "Coding agent; no model scores consulted", "dialogues": 48, "paired_scenarios": 24,
        "intended_counts": {topic: {"legitimate": 6, "scam": 6} for topic in TOPICS},
        "text_set_sha256": dataset_hash(rows), "turns_per_dialogue": 12,
        "word_length_min": min(k["words"] for k in key), "word_length_max": max(k["words"] for k in key),
        "maximum_paired_word_length_ratio": max(p["word_length_ratio"] for p in pair_stats),
        "pairing_limit": "Pairs share their first six turns; observations are paired, not 48 independent scenarios. Early 25%/50% prefixes do not reveal the differing requests by construction.",
        "reviewer_packet": "Share only review_packet.zip; keep author_only separate until independent reviews finish",
        "model_scoring_performed": False,
    })
    with zipfile.ZipFile(root / "review_packet.zip", "w", zipfile.ZIP_DEFLATED) as package:
        for path in packet.iterdir():
            package.write(path, path.name)
    print(f"Prepared {len(rows)} unscored dialogues and blind packet: {root / 'review_packet.zip'}")


def lock(root: Path):
    if (root / "challenge_lock.json").exists():
        raise FileExistsError("Challenge is already locked; do not replace it after scoring")
    packet = root / "review_packet"
    rows = read_csv(packet / "dialogues.csv")
    manifest = json.loads((root / "construction_manifest.json").read_text())
    assert len(rows) == 48 and dataset_hash(rows) == manifest["text_set_sha256"], "Challenge text changed; create/review a new version"
    texts = {r["dialogue_id"]: r["dialogue"] for r in rows}
    author = {r["dialogue_id"]: r for r in read_csv(root / "author_only/intended_labels.csv")}
    attestations = json.loads((packet / "review_attestations.json").read_text())
    reviews, hashes = {}, {}
    for reviewer in REVIEWERS:
        attestation = attestations[reviewer]
        if not (attestation["reviewer_name"].strip() and attestation["independent_review_completed"] is True
                and attestation["did_not_consult_author_key_or_other_reviewer_or_model_scores"] is True):
            raise ValueError(f"Independent review attestation missing for {reviewer}")
        path = packet / f"{reviewer.lower()}_labels.csv"
        labels = read_csv(path)
        assert len(labels) == 48 and len({r["dialogue_id"] for r in labels}) == 48
        assert {r["dialogue_id"] for r in labels} == set(texts)
        for row in labels:
            if row["label"] not in {"0", "1", "uncertain"} or not row["confidence_1_to_5"].isdigit():
                raise ValueError(f"Incomplete review: {reviewer}, {row['dialogue_id']}")
            if not 1 <= int(row["confidence_1_to_5"]) <= 5 or len(row["rationale"].strip()) < 10:
                raise ValueError("Review requires confidence and a substantive rationale")
            if not row["evidence_quote"].strip() or row["evidence_quote"] not in texts[row["dialogue_id"]]:
                raise ValueError("Review evidence must be a quotation from that dialogue")
        reviews[reviewer] = {r["dialogue_id"]: r for r in labels}
        hashes[reviewer] = digest(path.read_bytes())
    if attestations["Justin"]["reviewer_name"].strip().casefold() == attestations["Hugo"]["reviewer_name"].strip().casefold():
        raise ValueError("Two different independent reviewers are required")
    resolutions = {r["dialogue_id"]: r for r in read_csv(root / "adjudication.csv")}
    final = []
    for row in rows:
        ident = row["dialogue_id"]
        a, b = reviews["Justin"][ident]["label"], reviews["Hugo"][ident]["label"]
        intended = author[ident]["intended_label"]
        needs_resolution = a != b or a == "uncertain" or a != intended
        if needs_resolution:
            resolved = resolutions[ident]
            if resolved["final_label"] not in {"0", "1"} or not resolved["resolved_by"].strip() or len(resolved["reason"].strip()) < 15:
                raise ValueError(f"Adjudication required before lock: {ident}")
            label = int(resolved["final_label"])
        else:
            label = int(a)
        final.append({**row, "topic": author[ident]["topic"], "label": label, "pair_id": author[ident]["pair_id"]})
    for topic in TOPICS:
        for label in [0, 1]:
            if sum(r["topic"] == topic and r["label"] == label for r in final) != 6:
                raise ValueError("Final labels do not meet the requested 6/6 per-topic design; revise/re-review unscored text rather than forcing labels")
    for pair_id in {r["pair_id"] for r in final}:
        if {r["label"] for r in final if r["pair_id"] == pair_id} != {0, 1}:
            raise ValueError("A paired scenario lacks both resolved labels; revise/re-review the unscored pair")
    dump_csv(root / "locked_dialogues.csv", final, ["dialogue_id", "dialogue", "topic", "label", "pair_id"])
    locked = {"status": "LOCKED_BEFORE_SCORING", "constructed_diagnostic": True,
              "text_set_sha256": dataset_hash(final), "locked_csv_sha256": digest((root / "locked_dialogues.csv").read_bytes()),
              "review_sha256": hashes, "attestations_sha256": digest((packet / "review_attestations.json").read_bytes()),
              "adjudication_sha256": digest((root / "adjudication.csv").read_bytes()), "count": len(final)}
    dump_json(root / "challenge_lock.json", locked)
    print(f"Locked {len(final)} independently reviewed dialogues before scoring.")


def require_locked(root: Path) -> list[dict]:
    path = root / "challenge_lock.json"
    if not path.exists():
        raise RuntimeError("Challenge scoring blocked: independent reviews, adjudication, and challenge lock are required first")
    locked = json.loads(path.read_text())
    csv_path = root / "locked_dialogues.csv"
    if locked["status"] != "LOCKED_BEFORE_SCORING" or digest(csv_path.read_bytes()) != locked["locked_csv_sha256"]:
        raise ValueError("Locked challenge was changed")
    rows = read_csv(csv_path)
    assert len(rows) == 48 and dataset_hash(rows) == locked["text_set_sha256"]
    for reviewer, expected in locked["review_sha256"].items():
        assert digest((root / "review_packet" / f"{reviewer.lower()}_labels.csv").read_bytes()) == expected
    assert digest((root / "review_packet/review_attestations.json").read_bytes()) == locked["attestations_sha256"]
    assert digest((root / "adjudication.csv").read_bytes()) == locked["adjudication_sha256"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "lock", "check"])
    parser.add_argument("--directory", type=Path, default=Path("notebooks/text/challenge_v1"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.directory)
    elif args.action == "lock":
        lock(args.directory)
    else:
        rows = require_locked(args.directory)
        print(f"Verified {len(rows)} locked challenge dialogues.")


if __name__ == "__main__":
    main()
