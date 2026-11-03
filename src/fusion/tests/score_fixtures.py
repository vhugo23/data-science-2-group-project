"""Shared helpers that write fake score files into a temporary folder for the fusion tests.

Not a test module. Every id starts with ``a_`` or ``d_`` (never ``dummy_``), so the files look
like real score tables to the loaders.
"""
from pathlib import Path

AUDIO_PATH = "outputs/audio/smoke/wav2vec2_scores.csv"
TEXT_PATH = "outputs/text/minilm/fusion_scores_train_validation.csv"
PREFIX_PATH = "outputs/text/minilm/validation_prefix_scores.csv"


def write(root, relative, text):
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_audio(root, mode="smoke", settings=True, fold=False, test_score="0.7", run="smoke",
                model="wav2vec2", splits=("train", "validation", "test", "external")):
    header = "sample_id,split,label,score" + (",fold" if fold else "")
    rows = []
    for split in splits:
        for i, label in enumerate((0, 0, 1, 1)):
            score = test_score if split == "test" and i == 0 else f"{0.2 + 0.2 * label + 0.01 * i:.2f}"
            extra = f",{i % 5 if split == 'train' else -1}" if fold else ""
            rows.append(f"a_{split}_{i},{split},{label},{score}{extra}")
    write(root, f"outputs/audio/{run}/{model}_scores.csv", "\n".join([header] + rows) + "\n")
    if settings:
        write(root, f"outputs/audio/{run}/settings.json", f'{{"mode": "{mode}"}}')


def clean_text_score(label, i):
    return f"{0.1 + 0.3 * label + 0.01 * i:.2f}"


def write_text(root, model_dir="minilm", model_name="minilm_logreg", test_score="0.8", prefix=True,
               prefix_consistent=True, splits=("train", "validation", "test")):
    """Clean text scores, plus (optionally) validation prefix scores at fractions 0.25, 0.5 and 1.0.

    The fraction-1.0 prefix scores equal the clean validation scores when ``prefix_consistent``,
    and differ from them for both prefix dialogues otherwise.
    """
    header = "dialogue_id,split,label,model,scam_score,fold"
    rows = []
    for split in splits:
        for i, label in enumerate((0, 0, 1, 1)):
            score = test_score if split == "test" and i == 0 else clean_text_score(label, i)
            rows.append(f"d_{split}_{i},{split},{label},{model_name},{score},{i % 5 if split == 'train' else -1}")
    write(root, f"outputs/text/{model_dir}/fusion_scores_train_validation.csv", "\n".join([header] + rows) + "\n")
    if prefix:
        lines = ["dialogue_id,split,label,model,scam_score,fold,turn_fraction"]
        for fraction in (0.25, 0.5, 1.0):
            for i, label in ((0, 0), (2, 1)):  # two validation dialogues, one per class
                base = float(clean_text_score(label, i))
                if fraction == 1.0:
                    score = base if prefix_consistent else base + 0.2
                else:
                    score = base * (0.5 + 0.5 * fraction)
                lines.append(f"d_validation_{i},validation,{label},{model_name},{score:.2f},-1,{fraction}")
        write(root, f"outputs/text/{model_dir}/validation_prefix_scores.csv", "\n".join(lines) + "\n")
