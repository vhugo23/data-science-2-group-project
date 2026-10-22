# Detecting Voice-Cloned Scam Calls

Starter scaffold for Group 30's COSC 4337 project. Goal: combine audio spoof detection and transcript-based scam classification in an offline call-triage prototype. This is a research prototype, not a real-time service.

## Setup

Use the shared coursework environment from `python-coursework/`:

```bash
cd /path/to/python-coursework
uv sync --group gen-ai
```

`requirements.txt` lists the project-specific additions for reference; the root `pyproject.toml` and `uv.lock` remain the authoritative shared environment.

Copy `.env.example` to `.env` in this project folder and set paths to local datasets. `.env` is ignored by git; never commit dataset contents or credentials.

## Datasets

- **ASVspoof 2019 LA**: official train/dev/evaluation splits and labels. Use official evaluation split for final in-domain audio evaluation.
- **BothBosu/scam-dialogue**: 1,600 labeled synthetic English phone dialogues; preserve the provided 1,280/320 train/test split.
- **In-the-Wild**: external audio test only; do not train or tune on it.
- **ASVspoof 2021 LA**: optional external test for telephony/channel effects.

Obtain datasets from their official or approved sources and follow each dataset's license and access rules. See `src/datasets.py` for path configuration.

## Suggested order

1. Confirm dataset access, exact directory layouts, and group task ownership.
2. Run `notebooks/00_project_setup.ipynb` to check environment and configured paths.
3. Establish separate baselines for spoof detection and scam-text classification.
4. Evaluate the two branches separately before score-level fusion.
5. Compare clean and degraded audio; include the untrimmed-audio ablation and report external tests without retraining.
6. Record split choices, metrics, seeds, model settings, and limitations in `reports/`.

## Starter files

- `notebooks/00_project_setup.ipynb` — environment and local-data checklist.
- `src/datasets.py` — `.env` loading, path map, and small dataset-loading examples.
- `src/metrics.py` — evaluation metrics placeholders.
- `reports/` — save experiment notes and figures.

Keep experiments flat and readable. Do not infer caller identity or treat either output as proof of fraud; report spoof authenticity and scam intent as separate tasks.
