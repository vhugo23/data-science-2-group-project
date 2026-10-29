# Detecting Voice-Cloned Scam Calls

Group 30's COSC 4337 project combines audio-spoof detection with transcript-based scam classification in an offline call-triage prototype. It is a research prototype, not a real-time service.

## Setup

Install [uv](https://docs.astral.sh/uv/) and Python 3.12. From the repository root, create the environment outside any cloud-synced folder, then install the exact versions in `uv.lock`.

Linux/macOS/WSL:

```bash
mkdir -p "$HOME/.venvs"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/data-science-2-group-project"
uv sync --locked
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$HOME\.venvs" | Out-Null
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\data-science-2-group-project"
uv sync --locked
```

Set `UV_PROJECT_ENVIRONMENT` again in each new terminal session. The project-local `pyproject.toml` and `uv.lock` define the same dependencies for every teammate; each machine keeps its environment and downloaded model cache locally.

Copy `.env.example` to `.env` in the repository root and set paths to datasets available on that machine. `.env` and dataset contents are ignored by Git. Do not commit local paths, dataset files, credentials, or model caches.

Start Jupyter from the repository root with `uv run jupyter lab`, then open `notebooks/project_setup.ipynb` to check the local dataset paths.

## Team ownership

Each person works on their own branch and changes files in their assigned area. Create a pull request into `main` when a track is ready to integrate.

| Owner | Owns | Work |
| --- | --- | --- |
| Hugo Santos | `src/audio/`, `notebooks/audio/` | Audio preprocessing and embeddings, spoof model, audio benchmarks |
| Justin Kondratenko | `src/text/`, `notebooks/text/`, shared setup files | Text preprocessing, NLP baselines and model, cross-dataset text evaluation |
| Joshua Eworo | `src/fusion/`, `notebooks/fusion/`, `src/metrics.py` | Pair construction and leakage checks, fusion, robustness, shared evaluation and failure analysis |

Keep shared setup changes in Justin's branch and coordinate any changes that cross track folders before editing. `src/datasets.py` is the shared `.env` path helper.

## Integration contract

- Audio and text tracks each export a score table with a stable sample ID, `split`, `label`, and model `score`.
- Use `train`, `validation`, and `test` as the shared split names. Map ASVspoof 2019 LA's official train/dev/evaluation splits without changing their membership. Preserve the scam-dialogue 1,280/320 train/test split; draw validation data only from its training split.
- Pairing tables identify both source IDs and their split. Split source pools before pairing; no audio clip or dialogue may appear in multiple splits.
- Train the learned fusion model on out-of-fold component scores. Keep external datasets for evaluation only; do not tune models or thresholds on them.
- Keep downloaded data and generated artifacts out of Git. Record preprocessing, seeds, metrics, model settings, and limitations with the experiment results.

## Datasets and experiment sequence

- **ASVspoof 2019 LA**: official train/dev/evaluation splits and labels; use its evaluation split for final in-domain audio evaluation.
- **BothBosu/scam-dialogue**: 1,600 labeled synthetic English phone dialogues; preserve its provided train/test split.
- **In-the-Wild**: external audio test only; do not train or tune on it.
- **ASVspoof 2021 LA**: optional external test for telephony/channel effects.

Obtain datasets from approved sources and follow each dataset's license and access rules. See `src/datasets.py` for path configuration.

1. Confirm dataset access, directory layouts, and labels.
2. Establish separate audio-spoof and scam-text baselines.
3. Evaluate both tracks separately before score-level fusion.
4. Compare clean and degraded audio, including the untrimmed-audio ablation.
5. Run robustness and external tests without retraining or tuning on external data.

Keep experiments flat and readable. Do not infer caller identity or treat either output as proof of fraud; report spoof authenticity and scam intent as separate tasks.
