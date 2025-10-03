# Copilot Instructions for transformer-recommenders

## Project overview

This repository implements transformer-based recommender models in PyTorch.
It is primarily designed for experimentation on MovieLens datasets.
The codebase separates data handling, model definitions, training, and
deployment utilities to keep experiments reproducible and modular.

## Key components

- `xfmr_rec/`: core package. Important modules:
  - `data.py`: dataset download, conversion and loader utilities
    (Parquet / LanceDB)
  - `models.py`, `mf/`, `seq/`, `seq_embedded/`: model definitions (MF,
    sequential, transformer)
  - `losses.py`: custom losses (BPR, CCL, SSM, etc.)
  - `metrics.py`: evaluation metrics and helpers
  - `trainer.py`: training loop, Lightning integration and experiment
    wiring
  - `service.py`, `deploy.py`: lightweight serving helpers for
    checkpoints

## Docstring & code style

- Docstrings: prefer Google style for public functions and classes. Include
  Args/Returns/Raises where appropriate.
- Types: add type hints for new public APIs. Keep typing incremental and
  pragmatic.
- Tests: add focused unit tests under `tests/` for new behavior. Favor
  fast, deterministic tests.

## Developer workflows

- Training: exposed as `uv` tasks (see `pyproject.toml`). Training uses
  PyTorch Lightning; pass hyperparameter overrides via CLI (e.g.
  `--trainer.max_epochs 16`).
- Data: use `uv run data` to prepare MovieLens datasets (Parquet). For
  retrieval benchmarks, create LanceDB datasets under `lance_db/`.
- Experiment tracking: Lightning + MLflow store runs in
  `lightning_logs/` and `mlruns/`.
- Deployment: `xfmr_rec/deploy.py` and `xfmr_rec/service.py` can load
  checkpoints and provide a simple retrieval API. For production, embed
  these into a proper HTTP server and add batching and concurrency
  controls.

## Patterns & conventions

- Keep model implementations small and modular; prefer composition over
  monolithic classes.
- Place new model variants in the corresponding folder (`mf/`, `seq/`, or
  `seq_embedded/`).
- Implement new loss variants in `xfmr_rec/losses.py` and add unit tests
  that check shapes, dtypes, and small numerical sanity checks.
- Use Parquet for development datasets; use LanceDB for
  retrieval/benchmarking workloads.

## Commit messages and PRs

Follow Conventional Commits: `type[optional scope]: short description`.
Optionally add a longer body and footers for breaking changes.

Common types: feat, fix, docs, style, refactor, perf, test, ci, chore.

Examples:

- `feat(data): add lance db ingestion script`
- `fix(trainer): handle empty validation set`
- `docs: update README with usage examples`

PR checklist for contributors and automated agents:

1. Run unit tests and linting for modified files.
2. Add or update unit tests for behavior changes.
3. Confirm no large data files or secrets were added.
4. Include a short PR description with verification steps (build, tests,
  smoke checks).

## Automated agent guidance

- When an automated agent (bot/copilot) modifies code, create focused
  commits and include `bot` in the scope when relevant (for example,
  `chore(bot): update docs`).
- Ensure at least the tests covering changed modules are run locally.
  Prefer adding a small unit test when changing public behavior.
- If an agent cannot run tests in the environment, state which checks
  were attempted and any limitations in the PR description.

## Tooling suggestions (optional)

- commitizen to help format Conventional Commits
- git-cliff or conventional-changelog for changelog generation
- husky + commitlint to enforce commit message format locally

For more details, see the README and referenced papers.
