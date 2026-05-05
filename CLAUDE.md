# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Lint and format (via pre-commit)
uv run ruff check --fix-only --unsafe-fixes .
uv run ruff format .

# Verify a trainer config prints cleanly
uv run seq_train fit --print_config
uv run mf_train fit --print_config

# Quick smoke test (1 train/val batch, no data workers)
uv run seq_train fit --trainer.limit_train_batches 1 --trainer.limit_val_batches 1 --data.config.num_workers 0
uv run mf_train fit --trainer.limit_train_batches 1 --trainer.limit_val_batches 1 --data.config.num_workers 0
uv run seq_embedded_train fit --trainer.limit_train_batches 1 --trainer.limit_val_batches 1 --data.config.num_workers 0

# Deploy (validate a saved checkpoint)
uv run seq_deploy
uv run mf_deploy

# Prepare data
uv run data
```

## Architecture

The package is `xfmr_rec/`. Three model families, each in its own subpackage with a consistent file layout
(`trainer.py`, `data.py`, `models.py`, `service.py`, `deploy.py`, `tune.py`):

| Subpackage      | Description                                                           |
|-----------------|-----------------------------------------------------------------------|
| `mf/`           | Matrix factorisation two-tower model                                  |
| `seq/`          | Sequential transformer — learns from interaction history              |
| `seq_embedded/` | Like `seq/` but with pre-trained sentence-transformer item embeddings |

**Shared utilities in `xfmr_rec/`:**

- `models.py` — `ModelConfig` (pydantic), `init_bert`, `init_sent_transformer`;
  the base transformer building blocks used by each subpackage
- `losses.py` — pluggable contrastive/ranking losses (BPR, CCL, SSM, DirectAU, InfoNCE+, etc.) controlled by `LossConfig`
- `metrics.py` — retrieval evaluation metrics
- `trainer.py` — `LightningCLI` wrapper: sets up TensorBoard + MLflow loggers and bf16-mixed precision defaults;
  all trainers go through this
- `data.py` — MovieLens download, Parquet conversion, LanceDB ingestion
- `index.py` / `service.py` / `deploy.py` — FAISS/LanceDB index construction, BentoML service scaffolding,
  checkpoint-to-service helpers

**Training flow:** Each `<model>/trainer.py` defines a `LightningModule` + `LightningDataModule` and calls
`xfmr_rec.trainer.LightningCLI.main()`. The CLI is JSONArgparse-based, so all hyperparameters can be
overridden from the command line or a YAML config file.

**Serving flow:** `<model>/deploy.py` loads a Lightning checkpoint, reconstructs the model config via
`LightningCLI.load_args`, builds a FAISS/LanceDB index, and registers the model in BentoML.
`<model>/service.py` wraps it in a BentoML `Service`.

## Toolchain

- **Package manager / task runner:** `uv` (locked, `UV_LOCKED=1` in CI)
- **Linter / formatter:** `ruff` — `ALL` rules selected; `D`, `E501`, `S101`, `T201`, `COM812`, `ERA001`,
  `ISC001`, `PLC0415` ignored
- **Pre-commit hooks:** ruff, taplo (TOML), yamlfmt, markdownlint-cli2, shfmt, shellcheck, hadolint, gitleaks, uv-lock
- **Experiment tracking:** MLflow + TensorBoard (logs in `mlruns/` and `lightning_logs/`)
- **Container:** `Dockerfile` + `compose.yaml`; image at `ghcr.io/yxtay/transformer-recommenders`
