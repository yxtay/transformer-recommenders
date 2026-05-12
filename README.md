# transformer-recommenders

Transformer-based Recommender Models in PyTorch for MovieLens

## Overview

This repository provides a sequential transformer recommender system using item embeddings from pre-trained sentence-transformers. It is designed for research and experimentation on MovieLens data, with scalable data access and experiment tracking.

## Architecture & Components

- **Core package:** `xfmr_rec/`
  - `data.py`: Data loading, preprocessing (MovieLens, LanceDB), and PyTorch Lightning DataModule.
  - `models.py`: Sequential transformer architecture.
  - `losses.py`: Custom loss functions (BPR, CCL, SSM, etc.)
  - `metrics.py`: Evaluation metrics
  - `trainer.py`: Training loop and experiment management (PyTorch Lightning)
  - `service.py`, `deploy.py`: Model serving and deployment utilities (BentoML)
- **Data:**
  - `data/`: Raw and processed MovieLens datasets (Parquet format)
  - `lance_db/`: LanceDB format for fast retrieval
- **Experiment Logs:**
  - `lightning_logs/`, `mlruns/`: Model checkpoints and experiment tracking (MLflow)

## Installation

Requirements

- Python 3.12+ (the project is developed and tested on 3.12)
- The repository uses `uv` to manage virtual environments and tasks.
  See `pyproject.toml` for pinned dependencies.

Install dependencies with uv (recommended):

```bash
# set up the environment and install pinned deps
uv sync
```

## Usage

### Data preparation

This repo ships helper scripts to download and convert MovieLens
datasets into Parquet and LanceDB formats.

```bash
# fetch, extract and convert to parquet
uv run data
```

### Training

Training is implemented with PyTorch Lightning.

```bash
# Train the model for 16 epochs
uv run train fit --trainer.max_epochs 16
```

### Deployment and serving

The repository contains utilities to run a retrieval service from a Lightning checkpoint using BentoML.

```bash
# Deploy a model checkpoint to BentoML
uv run deploy --ckpt_path <path/to/checkpoint.ckpt>
```

## Entrypoints

Task entrypoints are defined in `pyproject.toml` and wired to `uv` tasks.

- `data`: datasets download and conversion utilities
- `train`: transformer training workflow
- `deploy`: transformer deploy workflow

Run `uv run` (without args) to list available tasks, or inspect `pyproject.toml` for concrete command mappings.

## References

- [Sentence Transformers](https://www.sbert.net/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [BentoML](https://www.bentoml.com/)
- [LanceDB](https://lancedb.com/)
