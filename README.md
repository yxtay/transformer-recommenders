# transformer-recommenders

Transformer-based Recommender Models in PyTorch for MovieLens

## Overview

This repository provides modular implementations of recommender systems
using transformer architectures, matrix factorization, and sequential models.
It is designed for research and experimentation on MovieLens data,
with scalable data access and experiment tracking.

## Architecture & Components

- **Core package:** `xfmr_rec/`
  - `data.py`: Data loading and preprocessing (MovieLens, LanceDB)
  - `models.py`, `mf/`, `seq/`, `seq_embedded/`:
    Model architectures (MF, sequential, transformer)
  - `losses.py`: Custom loss functions (BPR, CCL, SSM, etc.)
  - `metrics.py`: Evaluation metrics
  - `trainer.py`: Training loop and experiment management (PyTorch Lightning)
  - `service.py`, `deploy.py`: Model serving and deployment utilities
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

Example: prepare MovieLens 1M (ml-1m) and write parquet files into
`data/`:

```bash
# fetch, extract and convert to parquet
uv run data
```

If you already have the original files (for example `ml-1m.zip`), place
them under `data/` and `uv run data` will pick them up. Otherwise the
script will download and extract the dataset.

### Training

Training is implemented with PyTorch Lightning. The repository exposes
several task entrypoints.

Common training commands:

```bash
# Train a sequential transformer model for 16 epochs
uv run seq_train fit --trainer.max_epochs 16

# Train a matrix factorization model
uv run mf_train fit --trainer.max_epochs 10
```

Check `pyproject.toml` entrypoints for available tasks and the
`xfmr_rec/` modules for model and trainer configuration.

### Deployment and serving

The repository contains light-weight deployment utilities to run a retrieval
service from a Lightning checkpoint.

```bash
# Serve a sequential model checkpoint on localhost
uv run python -m xfmr_rec.seq.deploy --ckpt_path <path/to/checkpoint.ckpt>
```

See `xfmr_rec/service.py` and `xfmr_rec/deploy.py` for convenience
functions that load a checkpoint and expose a simple predict/retrieve
API. The code uses LanceDB or parquet data for fast lookups when
available.

## Project conventions

- Models are organized by type in subfolders (`mf/`, `seq/`,
  `seq_embedded/`) for extensibility.
- Custom loss functions live in `xfmr_rec/losses.py` and are referenced
  by trainer hooks.
- Experiment tracking is handled by PyTorch Lightning and MLflow;
  checkpoints and logs are stored in `lightning_logs/` and `mlruns/`.
- Data access is optimized using Parquet and (optionally) LanceDB for
  retrieval workloads.

## Entrypoints

Task entrypoints are defined in `pyproject.toml` and wired to `uv` tasks.
Typical entrypoints include:

- `data`: datasets download and conversion utilities
- `mf_train`, `mf_deploy`, `mf_tune`:
  matrix factorization training / deploy / tuning workflows
- `seq_train`, `seq_deploy`, `seq_tune`:
  sequential / transformer training / deploy / tuning workflows
- `seq_embedded_train`, `seq_embedded_deploy`:
  transformer (embedded) sequential workflows

Run `uv run` (without args) to list available tasks, or inspect
`pyproject.toml` for concrete command mappings.

## Development notes & troubleshooting

- If you see dependency or Python version errors, confirm you are using
  Python 3.12 and run `uv sync` to recreate the virtual environment.
- If training fails with out-of-memory errors, reduce
  `trainer.batch_size` or enable gradient accumulation via
  `trainer.accumulate_grad_batches`.
- Use the Lightning logs folder (`lightning_logs/`) to inspect
  checkpoints and tensorboard summaries.

## References

- Google Slides: [Collaborative Filtering with Implicit Feedback][google-slides]
- [[2101.08769]][implicit-feedback] Item Recommendation from Implicit Feedback
- [TensorFlow Recommenders Retrieval][tfrs-retrieval]
- BPR: [[1205.2618]][bpr] Bayesian Personalized Ranking from Implicit Feedback
- CCL: [[2109.12613]][ccl]
  SimpleX: A Simple and Strong Baseline for Collaborative Filtering
- SSM: [[2201.02327]][ssm]
  On the Effectiveness of Sampled Softmax Loss for Item Recommendation
- DirectAU: [[2206.12811]][direct-au]
  Towards Representation Alignment and Uniformity in Collaborative Filtering
- MAWU: [[2308.06091]][mawu]
  Toward a Better Understanding of Loss Functions for Collaborative Filtering
- InfoNCE+, MINE+: [[2312.08520]][mine+]
  Revisiting Recommendation Loss Functions through Contrastive Learning
- LogQ correction:
  [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations][logq]
- MNS:
  [Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations][mns]
- Hashing Trick: [[0902.2206]][hashing-trick]
  Feature Hashing for Large Scale Multitask Learning
- Hash Embeddings: [[1709.03933]][hash-embeddings]
  Hash Embeddings for Efficient Word Representations
- Bloom embeddings: [Compact word vectors with Bloom embeddings][bloom-embeddings]

[google-slides]: https://docs.google.com/presentation/d/15nLFgmkSEJPXkhLiXExXDByV_lot7bdHAhtqX_qLp7w/
[implicit-feedback]: https://arxiv.org/abs/2101.08769
[tfrs-retrieval]: https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval
[bpr]: https://arxiv.org/abs/1205.2618
[ccl]: https://arxiv.org/abs/2109.12613
[ssm]: https://arxiv.org/abs/2201.02327
[direct-au]: https://arxiv.org/abs/2206.12811
[mawu]: https://arxiv.org/abs/2308.06091
[mine+]: https://arxiv.org/abs/2312.08520
[logq]: https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/
[mns]: https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/
[hashing-trick]: https://arxiv.org/abs/0902.2206
[hash-embeddings]: https://arxiv.org/abs/1709.03933
[bloom-embeddings]: https://explosion.ai/blog/bloom-embeddings
