
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

Requires Python 3.12+. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

### Data Preparation

Download and process MovieLens 1m (`ml-1m`) data in `data/` and save in parquet format.

```bash
uv run data
```

### Training

Train a model using provided scripts:

```bash
uv run seq_train fit --trainer.max_epochs 16
```

### Deployment

Serve a trained model:

```bash
uv run python -m xfmr_rec.seq.deploy --ckpt_path <path>
```

## Project Conventions

- Models are organized by type in subfolders for extensibility
- Custom loss functions are in `losses.py` and referenced in training code
- Experiment tracking via PyTorch Lightning and MLflow
- Data access optimized with Parquet and LanceDB

## Entrypoints

Project scripts (see `pyproject.toml`):

- `data`: Data utilities
- `mf_train`, `mf_deploy`, `mf_tune`: Matrix factorization workflows
- `seq_train`, `seq_deploy`, `seq_tune`: Sequential model workflows
- `seq_embedded_train`, `seq_embedded_deploy`: Transformer-based sequential workflows

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
