# Copilot Instructions for transformer-recommenders

## Project Overview

This repository implements transformer-based recommender models in PyTorch,
focused on MovieLens data. The architecture is modular, with clear separation between
data loading, model definitions, training logic, and deployment scripts.

## Key Components & Structure

- `xfmr_rec/`: Main package. Submodules include:
  - `data.py`: Data loading and preprocessing (MovieLens, LanceDB)
  - `models.py`, `mf/`, `seq/`, `seq_embedded/`:
    Model architectures (matrix factorization, sequential, transformer-based)
  - `losses.py`: Custom loss functions for recommendation (BPR, CCL, SSM, etc.)
  - `metrics.py`: Evaluation metrics
  - `trainer.py`: Training loop and experiment management
  - `service.py`, `deploy.py`: Deployment and serving utilities
- `data/`: Contains raw and processed MovieLens datasets
- `lance_db/`: LanceDB format for fast data access
- `lightning_logs/`, `mlruns/`: Experiment logs and model checkpoints

## Developer Workflows

- **Training**: Run model training via scripts in `xfmr_rec/` (e.g., `trainer.py`).
  Use PyTorch Lightning for experiment management.
- **Data**:
  Use `data.py` for loading and preprocessing. Supports LanceDB for scalable access.
- **Experiment Tracking**:
  Results and checkpoints are stored in `lightning_logs/` and `mlruns/`.
- **Deployment**:
  Use `deploy.py` and `service.py` for model serving.
  Integration with LanceDB for retrieval.

## Patterns & Conventions

- **Loss Functions**:
  Custom losses are implemented in `losses.py` and referenced in model/trainer code.
  See references in README for theoretical background.
- **Modularity**: Models are organized by type (MF, sequential, transformer)
  in subfolders for clarity and extensibility.
- **Data Format**: Parquet files for efficient I/O; LanceDB for fast retrieval.
- **Experiment Logging**: PyTorch Lightning and MLflow are used for tracking.
- **Naming**:
  Model variants and experiments are named by approach (e.g., `xfmr_mf_rec`, `xfmr_seq_rec`).

## Integration Points

- **External Libraries**: PyTorch, PyTorch Lightning, LanceDB, MLflow
- **References**:
  See README for key papers and links that inform model and loss design.

## Example Workflow

1. Prepare data in `data/` (ensure Parquet format)
2. Train a model: `uv run seq_train fit`
3. Check results in `lightning_logs/`
4. Deploy: `uv run python -m xfmr_rec.seq.deploy --ckpt_path <path>`

## Tips for AI Agents

- Always check `xfmr_rec/` for core logic and conventions
- Losses and metrics are project-specific; reference `losses.py` and `metrics.py`
- Use experiment logs for debugging and validation
- Follow modular structure for adding new models or data sources

---
For more details, see the README and referenced papers.
