from __future__ import annotations

import datetime
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any

import bentoml
import lightning as lp
import lightning.pytorch.callbacks as lp_callbacks
import lightning.pytorch.cli as lp_cli
import lightning.pytorch.loggers as lp_loggers
import torch
from loguru import logger

from xfmr_rec import losses as loss_classes
from xfmr_rec.data import SeqBatch, SeqDataModule
from xfmr_rec.index import LanceIndex, LanceIndexConfig
from xfmr_rec.losses import LOSS_CLASSES, LossConfig, LossType
from xfmr_rec.metrics import compute_retrieval_metrics
from xfmr_rec.models import ModelConfig, RecommenderModel
from xfmr_rec.params import (
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    METRIC,
    TOP_K,
    TRANSFORMER_PATH,
    USERS_TABLE_NAME,
)

if TYPE_CHECKING:
    import datasets
    import numpy as np
    from mlflow.tracking import MlflowClient


class LoggerSaveConfigCallback(lp_cli.SaveConfigCallback):
    def save_config(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,  # noqa: ARG002
        stage: str,  # noqa: ARG002
    ) -> None:
        """Save the JSONArgparse config to MLflow and log hyperparameters.

        Iterates over the trainer's loggers and for each MLflow logger:
        - Logs hyperparameters using the MLflow logger helper.
        - Writes the JSONArgparse config to a temporary file and uploads it
            as an artifact to the MLflow run.

        Args:
            trainer (lp.Trainer): The Lightning trainer instance which holds
                attached loggers.
            pl_module (lp.LightningModule): The Lightning module being trained.
                (Unused in this implementation but kept for signature
                compatibility with the parent callback.)
            stage (str): The current stage name (e.g., "fit", "validate").

        Notes:
            Only acts on loggers that are instances of
            ``lp_loggers.MLFlowLogger``. The JSONArgparse config is saved to a
            temporary file and uploaded to the active MLflow run via the
            MlflowClient API.
        """
        for lp_logger in trainer.loggers:
            if not isinstance(lp_logger, lp_loggers.MLFlowLogger):
                continue

            lp_logger.log_hyperparams(self.config.as_dict())
            with tempfile.TemporaryDirectory() as path:
                config_path = pathlib.Path(path, self.config_filename)
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=False,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )
                mlflow_client: MlflowClient = lp_logger.experiment
                mlflow_client.log_artifact(
                    run_id=lp_logger.run_id, local_path=config_path
                )


def time_now_isoformat() -> str:
    """Return the current time as an ISO 8601 formatted string.

    The returned string is timezone-aware (the system local timezone) and
    formatted to seconds precision. Example: "2025-10-02T12:34:56+08:00".

    Returns:
        str: ISO 8601 formatted current datetime with timezone offset.
    """
    datetime_now = datetime.datetime.now(datetime.UTC).astimezone()
    return datetime_now.isoformat(timespec="seconds")


class LightningConfig(LossConfig, ModelConfig):
    train_loss: LossType = "InfoNCELoss"
    learning_rate: float = 0.001
    weight_decay: float = 0.01

    items_config: LanceIndexConfig = LanceIndexConfig(
        table_name=ITEMS_TABLE_NAME,
        id_col="item_id",
        text_col="item_text",
        embedding_col="embedding",
    )
    users_config: LanceIndexConfig = LanceIndexConfig(
        table_name=USERS_TABLE_NAME,
        id_col="user_id",
        text_col="user_text",
        embedding_col=None,
    )
    top_k: int = TOP_K


class RecommenderLightningModule(lp.LightningModule):
    def __init__(self, config: LightningConfig) -> None:
        """Initialize the Recommender Lightning module.

        Args:
            config (LightningConfig): Configuration dataclass
                containing model and training hyperparameters.
        """
        super().__init__()
        self.config = LightningConfig.model_validate(config)
        self.save_hyperparameters(self.config.model_dump())
        self.strict_loading = False

        self.model: RecommenderModel | None = None
        self.items_dataset: datasets.Dataset | None = None
        self.loss_fns: torch.nn.ModuleList | None = None
        self.items_index = LanceIndex(self.config.items_config)
        self.users_index = LanceIndex(self.config.users_config)

        logger.info(repr(self.config))

    def configure_model(self) -> None:
        """Ensure the model and item embeddings are initialized.

        This will create the ``RecommenderModel`` instance if missing,
        attach the datamodule's items dataset when available, and call the
        model's ``configure_embeddings`` helper to prepare item embeddings.
        Also instantiates loss functions.
        """
        if self.model is None:
            self.model = RecommenderModel(self.config, device=self.device)

        if self.items_dataset is None:
            try:
                self.items_dataset = self.trainer.datamodule.items_dataset
            except RuntimeError as e:
                # RuntimeError if trainer is not attached
                logger.warning(repr(e))

        if self.items_dataset is not None:
            self.model.configure_embeddings(self.items_dataset)

        if self.loss_fns is None:
            self.loss_fns = self.get_loss_fns()

    def get_loss_fns(self) -> torch.nn.ModuleList:
        """Create the configured loss function modules.

        Returns:
            torch.nn.ModuleList: Instantiated loss modules.
        """
        loss_fns = [loss_class(self.config) for loss_class in LOSS_CLASSES]
        return torch.nn.ModuleList(loss_fns)

    def forward(self, item_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the model forward on a tensor of item indices.

        Args:
            item_idx (torch.Tensor): Tensor of item indices with shape
                (batch, seq_len) or similar as expected by the model.

        Returns:
            dict[str, torch.Tensor]: Model outputs including embeddings and
                attention masks used by downstream loss computation.
        """
        assert self.model is not None
        return self.model(item_idx.to(self.device))

    @torch.inference_mode()
    def recommend(
        self,
        item_ids: list[str],
        *,
        top_k: int = 0,
        exclude_item_ids: list[str] | None = None,
    ) -> datasets.Dataset:
        """Return nearest-neighbour recommendations for given item ids.

        Args:
            item_ids (list[str]): Item ids used as queries.
            top_k (int, optional): Number of results to return.
            exclude_item_ids (list[str] | None, optional): Item ids to
                exclude from results.

        Returns:
            datasets.Dataset: Search results from the items index.
        """
        assert self.model is not None
        embedding = self.model.encode(item_ids).numpy(force=True)
        return self.items_index.search(
            embedding,
            exclude_item_ids=exclude_item_ids,
            top_k=top_k or self.config.top_k,
        )

    def compute_losses(self, batch: SeqBatch) -> dict[str, torch.Tensor]:
        """Compute losses and logging metrics for a batch.

        Computes query and candidate embeddings via the model, derives
        batch-level statistics (sequence density etc.), computes logits
        statistics and evaluates each configured loss function.

        Args:
            batch (SeqBatch): Batch containing history/pos/neg item index
                tensors suitable for the model's embedder.

        Returns:
            dict[str, torch.Tensor]: Mapping of loss and metric names to
                scalar tensors for logging.
        """
        assert self.model is not None
        assert self.loss_fns is not None
        embeds = self.model.compute_embeds(
            batch["history_item_idx"],
            batch["pos_item_idx"],
            batch["neg_item_idx"],
        )

        attention_mask = embeds["attention_mask"]
        batch_size, seq_len = attention_mask.size()
        numel = attention_mask.numel()
        attn_non_zero = attention_mask.count_nonzero().item()
        pos_non_zero = embeds["positive_mask"].count_nonzero().item()
        metrics: dict[str, float] = {
            "batch/size": batch_size,
            "batch/seq_len": seq_len,
            "batch/numel": numel,
            "batch/attention_non_zero": attn_non_zero,
            "batch/attention_density": attn_non_zero / (numel + 1e-9),
            "batch/positive_non_zero": pos_non_zero,
            "batch/positive_density": pos_non_zero / (attn_non_zero + 1e-9),
        }
        metrics |= loss_classes.LogitsStatistics(self.config)(
            query_embed=embeds["query_embed"],
            candidate_embed=embeds["candidate_embed"],
        )

        losses: dict[str, torch.Tensor] = {}
        for loss_fn in self.loss_fns:
            loss = loss_fn(
                query_embed=embeds["query_embed"],
                candidate_embed=embeds["candidate_embed"],
            )
            key = f"loss/{loss_fn.__class__.__name__}"
            losses[key] = loss
            losses[f"{key}Mean"] = loss / (pos_non_zero + 1e-9)
        return losses | metrics

    def compute_metrics(
        self, row: dict[str, dict[str, np.ndarray]], stage: str = "val"
    ) -> dict[str, torch.Tensor]:
        """Compute retrieval metrics for a validation/test row.

        Args:
            row (dict): A datamodule-produced row with targets and history.
            stage (str, optional): Metric prefix (e.g., "val", "test").

        Returns:
            dict[str, torch.Tensor]: Mapping of metric names to tensors.
        """
        recs = self.predict_step(row)
        metrics = compute_retrieval_metrics(
            rec_ids=recs["item_id"][:],
            target_ids=row["target"]["item_id"][
                row["target"]["label"].tolist()
            ].tolist(),
            top_k=self.config.top_k,
        )
        return {f"{stage}/{key}": value for key, value in metrics.items()}

    def training_step(self, batch: SeqBatch) -> torch.Tensor:
        loss_dict = self.compute_losses(batch)
        self.log_dict(loss_dict)
        return loss_dict[f"loss/{self.config.train_loss}"]

    def validation_step(
        self, row: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, torch.Tensor]:
        metrics = self.compute_metrics(row, stage="val")
        self.log_dict(metrics, batch_size=1)
        return metrics

    def test_step(
        self, row: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, torch.Tensor]:
        metrics = self.compute_metrics(row, stage="test")
        self.log_dict(metrics, batch_size=1)
        return metrics

    def predict_step(self, row: dict[str, dict[str, np.ndarray]]) -> datasets.Dataset:
        """Prediction step: return nearest-neighbour recommendations.

        Excludes history item ids from the returned results.
        """
        return self.recommend(
            row["history"]["item_id"].tolist(),
            top_k=self.config.top_k,
            exclude_item_ids=row["history"]["item_id"].tolist(),
        )

    def on_validation_start(self) -> None:
        """Hook executed at the start of validation to index data.

        Ensures both items and users Lance indexes are populated with the
        current embeddings before validation or prediction runs.
        """
        self.items_index.index_data(self.trainer.datamodule.items_dataset)
        self.users_index.index_data(self.trainer.datamodule.users_dataset)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def configure_callbacks(self) -> list[lp.Callback]:
        checkpoint = lp_callbacks.ModelCheckpoint(
            monitor=METRIC["name"], mode=METRIC["mode"]
        )
        early_stop = lp_callbacks.EarlyStopping(
            monitor=METRIC["name"], mode=METRIC["mode"]
        )
        return [checkpoint, early_stop]

    @property
    def example_input_array(self) -> tuple[torch.Tensor]:
        """Example input used by Lightning for shape inference.

        Returns a tensor of item indices on the module device representative
        of a small batch.
        """
        return (torch.as_tensor([[0], [1]], device=self.device),)

    def state_dict(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        """Return the module state dict with large embedding weights removed.

        The SeqEmbedded model stores a potentially large `model.embeddings`
        weight that is intentionally omitted from the Lightning checkpoint
        state to reduce checkpoint size; this helper removes that entry if
        present.
        """
        state_dict: dict[str, torch.Tensor] = super().state_dict(*args, **kwargs)
        state_dict.pop("model.embeddings.weight", None)
        return state_dict

    def save(self, path: str | pathlib.Path) -> None:
        """Persist the transformer and Lance DB index for this module.

        Args:
            path (str): Directory path where artifacts will be stored.
        """
        assert self.model is not None
        path = pathlib.Path(path)
        self.model.save(path / TRANSFORMER_PATH)
        self.items_index.save(path / LANCE_DB_PATH)


class LightningCLI:
    def __init__(
        self,
        lightning_module_cls: type[lp.LightningModule] = RecommenderLightningModule,
        data_module_cls: type[lp.LightningDataModule] = SeqDataModule,
        *,
        model_name: str = "xfmr_rec",
    ) -> None:
        """Initialize the CLI helper for Lightning experiments.

        Args:
            lightning_module_cls (type[lp.LightningModule]): The Lightning
                module class used for training/evaluation.
            data_module_cls (type[lp.LightningDataModule]): The data module
                class providing data loaders and dataset preparation.
            model_name (str, optional): Default name for experiment logs.
        """
        self.lightning_module_cls = lightning_module_cls
        self.data_module_cls = data_module_cls
        self.model_name = model_name

    def main(
        self, args: lp_cli.ArgsType = None, *, run: bool = True, log_model: bool = True
    ) -> lp_cli.LightningCLI:
        """Create and return a configured ``LightningCLI`` instance.

        This method prepares default loggers (TensorBoard and MLflow) and a
        small set of trainer defaults (precision, callbacks, epoch/time
        limits). If there is an active MLflow run, its experiment name,
        run name and run id will be used to populate the loggers so that the
        CLI run is recorded under the active run.

        Args:
            args (lp_cli.ArgsType, optional): Optional overrides for the CLI
                arguments.
            run (bool, optional): Whether to immediately run the CLI.
            log_model (bool, optional): Whether the MLflow logger should
                track model artifacts. Defaults to True.

        Returns:
            lp_cli.LightningCLI: The constructed and optionally-run
            LightningCLI instance.
        """
        import mlflow

        model_name = self.model_name
        run_name = time_now_isoformat()
        run_id = None
        if active_run := mlflow.active_run():
            model_name = mlflow.get_experiment(active_run.info.experiment_id).name
            run_name = active_run.info.run_name
            run_id = active_run.info.run_id

        tensorboard_logger = {
            "class_path": "TensorBoardLogger",
            "init_args": {
                "save_dir": "lightning_logs",
                "name": model_name,
                "version": run_name,
                "default_hp_metric": False,
            },
        }
        mlflow_logger = {
            "class_path": "MLFlowLogger",
            "init_args": {
                "experiment_name": model_name,
                "run_name": run_name,
                "run_id": run_id,
                "log_model": log_model,
            },
        }

        trainer_defaults = {
            "precision": "bf16-mixed",
            "logger": [tensorboard_logger, mlflow_logger],
            "max_epochs": 1,
            "max_time": "00:04:00:00",
            "num_sanity_val_steps": 0,
        }
        return lp_cli.LightningCLI(
            self.lightning_module_cls,
            self.data_module_cls,
            save_config_callback=LoggerSaveConfigCallback,
            trainer_defaults=trainer_defaults,
            args=args,
            run=run,
        )

    def load_args(self, ckpt_path: str) -> dict[str, Any]:
        """Load configuration mappings from a Lightning checkpoint."""
        if not ckpt_path:
            return {"data": {"config": {"num_workers": 0}}}

        datamodule = self.data_module_cls.load_from_checkpoint(ckpt_path)
        model = self.lightning_module_cls.load_from_checkpoint(ckpt_path)
        return {
            "data": {"config": datamodule.config.model_dump()},
            "model": {"config": model.config.model_dump()},
        }

    def prepare_trainer(
        self, ckpt_path: str = "", stage: str = "validate", fast_dev_run: int = 0
    ) -> lp.Trainer:
        """Create and configure a Lightning Trainer optionally from a checkpoint."""
        if not ckpt_path:
            args = {"trainer": {"fast_dev_run": True, "accelerator": "cpu"}}
            return self.main({"fit": args}).trainer

        with tempfile.TemporaryDirectory() as tmp:
            trainer_args = {
                "accelerator": "cpu",
                "logger": False,
                "fast_dev_run": fast_dev_run,
                "enable_checkpointing": False,
                "default_root_dir": tmp,
            }
            args = {
                "trainer": trainer_args,
                "ckpt_path": ckpt_path,
                **self.load_args(ckpt_path),
            }
            return self.main({stage: args}).trainer

    def save_model(self, trainer: lp.Trainer) -> None:
        """Save the Lightning module and its artifacts into the BentoML store."""
        with bentoml.models.create(self.model_name) as model_ref:
            trainer.model.save(model_ref.path)


def main() -> None:
    """Module entrypoint to run experiments via LightningCLI."""
    LightningCLI().main()


if __name__ == "__main__":
    import rich

    from xfmr_rec.data import SeqDataModuleConfig

    datamodule = SeqDataModule(SeqDataModuleConfig())
    datamodule.prepare_data()
    datamodule.setup()
    model = RecommenderLightningModule(LightningConfig())
    model.items_dataset = datamodule.items_dataset
    model.configure_model()

    # train
    rich.print(model(*model.example_input_array))
    rich.print(model.compute_losses(next(iter(datamodule.train_dataloader()))))

    # validate
    assert datamodule.items_dataset is not None
    model.items_index.index_data(datamodule.items_dataset)
    rich.print(model.compute_metrics(next(iter(datamodule.val_dataloader())), "val"))

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer_args = {
            "accelerator": "cpu",
            "logger": False,
            "max_epochs": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "limit_test_batches": 1,
            "limit_predict_batches": 1,
            "default_root_dir": tmpdir,
        }
        data_args = {"config": {"num_workers": 0}}
        args = {"trainer": trainer_args, "data": data_args}

        cli = LightningCLI().main(args={"fit": args})
        rich.print(cli.trainer.validate(ckpt_path="best", datamodule=cli.datamodule)[0])
        rich.print(cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)[0])
        rich.print(cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)[0])

        ckpt_path = next(pathlib.Path(tmpdir).glob("**/*.ckpt"))
        model = RecommenderLightningModule.load_from_checkpoint(ckpt_path)
        rich.print(model)
