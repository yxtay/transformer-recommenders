from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import lightning as lp
import lightning.pytorch.callbacks as lp_callbacks
import torch
from loguru import logger

from xfmr_rec import losses as loss_classes
from xfmr_rec.index import LanceIndex, LanceIndexConfig
from xfmr_rec.losses import LOSS_CLASSES, LossConfig, LossType
from xfmr_rec.metrics import compute_retrieval_metrics
from xfmr_rec.params import (
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    METRIC,
    TOP_K,
    TRANSFORMER_PATH,
    USERS_TABLE_NAME,
)
from xfmr_rec.seq.data import SeqBatch, SeqDataModule, SeqDataModuleConfig
from xfmr_rec.seq_embedded import MODEL_NAME
from xfmr_rec.seq_embedded.models import SeqEmbeddedModel, SeqEmbeddedModelConfig
from xfmr_rec.trainer import LightningCLI

if TYPE_CHECKING:
    import datasets
    import numpy as np


class SeqEmbeddedLightningConfig(LossConfig, SeqEmbeddedModelConfig):
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


class SeqEmbeddedLightningModule(lp.LightningModule):
    def __init__(self, config: SeqEmbeddedLightningConfig) -> None:
        """Initialize the SeqEmbedded Lightning module.

        Args:
            config (SeqEmbeddedLightningConfig): Configuration dataclass
                containing model and training hyperparameters.
        """
        super().__init__()
        self.config = SeqEmbeddedLightningConfig.model_validate(config)
        self.save_hyperparameters(self.config.model_dump())
        self.strict_loading = False

        self.model: SeqEmbeddedModel | None = None
        self.items_dataset: datasets.Dataset | None = None
        self.loss_fns: torch.nn.ModuleList | None = None
        self.items_index = LanceIndex(self.config.items_config)
        self.users_index = LanceIndex(self.config.users_config)

        logger.info(repr(self.config))

    def configure_model(self) -> None:
        """Ensure the model and item embeddings are initialized.

        This will create the ``SeqEmbeddedModel`` instance if missing,
        attach the datamodule's items dataset when available, and call the
        model's ``configure_embeddings`` helper to prepare item embeddings.
        Also instantiates loss functions.
        """
        if self.model is None:
            self.model = SeqEmbeddedModel(self.config, device=self.device)

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
        """Compute losses and logging metrics for a SeqEmbedded batch.

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
        non_zero = attention_mask.count_nonzero().item()
        metrics = {
            "batch/size": batch_size,
            "batch/seq_len": seq_len,
            "batch/numel": numel,
            "batch/non_zero": non_zero,
            "batch/density": non_zero / (numel + 1e-9),
        }
        metrics |= loss_classes.LogitsStatistics(self.config)(
            query_embed=embeds["query_embed"],
            candidate_embed=embeds["candidate_embed"],
        )

        losses = {}
        for loss_fn in self.loss_fns:
            loss = loss_fn(
                query_embed=embeds["query_embed"],
                candidate_embed=embeds["candidate_embed"],
            )
            key = f"loss/{loss_fn.__class__.__name__}"
            losses[key] = loss
            losses[f"{key}Mean"] = loss / (non_zero + 1e-9)
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
        """Training step: compute and return the primary loss tensor.

        Args:
            batch (SeqBatch): Batch produced by the datamodule.

        Returns:
            torch.Tensor: The scalar loss used for backpropagation.
        """
        loss_dict = self.compute_losses(batch)
        self.log_dict(loss_dict)
        return loss_dict[f"loss/{self.config.train_loss}"]

    def validation_step(
        self, row: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, torch.Tensor]:
        """Validation step: compute retrieval metrics for a single row.

        Args:
            row (dict): A validation row containing history and target
                information as produced by the datamodule.

        Returns:
            dict[str, torch.Tensor]: Computed metrics mapped to tensors.
        """
        metrics = self.compute_metrics(row, stage="val")
        self.log_dict(metrics, batch_size=1)
        return metrics

    def test_step(
        self, row: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, torch.Tensor]:
        """Test step: compute retrieval metrics for a test row.

        Mirrors :meth:`validation_step` but uses the "test" metric prefix.
        """
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
        """Create and return the optimizer used for training.

        Returns:
            torch.optim.Optimizer: An AdamW optimizer configured using the
                module's learning rate and weight decay settings.
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def configure_callbacks(self) -> list[lp.Callback]:
        """Create and return common callbacks (checkpoint and early stop).

        Returns:
            list[lp.Callback]: Configured ModelCheckpoint and EarlyStopping
                callbacks monitoring the project metric.
        """
        checkpoint = lp_callbacks.ModelCheckpoint(
            monitor=METRIC["name"], mode=METRIC["mode"]
        )
        early_stop = lp_callbacks.EarlyStopping(
            monitor=METRIC["name"], mode=METRIC["mode"]
        )
        return [checkpoint, early_stop]

    # Duplicate method block removed: previous definitions above include
    # documentation and are the ones used by the module. This avoids
    # redefinition warnings from linters and keeps a single clear API.

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
        state_dict = super().state_dict(*args, **kwargs)
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


cli_main = LightningCLI(
    lightning_module_cls=SeqEmbeddedLightningModule,
    data_module_cls=SeqDataModule,
    model_name=MODEL_NAME,
).main


def main() -> None:
    """Module entrypoint to run seq-embedded experiments via LightningCLI.

    When executed as a script this function delegates control to the
    configured LightningCLI instance which will prepare the datamodule,
    model, loggers and then run the requested stage.
    """
    cli_main()


if __name__ == "__main__":
    import tempfile

    import rich

    datamodule = SeqDataModule(SeqDataModuleConfig())
    datamodule.prepare_data()
    datamodule.setup()
    model = SeqEmbeddedLightningModule(SeqEmbeddedLightningConfig())
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
            # "overfit_batches": 1,
            "default_root_dir": tmpdir,
        }
        data_args = {"config": {"num_workers": 0}}
        args = {"trainer": trainer_args, "data": data_args}

        cli = cli_main(args={"fit": args})
        rich.print(cli.trainer.validate(ckpt_path="best", datamodule=cli.datamodule)[0])
        rich.print(cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)[0])
        rich.print(cli.trainer.predict(ckpt_path="best", datamodule=cli.datamodule)[0])

        ckpt_path = next(pathlib.Path(tmpdir).glob("**/*.ckpt"))
        model = SeqEmbeddedLightningModule.load_from_checkpoint(ckpt_path)
        rich.print(model)
