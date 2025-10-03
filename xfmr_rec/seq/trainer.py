from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import lightning as lp
import lightning.pytorch.callbacks as lp_callbacks
import pandas as pd
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
    USERS_TABLE_NAME,
)
from xfmr_rec.seq import MODEL_NAME
from xfmr_rec.seq.data import SeqBatch, SeqDataModule, SeqDataModuleConfig
from xfmr_rec.seq.models import SeqRecModel, SeqRecModelConfig
from xfmr_rec.trainer import LightningCLI

if TYPE_CHECKING:
    import datasets
    import numpy as np


class SeqRecLightningConfig(LossConfig, SeqRecModelConfig):
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


class SeqRecLightningModule(lp.LightningModule):
    def __init__(self, config: SeqRecLightningConfig) -> None:
        """Initialize the sequential recommendation Lightning module.

        Args:
            config (SeqRecLightningConfig): Configuration dataclass for the
                sequential model and training hyperparameters.
        """
        super().__init__()
        self.config = SeqRecLightningConfig.model_validate(config)
        self.save_hyperparameters(self.config.model_dump())

        self.model: SeqRecModel | None = None
        self.items_dataset: datasets.Dataset | None = None
        self.id2text: pd.Series | None = None
        self.loss_fns: torch.nn.ModuleList | None = None
        self.items_index = LanceIndex(self.config.items_config)
        self.users_index = LanceIndex(self.config.users_config)

        logger.info(repr(self.config))

    def configure_model(self) -> None:
        """Lazily initialize the sequential model and related assets.

        Ensures the underlying ``SeqRecModel`` is created, attaches the
        datamodule's item dataset if available, and prepares the id-to-text
        mapping and loss functions.
        """
        if self.model is None:
            self.model = SeqRecModel(self.config, device=self.device)

        if self.items_dataset is None:
            try:
                self.items_dataset = self.trainer.datamodule.items_dataset
            except RuntimeError as e:
                # RuntimeError if trainer is not attached
                logger.warning(repr(e))

        if self.id2text is None and self.items_dataset is not None:
            self.id2text = pd.Series(
                self.items_dataset.with_format("pandas")["item_text"].array,
                index=self.items_dataset.with_format("pandas")["item_id"].array,
            )

        if self.loss_fns is None:
            self.loss_fns = self.get_loss_fns()

    def get_loss_fns(self) -> torch.nn.ModuleList:
        """Instantiate the list of loss functions to use during training.

        Returns:
            torch.nn.ModuleList: One instantiated loss module per class in
                ``LOSS_CLASSES``.
        """
        loss_fns = [loss_class(self.config) for loss_class in LOSS_CLASSES]
        return torch.nn.ModuleList(loss_fns)

    @torch.inference_mode()
    def index_items(self, items_dataset: datasets.Dataset) -> None:
        """Create item embeddings using the model and index them.

        Args:
            items_dataset (datasets.Dataset): Dataset of item records with
                an "item_text" column. The model's ``embed_item_text``
                method will be applied in batched mode to compute embeddings
                before writing to the Lance index.
        """
        assert self.model is not None
        item_embeddings = items_dataset.map(
            lambda batch: {"embedding": self.model.embed_item_text(batch["item_text"])},
            batched=True,
        )
        self.items_index.index_data(item_embeddings, overwrite=True)

    def forward(self, item_texts: list[list[str]]) -> dict[str, torch.Tensor]:
        """Compute model outputs for a batch of item text sequences.

        Args:
            item_texts (list[list[str]]): A batch where each element is a
                list of item text strings representing a user's history.

        Returns:
            dict[str, torch.Tensor]: Model outputs including ``sentence_embedding``
                and any other tensors produced by the sequence model.
        """
        assert self.model is not None
        return self.model(item_texts)

    @torch.inference_mode()
    def recommend(
        self,
        item_ids: list[str],
        *,
        top_k: int = 0,
        exclude_item_ids: list[str] | None = None,
    ) -> datasets.Dataset:
        """Return top-k recommendations for a list of item ids.

        The method looks up item text for the provided ids, computes a
        sentence embedding for the concatenated texts, and queries the
        items index to retrieve nearest neighbors.
        """
        assert self.id2text is not None
        item_ids = [item_id for item_id in item_ids if item_id in self.id2text.index]
        item_text = self.id2text[item_ids].tolist()
        embedding = self([item_text])["sentence_embedding"].numpy(force=True)
        return self.items_index.search(
            embedding,
            exclude_item_ids=exclude_item_ids,
            top_k=top_k or self.config.top_k,
        )

    def compute_losses(self, batch: SeqBatch) -> dict[str, torch.Tensor]:
        """Compute loss tensors and metrics for a sequence batch.

        This method computes query and candidate embeddings using the
        sequence model, gathers per-batch statistics (sequence lengths,
        density), computes logits statistics and evaluates each configured
        loss function. The returned dictionary contains per-loss tensors
        and logging metrics.

        Args:
            batch (SeqBatch): Batch from the Seq datamodule containing
                history/pos/neg item texts.

        Returns:
            dict[str, torch.Tensor]: Mapping of metric and loss names to
                tensors ready for logging.
        """
        assert self.model is not None
        assert self.loss_fns is not None
        embeds = self.model.compute_embeds(
            batch["history_item_text"],
            batch["pos_item_text"],
            batch["neg_item_text"],
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
        """Compute retrieval metrics for a given validation/test row.

        Args:
            row (dict): A row containing "history", "target" and other
                fields expected by the datamodule/predict pipeline.
            stage (str, optional): Metric name prefix. Defaults to "val".

        Returns:
            dict[str, torch.Tensor]: Mapping of metric names to tensors for
                logging.
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
        """Training step: compute losses and return primary loss.

        Args:
            batch (SeqBatch): Batch produced by the Seq datamodule.

        Returns:
            torch.Tensor: Primary scalar loss tensor for backprop.
        """
        loss_dict = self.compute_losses(batch)
        self.log_dict(loss_dict)
        return loss_dict[f"loss/{self.config.train_loss}"]

    def validation_step(
        self, row: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, torch.Tensor]:
        """Validation step: compute and log metrics for a single row.

        Args:
            row (dict): A validation row containing predictions/targets.

        Returns:
            dict[str, torch.Tensor]: Computed metrics for logging.
        """
        metrics = self.compute_metrics(row, stage="val")
        self.log_dict(metrics, batch_size=1)
        return metrics

    def test_step(
        self, row: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, torch.Tensor]:
        """Test step: compute and log metrics for test data rows.

        Mirrors :meth:`validation_step` but uses the "test" prefix for
        metrics.
        """
        metrics = self.compute_metrics(row, stage="test")
        self.log_dict(metrics, batch_size=1)
        return metrics

    def predict_step(self, row: dict[str, dict[str, np.ndarray]]) -> datasets.Dataset:
        """Prediction step used by Lightning's predict loop.

        Returns nearest-neighbour recommendations for the provided
        history item ids while excluding history items from results.
        """
        return self.recommend(
            row["history"]["item_id"].tolist(),
            top_k=self.config.top_k,
            exclude_item_ids=row["history"]["item_id"].tolist(),
        )

    def on_validation_start(self) -> None:
        """Hook executed before validation runs.

        Indexes item and user data into Lance indexes so prediction can use
        the latest embeddings.
        """
        self.index_items(self.trainer.datamodule.items_dataset)
        self.users_index.index_data(self.trainer.datamodule.users_dataset)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure and return optimizer (AdamW) for training.

        Returns:
            torch.optim.Optimizer: Configured optimizer instance.
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def configure_callbacks(self) -> list[lp.Callback]:
        """Create and return standard callbacks for model checkpointing
        and early stopping.

        Returns:
            list[lp.Callback]: Checkpoint and EarlyStopping callbacks.
        """
        checkpoint = lp_callbacks.ModelCheckpoint(
            monitor=METRIC["name"], mode=METRIC["mode"]
        )
        early_stop = lp_callbacks.EarlyStopping(
            monitor=METRIC["name"], mode=METRIC["mode"]
        )
        return [checkpoint, early_stop]

    @property
    def example_input_array(self) -> tuple[list[list[str]]]:
        """Example input array used for tracing/shape inference.

        Returns a tuple compatible with the model's forward signature.
        """
        return ([[], [""]],)

    def save(self, path: str | pathlib.Path) -> None:
        """Persist the sequence model and items index artifacts.

        Args:
            path (str): Destination directory for saved model and index.
        """
        assert self.model is not None
        path = pathlib.Path(path)
        self.model.save(path)
        self.items_index.save(path / LANCE_DB_PATH)


cli_main = LightningCLI(
    lightning_module_cls=SeqRecLightningModule,
    data_module_cls=SeqDataModule,
    model_name=MODEL_NAME,
).main


def main() -> None:
    """Script entry point for running sequential model experiments.

    Delegates to the preconfigured LightningCLI instance which will set up
    loggers, callbacks and execute the requested action (fit/validate/test/predict).
    """
    cli_main()


if __name__ == "__main__":
    import tempfile

    import rich

    datamodule = SeqDataModule(SeqDataModuleConfig())
    datamodule.prepare_data()
    datamodule.setup()
    model = SeqRecLightningModule(SeqRecLightningConfig())
    model.items_dataset = datamodule.items_dataset
    model.configure_model()

    # train
    rich.print(model(*model.example_input_array))
    rich.print(model.compute_losses(next(iter(datamodule.train_dataloader()))))

    # validate
    model.index_items(datamodule.items_dataset)
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
        model = SeqRecLightningModule.load_from_checkpoint(ckpt_path)
        rich.print(model)
