from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import lightning as lp
import lightning.pytorch.callbacks as lp_callbacks
import lightning.pytorch.loggers as lp_loggers
import pandas as pd
import torch
from loguru import logger

from xfmr_rec import losses as loss_classes
from xfmr_rec.index import LanceIndex, LanceIndexConfig
from xfmr_rec.losses import LOSS_CLASSES, LossConfig, LossType
from xfmr_rec.metrics import compute_retrieval_metrics
from xfmr_rec.mf import MODEL_NAME
from xfmr_rec.mf.data import MFDataModule, MFDataModuleConfig
from xfmr_rec.models import ModelConfig, init_sent_transformer
from xfmr_rec.params import (
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    METRIC,
    TOP_K,
    TRANSFORMER_PATH,
    USERS_TABLE_NAME,
)
from xfmr_rec.trainer import LightningCLI

if TYPE_CHECKING:
    import datasets
    from sentence_transformers import SentenceTransformer


class MFRecLightningConfig(LossConfig, ModelConfig):
    hidden_size: int = 32
    num_hidden_layers: int = 1
    num_attention_heads: int = 4
    intermediate_size: int = 32

    scale: float = 30
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


class MFRecLightningModule(lp.LightningModule):
    def __init__(self, config: MFRecLightningConfig) -> None:
        super().__init__()
        self.config = MFRecLightningConfig.model_validate(config)
        self.save_hyperparameters(self.config.model_dump())

        self.model: SentenceTransformer | None = None
        self.items_dataset: datasets.Dataset | None = None
        self.id2text: pd.Series | None = None
        self.loss_fns: torch.nn.ModuleList | None = None
        self.items_index = LanceIndex(self.config.items_config)
        self.users_index = LanceIndex(self.config.users_config)

        logger.info(repr(self.config))

    def configure_model(self) -> None:
        if self.model is None:
            self.model = init_sent_transformer(self.config, device=self.device)
            logger.info(self.model)

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
        loss_fns = [loss_class(self.config) for loss_class in LOSS_CLASSES]
        return torch.nn.ModuleList(loss_fns)

    @torch.inference_mode()
    def index_items(self, items_dataset: datasets.Dataset) -> None:
        item_embeddings = items_dataset.map(
            lambda batch: {"embedding": self.model.encode(batch["item_text"])},
            batched=True,
        )
        self.items_index.index_data(item_embeddings, overwrite=True)

    def forward(self, text: list[str]) -> dict[str, torch.Tensor]:
        tokenized = self.model.tokenize(text)
        tokenized = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in tokenized.items()
        }
        return self.model(tokenized)["sentence_embedding"]

    @torch.inference_mode()
    def recommend(
        self,
        query_text: str,
        *,
        top_k: int = 0,
        exclude_item_ids: list[str] | None = None,
    ) -> datasets.Dataset:
        embedding = self.model.encode(query_text)
        return self.items_index.search(
            embedding,
            exclude_item_ids=exclude_item_ids,
            top_k=top_k or self.config.top_k,
        )

    def compute_losses(self, batch: dict[str, list[str]]) -> dict[str, torch.Tensor]:
        query_embed = self(batch["query_text"])
        # shape: (batch_size, hidden_size)
        candidate_embed = self(batch["pos_text"] + batch["neg_text"])
        # shape: (2 * batch_size, hidden_size)
        candidate_embed = candidate_embed[None, :, :].expand(
            query_embed.size(0), -1, -1
        )
        # shape: (batch_size, 2 * batch_size, hidden_size)

        batch_size = query_embed.size(0)
        metrics = {"batch/size": batch_size}
        metrics |= loss_classes.LogitsStatistics(self.config)(
            query_embed=query_embed, candidate_embed=candidate_embed
        )

        losses = {}
        for loss_fn in self.loss_fns:
            loss = loss_fn(query_embed=query_embed, candidate_embed=candidate_embed)
            key = f"loss/{loss_fn.__class__.__name__}"
            losses[key] = loss
            losses[f"{key}Mean"] = loss / (batch_size + 1e-9)
        return losses | metrics

    def compute_metrics(
        self, row: dict[str, list[str]], stage: str = "val"
    ) -> dict[str, torch.Tensor]:
        recs = self.predict_step(row)
        metrics = compute_retrieval_metrics(
            rec_ids=recs["item_id"][:],
            target_ids=row["target"]["item_id"][
                row["target"]["label"].tolist()
            ].tolist(),
            top_k=self.config.top_k,
        )
        metrics = {f"{stage}/{key}": value for key, value in metrics.items()}

        try:
            item_id = next(
                item_id
                for item_id in reversed(row["history"]["item_id"].tolist())
                if item_id in self.id2text.index
            )
        except StopIteration:
            return metrics

        item_text = self.id2text[item_id]
        item_recs = self.recommend(
            item_text,
            top_k=self.config.top_k,
            exclude_item_ids=row["history"]["item_id"].tolist(),
        )
        item_metrics = compute_retrieval_metrics(
            rec_ids=item_recs["item_id"][:],
            target_ids=row["target"]["item_id"][
                row["target"]["label"].tolist()
            ].tolist(),
            top_k=self.config.top_k,
        )
        item_metrics = {
            f"{stage}/item/{key}": value for key, value in item_metrics.items()
        }
        return metrics | item_metrics

    def training_step(self, batch: dict[str, list[str]]) -> torch.Tensor:
        loss_dict = self.compute_losses(batch)
        self.log_dict(loss_dict)
        return loss_dict[f"loss/{self.config.train_loss}"]

    def validation_step(self, row: dict[str, str]) -> dict[str, float]:
        metrics = self.compute_metrics(row, stage="val")
        self.log_dict(metrics, batch_size=1)
        return metrics

    def test_step(self, row: dict[str, str]) -> dict[str, float]:
        metrics = self.compute_metrics(row, stage="test")
        self.log_dict(metrics, batch_size=1)
        return metrics

    def predict_step(self, row: dict[str, str]) -> datasets.Dataset:
        return self.recommend(
            row["user_text"],
            top_k=self.config.top_k,
            exclude_item_ids=row["history"]["item_id"].tolist(),
        )

    def on_train_start(self) -> None:
        params = self.hparams | self.trainer.datamodule.hparams
        metrics = {
            key: value
            for key, value in self.trainer.callback_metrics.items()
            if key.startswith("val/")
        }
        for lp_logger in self.loggers:
            if isinstance(lp_logger, lp_loggers.TensorBoardLogger):
                lp_logger.log_hyperparams(params=params, metrics=metrics)

            if isinstance(lp_logger, lp_loggers.MLFlowLogger):
                # reset mlflow run status to "RUNNING"
                lp_logger.experiment.update_run(lp_logger.run_id, status="RUNNING")

    def on_validation_start(self) -> None:
        self.index_items(self.trainer.datamodule.items_dataset)
        self.users_index.index_data(self.trainer.datamodule.users_dataset)

    def on_test_start(self) -> None:
        self.on_validation_start()

    def on_predict_start(self) -> None:
        self.on_validation_start()

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
            monitor=METRIC["name"], mode=METRIC["mode"], patience=7
        )
        return [checkpoint, early_stop]

    @property
    def example_input_array(self) -> tuple[list[str]]:
        return (["", ""],)

    def save(self, path: str) -> None:
        path = pathlib.Path(path)
        self.model.save(path / TRANSFORMER_PATH)
        self.items_index.save(path / LANCE_DB_PATH)


cli_main = LightningCLI(
    lightning_module_cls=MFRecLightningModule,
    data_module_cls=MFDataModule,
    experiment_name=MODEL_NAME,
).main


def main() -> None:
    cli_main()


if __name__ == "__main__":
    import tempfile

    import rich

    datamodule = MFDataModule(MFDataModuleConfig())
    datamodule.prepare_data()
    datamodule.setup()
    model = MFRecLightningModule(MFRecLightningConfig())
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
        model = MFRecLightningModule.load_from_checkpoint(ckpt_path)
        rich.print(model)
