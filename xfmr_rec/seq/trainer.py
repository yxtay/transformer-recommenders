from __future__ import annotations

import pathlib
import shutil
from typing import TYPE_CHECKING

import lightning as lp
import lightning.pytorch.callbacks as lp_callbacks
import lightning.pytorch.cli as lp_cli
import lightning.pytorch.loggers as lp_loggers
import mlflow
import numpy as np
import torch

from xfmr_rec import losses
from xfmr_rec.common.trainer import LoggerSaveConfigCallback, time_now_isoformat
from xfmr_rec.index import LanceIndex, LanceIndexConfig
from xfmr_rec.losses import LossConfig, LossType
from xfmr_rec.metrics import compute_retrieval_metrics
from xfmr_rec.params import (
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    METRIC,
    TENSORBOARD_DIR,
    TOP_K,
    USERS_TABLE_NAME,
)
from xfmr_rec.seq import MODEL_NAME
from xfmr_rec.seq.data import SeqDataModule, SeqDataModuleConfig
from xfmr_rec.seq.models import SeqRecModel, SeqRecModelConfig

if TYPE_CHECKING:
    import datasets


class SeqRecLightningConfig(SeqRecModelConfig, LossConfig):
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
        super().__init__()
        self.config = SeqRecLightningConfig.model_validate(config)
        self.save_hyperparameters(self.config.model_dump())

        self.model: SeqRecModel | None = None
        self.loss_fns: torch.nn.ModuleList | None = None
        self.items_index = LanceIndex(config=self.config.items_config)
        self.users_index = LanceIndex(config=self.config.users_config)

    def configure_model(self) -> None:
        if self.model is None:
            self.model = SeqRecModel(config=self.config, device=self.device)
            # self.model.compile()

        if self.loss_fns is None:
            self.loss_fns = self.get_loss_fns()

    def get_loss_fns(self) -> torch.nn.ModuleList:
        loss_classes = [
            losses.AlignmentLoss,
            losses.AlignmentContrastiveLoss,
            losses.InfoNCELoss,
            losses.NCELoss,
            losses.PairwiseHingeLoss,
            losses.PairwiseLogisticLoss,
        ]
        loss_fns = [loss_class(config=self.config) for loss_class in loss_classes]
        return torch.nn.ModuleList(loss_fns)

    @torch.inference_mode()
    def index_items(self, items_dataset: datasets.Dataset) -> None:
        item_embeddings = items_dataset.map(
            lambda batch: {"embedding": self.model.embed_item_text(batch["item_text"])},
            batched=True,
            batch_size=32,
        )
        self.items_index.index_data(item_embeddings)

    def forward(self, item_texts: list[list[str]]) -> dict[str, torch.Tensor]:
        return self.model(item_texts)

    @torch.inference_mode()
    def recommend(
        self,
        item_text: list[str],
        *,
        top_k: int = 0,
        exclude_item_ids: list[str] | None = None,
    ) -> datasets.Dataset:
        embedding = self([item_text])["sentence_embedding"].numpy(force=True)
        return self.items_index.search(
            embedding,
            exclude_item_ids=exclude_item_ids,
            top_k=top_k or self.config.top_k,
        )

    def compute_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embeds = self.model.compute_embeds(
            batch["history_item_text"],
            batch["pos_item_text"],
            batch["neg_item_text"],
        )

        attention_mask = embeds["attention_mask"]
        batch_size, seq_len = attention_mask.size()
        numel = attention_mask.numel()
        non_zero = embeds["anchor_embed"].size(0)
        metrics = {
            "batch/size": batch_size,
            "batch/seq_len": seq_len,
            "batch/numel": numel,
            "batch/non_zero": non_zero,
            "batch/sparsity": non_zero / (numel + 1e-9),
        }

        losses = {}
        for loss_fn in self.loss_fns:
            key = f"loss/{loss_fn.__class__.__name__}"
            loss = loss_fn(
                anchor_embed=embeds["anchor_embed"],
                pos_embed=embeds["pos_embed"],
                neg_embed=embeds["neg_embed"],
            )
            losses[key] = loss
            losses[f"{key}Mean"] = loss / (non_zero + 1e-9)
        return losses | metrics

    def compute_metrics(
        self, row: dict[str, list[str]], stage: str = "val"
    ) -> dict[str, torch.Tensor]:
        recs = self.predict_step(row)
        metrics = compute_retrieval_metrics(
            rec_ids=recs["item_id"][:],
            target_ids=np.asarray(row["target"]["item_id"])[row["target"]["label"]],
            top_k=self.config.top_k,
        )
        return {f"{stage}/{key}": value for key, value in metrics.items()}

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        loss_dict = self.compute_losses(batch)
        self.log_dict(loss_dict)
        return loss_dict[f"loss/{self.config.train_loss}"]

    def validation_step(self, row: dict[str, list[str]]) -> dict[str, float]:
        metrics = self.compute_metrics(row, stage="val")
        self.log_dict(metrics, batch_size=1)
        return metrics

    def test_step(self, row: dict[str, list[str]]) -> dict[str, float]:
        metrics = self.compute_metrics(row, stage="test")
        self.log_dict(metrics, batch_size=1)
        return metrics

    def predict_step(self, row: dict[str, list[str]]) -> datasets.Dataset:
        return self.recommend(
            row["history"]["item_text"],
            top_k=self.config.top_k,
            exclude_item_ids=row["history"]["item_id"],
        )

    def on_train_start(self) -> None:
        params = self.hparams | self.trainer.datamodule.hparams
        metrics = {
            key: value
            for key, value in self.trainer.callback_metrics.items()
            if key.startswith("val/")
        }
        for logger in self.loggers:
            if isinstance(logger, lp_loggers.TensorBoardLogger):
                logger.log_hyperparams(params=params, metrics=metrics)

            if isinstance(logger, lp_loggers.MLFlowLogger):
                # reset mlflow run status to "RUNNING"
                logger.experiment.update_run(logger.run_id, status="RUNNING")

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
            monitor=METRIC["name"], mode=METRIC["mode"]
        )
        return [checkpoint, early_stop]

    @property
    def example_input_array(self) -> tuple[list[list[str]]]:
        return ([[""], []],)

    def save(self, path: str) -> None:
        path = pathlib.Path(path)
        self.model.save(path)

        lancedb_path = self.config.items_config.lancedb_path
        shutil.copytree(lancedb_path, path / LANCE_DB_PATH)


def cli_main(
    args: lp_cli.ArgsType = None, *, run: bool = True, log_model: bool = True
) -> lp_cli.LightningCLI:
    from jsonargparse import lazy_instance

    experiment_name = MODEL_NAME
    run_name = time_now_isoformat()
    run_id = None
    if active_run := mlflow.active_run():
        experiment_name = mlflow.get_experiment(active_run.info.experiment_id).name
        run_name = active_run.info.run_name
        run_id = active_run.info.run_id

    tensorboard_logger = {
        "class_path": "TensorBoardLogger",
        "init_args": {
            "save_dir": TENSORBOARD_DIR,
            "name": experiment_name,
            "version": run_name,
            # "log_graph": True,
            "default_hp_metric": False,
        },
    }
    mlflow_logger = {
        "class_path": "MLFlowLogger",
        "init_args": {
            "experiment_name": experiment_name,
            "run_name": run_name,
            "run_id": run_id,
            "log_model": log_model,
        },
    }

    progress_bar = lazy_instance(lp_callbacks.RichProgressBar)
    trainer_defaults = {
        "precision": "bf16-mixed",
        "logger": [tensorboard_logger, mlflow_logger],
        "callbacks": [progress_bar],
        "max_epochs": 1,
        "max_time": "00:04:00:00",
        "num_sanity_val_steps": 0,
    }
    return lp_cli.LightningCLI(
        SeqRecLightningModule,
        SeqDataModule,
        save_config_callback=LoggerSaveConfigCallback,
        trainer_defaults=trainer_defaults,
        args=args,
        run=run,
    )


def main() -> None:
    cli_main()


if __name__ == "__main__":
    import contextlib

    import rich

    datamodule = SeqDataModule(SeqDataModuleConfig())
    datamodule.prepare_data()
    datamodule.setup()
    model = SeqRecLightningModule(SeqRecLightningConfig())
    model.configure_model()

    # train
    rich.print(model(*model.example_input_array))
    rich.print(model.compute_losses(next(iter(datamodule.train_dataloader()))))

    # validate
    model.index_items(datamodule.items_dataset)
    rich.print(model.compute_metrics(next(iter(datamodule.val_dataloader())), "val"))

    trainer_args = {
        "accelerator": "cpu",
        "fast_dev_run": True,
        "max_epochs": 1,
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        # "overfit_batches": 1,
    }
    data_args = {"config": {"num_workers": 0}}
    cli = cli_main(args={"trainer": trainer_args, "data": data_args}, run=False)
    with contextlib.suppress(ReferenceError):
        # suppress weak reference on ModelCheckpoint callback
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        cli.trainer.validate(cli.model, datamodule=cli.datamodule)
        cli.trainer.test(cli.model, datamodule=cli.datamodule)
        cli.trainer.predict(cli.model, datamodule=cli.datamodule)
