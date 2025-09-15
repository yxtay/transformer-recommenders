from __future__ import annotations

import pathlib
import tempfile
from typing import TYPE_CHECKING, Literal

import lightning as lp
import lightning.pytorch.callbacks as lp_callbacks
import lightning.pytorch.cli as lp_cli
import lightning.pytorch.loggers as lp_loggers
import torch

from xfmr_rec.seq.models import ItemsIndex, SeqRecModel, SeqRecModelConfig

if TYPE_CHECKING:
    import datasets
    from mlflow.tracking import MlflowClient

METRIC = {"name": "val/retrieval_normalized_dcg", "mode": "max"}


class SeqRecLightningConfig(SeqRecModelConfig):
    train_loss: Literal["cross_entropy", "binary_cross_entropy"] = "cross_entropy"
    learning_rate: float = 0.001
    weight_decay: float = 0.01

    lancedb_path: str = "lance_db"
    table_name: str = "items"
    top_k: int = 20


def compute_retrieval_metrics(
    rec_ids: list[str], target_ids: list[str], top_k: int
) -> dict[str, torch.Tensor]:
    import torchmetrics.functional.retrieval as tm_retrieval

    if len(rec_ids) == 0:
        return torch.as_tensor(0.0)

    top_k = min(top_k, len(rec_ids))
    target_ids = set(target_ids)
    # rec_ids first, followed by target_ids at the end
    all_items = rec_ids + list(target_ids - set(rec_ids))
    preds = torch.linspace(1, 0, len(all_items))
    target = torch.as_tensor([item in target_ids for item in all_items])

    return {
        metric_fn.__name__: metric_fn(preds=preds, target=target, top_k=top_k)
        for metric_fn in [
            tm_retrieval.retrieval_auroc,
            tm_retrieval.retrieval_average_precision,
            tm_retrieval.retrieval_hit_rate,
            tm_retrieval.retrieval_normalized_dcg,
            tm_retrieval.retrieval_precision,
            tm_retrieval.retrieval_recall,
            tm_retrieval.retrieval_reciprocal_rank,
        ]
    }


class SeqRecLightningModule(lp.LightningModule):
    def __init__(self, config: SeqRecLightningConfig) -> None:
        super().__init__()
        self.config = SeqRecLightningConfig.model_validate(config)
        self.save_hyperparameters(self.config.model_dump())

        self.model: SeqRecModel | None = None
        self.items_index: ItemsIndex | None = None

    def configure_model(self) -> None:
        if self.model is None:
            self.model = SeqRecModel(config=self.config, device=self.device)

    def configure_index(self, items_dataset: datasets.Dataset) -> ItemsIndex:
        item_embeddings = items_dataset.add_column(
            "embedding",
            self.model.embed_items(items_dataset["item_text"]).tolist(),
        )
        return ItemsIndex(
            item_embeddings,
            lancedb_path=self.config.lancedb_path,
            table_name=self.config.table_name,
        )

    def forward(self, item_texts: list[list[str]]) -> dict[str, torch.Tensor]:
        return self.model(item_texts)

    @torch.inference_mode()
    def recommend(
        self,
        item_text: list[str],
        *,
        top_k: int = 0,
        exclude_item_ids: list[int] | None = None,
    ) -> datasets.Dataset:
        if self.items_index is None:
            msg = "`items_index` must be initialised first"
            raise ValueError(msg)

        embedding = self.model([item_text])["sentence_embedding"].numpy(force=True)
        return self.items_index.search(
            embedding,
            exclude_item_ids=exclude_item_ids,
            top_k=top_k or self.config.top_k,
        )

    def compute_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model.compute_loss(
            batch["history_item_text"], batch["pos_item_text"], batch["neg_item_text"]
        )

    def compute_metrics(
        self, row: dict[str, list[str]], stage: str = "val"
    ) -> dict[str, torch.Tensor]:
        recs = self.predict_step(row)
        metrics = compute_retrieval_metrics(
            rec_ids=recs["item_id"][:],
            target_ids=row["target.item_id"],
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
            row["history.item_text"],
            top_k=self.config.top_k,
            exclude_item_ids=row["history.item_id"],
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

    @torch.inference_mode()
    def on_validation_start(self) -> None:
        self.items_index = self.configure_index(self.trainer.datamodule.items_dataset)

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
    def example_input_array(self) -> tuple[torch.Tensor]:
        return ([[""], []],)


class LoggerSaveConfigCallback(lp_cli.SaveConfigCallback):
    def save_config(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,  # noqa: ARG002
        stage: str,  # noqa: ARG002
    ) -> None:
        for logger in trainer.loggers:
            if not isinstance(logger, lp_loggers.MLFlowLogger):
                continue

            with tempfile.TemporaryDirectory() as path:
                config_path = pathlib.Path(path, self.config_filename)
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=False,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )
                mlflow_client: MlflowClient = logger.experiment
                mlflow_client.log_artifact(run_id=logger.run_id, local_path=config_path)


def time_now_isoformat() -> str:
    import datetime

    datetime_now = datetime.datetime.now(datetime.UTC).astimezone()
    return datetime_now.isoformat(timespec="seconds")


def cli_main(
    args: lp_cli.ArgsType = None,
    *,
    run: bool = True,
    experiment_name: str = time_now_isoformat(),
    run_name: str = "",
    log_model: bool = True,
) -> lp_cli.LightningCLI:
    from jsonargparse import lazy_instance

    from xfmr_rec.params import MLFLOW_DIR, TENSORBOARD_DIR
    from xfmr_rec.seq.data import SeqDataModule

    run_name = run_name or time_now_isoformat()
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
            "save_dir": MLFLOW_DIR,
            "experiment_name": experiment_name,
            "run_name": run_name,
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

    from xfmr_rec.seq.data import SeqDataModule, SeqDataModuleConfig

    datamodule = SeqDataModule(SeqDataModuleConfig())
    datamodule.prepare_data()
    datamodule.setup()
    model = SeqRecLightningModule(SeqRecLightningConfig())
    model.configure_model()

    # train
    rich.print(model(*model.example_input_array))
    rich.print(model.compute_losses(next(iter(datamodule.train_dataloader()))))

    # validate
    model.items_index = model.configure_index(datamodule.items_dataset)
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
