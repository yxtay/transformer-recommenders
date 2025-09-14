from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import lightning.pytorch.callbacks as lp_callbacks
import lightning.pytorch.loggers as lp_loggers
import torch
from lightning import LightningModule
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback

from xfmr_rec.models import ModelConfig
from xfmr_rec.params import TOP_K

if TYPE_CHECKING:
    import pandas as pd
    import torchmetrics
    from lightning import Callback, Trainer
    from lightning.pytorch.cli import ArgsType
    from mlflow import MlflowClient
    from sentence_transformers import SentenceTransformer

    from xfmr_rec.data.lightning import (
        InteractionBatchType,
        ItemProcessor,
        UserFeaturesType,
        UserProcessor,
    )


class MatrixFactorizationLitConfig(ModelConfig):
    hidden_size: int = 32
    num_hidden_layers: int = 1
    num_attention_heads: int = 4
    intermediate_size: int = 32

    train_loss: str = "PairwiseHingeLoss"
    num_negatives: int = 4
    sigma: float = 1.0
    margin: float = 1.0
    learning_rate: float = 0.0001
    top_k: int = TOP_K


class MatrixFactorizationLitModule(LightningModule):
    def __init__(
        self,
        config: MatrixFactorizationLitConfig,
    ) -> None:
        super().__init__()
        self.config = MatrixFactorizationLitConfig.model_validate(config)
        self.save_hyperparameters(self.config.model_dump())
        self.model: SentenceTransformer | None = None
        self.loss_fns: torch.nn.ModuleList | None = None
        self.metrics: torch.nn.ModuleDict | None = None
        self.user_processor: UserProcessor | None = None
        self.item_processor: ItemProcessor | None = None

    def forward(self, text: list[str]) -> torch.Tensor:
        if self.model is None:
            msg = "`model` must be initialised first"
            raise ValueError(msg)

        # input shape: (batch_size,)
        tokens = self.model.tokenizer(
            text,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        ).to(self.device)
        # shape: (batch_size, seq_len)
        # output shape: (batch_size, embed_dim)
        return self.model(tokens)["sentence_embedding"]

    @torch.inference_mode()
    def recommend(
        self,
        text: list[str],
        *,
        top_k: int = TOP_K,
        user_id: int | None = None,
        exclude_item_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        if self.user_processor is None or self.item_processor is None:
            msg = "`user_processor` and `item_processor` must be initialised first"
            raise ValueError(msg)

        history = self.user_processor.get_activity(user_id, "history")
        exclude_item_ids = (exclude_item_ids or []) + list(history.keys())

        embed = self([text]).numpy(force=True)
        return self.item_processor.search(
            embed, exclude_item_ids=exclude_item_ids, top_k=top_k
        ).drop(columns="embedding")

    def compute_losses(
        self, batch: InteractionBatchType, step_name: str = "train"
    ) -> dict[str, torch.Tensor]:
        if self.loss_fns is None:
            msg = "`loss_fns` must be initialised first"
            raise ValueError(msg)

        target: torch.Tensor = batch["target"]
        # shape: (num_users, num_items)

        # user
        user: dict[str, torch.Tensor] = batch["user"]
        pos_idx = user["pos_idx"]
        # shape: (num_users,)
        user_text = user["text"]
        # shape: (num_users,)
        user_embed = self(user_text)
        # shape: (num_users, embed_dim)

        # item
        item: dict[str, torch.Tensor] = batch["item"]
        item_idx = item["idx"]
        # shape: (num_items,)
        item_text = item["text"]
        # shape: (num_items,)
        item_embed = self(item_text)
        # shape: (num_items, embed_dim)

        # neg item
        neg_item = batch["neg_item"]
        neg_item_idx = neg_item["idx"]
        # shape: (num_items,)
        neg_item_text = neg_item["text"]
        # shape: (num_items,)
        neg_item_embed = self(neg_item_text)
        # shape: (num_items, embed_dim)
        item_idx = torch.cat([item_idx, neg_item_idx])
        item_embed = torch.cat([item_embed, neg_item_embed])
        # shape: (num_items, embed_dim)

        losses = {}
        for loss_fn in self.loss_fns:
            key = f"{step_name}/{loss_fn.__class__.__name__}"
            losses[key] = loss_fn(
                user_embed=user_embed,
                item_embed=item_embed,
                target=target,
                item_idx=item_idx,
                pos_idx=pos_idx,
            )
        return losses

    def update_metrics(
        self, example: dict[str, torch.Tensor], step_name: str = "train"
    ) -> torchmetrics.MetricCollection:
        import torchmetrics.retrieval as tm_retrieval

        from xfmr_rec.params import TARGET_COL

        if self.metrics is None:
            msg = "`metrics` must be initialised first"
            raise ValueError(msg)

        item_id_col = self.item_processor.id_col
        user_idx_col = self.user_processor.idx_col

        pred_scores = self.predict_step(example, 0)
        pred_scores = dict(
            zip(pred_scores[item_id_col], pred_scores["score"], strict=True)
        )
        target_scores = example["target"]
        target_scores = dict(
            zip(target_scores[item_id_col], target_scores[TARGET_COL], strict=True)
        )

        item_ids = list(target_scores.keys() | pred_scores.keys())
        rand_scores = torch.rand(len(item_ids)).tolist()  # devskim: ignore DS148264
        preds = [
            pred_scores.get(item_id, -rand_scores[i])
            for i, item_id in enumerate(item_ids)
        ]
        preds = torch.as_tensor(preds)
        target = [target_scores.get(item_id, 0) for item_id in item_ids]
        target = torch.as_tensor(target)
        indexes = torch.ones_like(preds, dtype=torch.long) * example[user_idx_col]

        metrics: torchmetrics.MetricCollection = self.metrics[step_name]
        for metric in metrics.values():
            if isinstance(metric, tm_retrieval.RetrievalNormalizedDCG):
                metric.update(preds=preds, target=target, indexes=indexes)
            else:
                metric.update(preds=preds, target=target > 0, indexes=indexes)
        return metrics

    def training_step(self, batch: InteractionBatchType, _: int) -> torch.Tensor:
        losses = self.compute_losses(batch, step_name="train")
        self.log_dict(losses)
        return losses[f"train/{self.config.train_loss}"]

    def validation_step(self, batch: UserFeaturesType, _: int) -> None:
        metrics = self.update_metrics(batch, step_name="val")
        self.log_dict(metrics)

    def test_step(self, batch: UserFeaturesType, _: int) -> None:  # noqa: PT019
        metrics = self.update_metrics(batch, step_name="test")
        self.log_dict(metrics)

    def predict_step(self, batch: UserFeaturesType, _: int) -> pd.DataFrame:
        user_id_col = self.user_processor.id_col
        return self.recommend(
            batch["text"], top_k=self.config.top_k, user_id=batch[user_id_col]
        )

    def on_train_start(self) -> None:
        if self.metrics is None:
            msg = "`metrics` must be initialised first"
            raise ValueError(msg)

        params = self.hparams | self.trainer.datamodule.hparams
        metrics = {
            key: self.trainer.callback_metrics.get(key, 0.0)
            for key in self.metrics["val"]
        }
        for logger in self.loggers:
            if isinstance(logger, lp_loggers.TensorBoardLogger):
                logger.log_hyperparams(params=params, metrics=metrics)

            if isinstance(logger, lp_loggers.MLFlowLogger):
                # reset mlflow run status to "RUNNING"
                logger.experiment.update_run(logger.run_id, status="RUNNING")

    def on_validation_start(self) -> None:
        self.user_processor = self.trainer.datamodule.user_processor
        self.user_processor.get_index()
        self.item_processor = self.trainer.datamodule.item_processor
        self.item_processor.get_index(self)

    def on_test_start(self) -> None:
        self.on_validation_start()

    def on_predict_start(self) -> None:
        self.on_validation_start()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)

    def configure_callbacks(self) -> list[Callback]:
        from xfmr_rec.params import METRIC

        checkpoint = lp_callbacks.ModelCheckpoint(
            monitor=METRIC["name"], mode=METRIC["mode"]
        )
        early_stop = lp_callbacks.EarlyStopping(
            monitor=METRIC["name"], mode=METRIC["mode"], min_delta=0.001
        )
        return [checkpoint, early_stop]

    def configure_model(self) -> None:
        if self.model is None:
            self.model = self.get_model()
            # self.compile()
        if self.loss_fns is None:
            self.loss_fns = self.get_loss_fns()
        if self.metrics is None:
            self.metrics = self.get_metrics()

    def get_model(self) -> torch.nn.Module:
        from xfmr_rec.models import init_bert, to_sentence_transformer

        bert_model = init_bert(self.config)
        return to_sentence_transformer(bert_model)

    def get_loss_fns(self) -> torch.nn.ModuleList:
        import xfmr_rec.losses as mf_losses

        loss_classes = [
            mf_losses.AlignmentLoss,
            mf_losses.ContrastiveLoss,
            mf_losses.AlignmentContrastiveLoss,
            mf_losses.InfomationNoiseContrastiveEstimationLoss,
            mf_losses.MutualInformationNeuralEstimationLoss,
            mf_losses.PairwiseHingeLoss,
            mf_losses.PairwiseLogisticLoss,
        ]
        loss_fns = [
            loss_class(
                num_negatives=self.config.num_negatives,
                sigma=self.config.sigma,
                margin=self.config.margin,
            )
            for loss_class in loss_classes
        ]
        return torch.nn.ModuleList(loss_fns)

    def get_metrics(self) -> torch.nn.ModuleDict:
        import torchmetrics
        import torchmetrics.retrieval as tm_retrieval

        top_k = self.config.top_k
        metrics = {
            step_name: torchmetrics.MetricCollection(
                tm_retrieval.RetrievalNormalizedDCG(top_k=top_k),
                tm_retrieval.RetrievalRecall(top_k=top_k),
                tm_retrieval.RetrievalPrecision(top_k=top_k),
                tm_retrieval.RetrievalMAP(top_k=top_k),
                tm_retrieval.RetrievalHitRate(top_k=top_k),
                tm_retrieval.RetrievalMRR(top_k=top_k),
                prefix=f"{step_name}/",
            )
            for step_name in ["val", "test"]
        }
        return torch.nn.ModuleDict(metrics)

    @property
    def example_input_array(self) -> tuple[list[str]]:
        return (["", "{}"],)

    def save(self, path: str | pathlib.Path) -> None:
        import json
        import shutil

        from xfmr_rec.params import LANCE_DB_PATH, PROCESSORS_JSON, TRANSFORMER_PATH

        path = pathlib.Path(path)
        self.model.save_pretrained((path / TRANSFORMER_PATH).as_posix())

        processors_args = {
            "users": self.user_processor.model_dump(),
            "items": self.item_processor.model_dump(),
        }
        (path / PROCESSORS_JSON).write_text(json.dumps(processors_args, indent=2))

        lance_db_path = self.item_processor.lance_db_path
        shutil.copytree(lance_db_path, path / LANCE_DB_PATH)


class LoggerSaveConfigCallback(SaveConfigCallback):
    @rank_zero_only
    def save_config(
        self,
        trainer: Trainer,
        pl_module: LightningModule,  # noqa: ARG002
        stage: str,  # noqa: ARG002
    ) -> None:
        import tempfile

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
    args: ArgsType = None,
    *,
    run: bool = True,
    experiment_name: str = time_now_isoformat(),
    run_name: str = "",
    log_model: bool = True,
) -> LightningCLI:
    from jsonargparse import lazy_instance

    from xfmr_rec.data.lightning import MatrixFactorizationDataModule
    from xfmr_rec.params import MLFLOW_DIR, TENSORBOARD_DIR

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
        "max_time": "01:00:00:00",
        "val_check_interval": 1 / 4,
        "num_sanity_val_steps": 0,
    }
    return LightningCLI(
        MatrixFactorizationLitModule,
        MatrixFactorizationDataModule,
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

    from xfmr_rec.data.lightning import (
        MatrixFactorizationDataConfig,
        MatrixFactorizationDataModule,
    )

    datamodule = MatrixFactorizationDataModule(MatrixFactorizationDataConfig())
    datamodule.prepare_data()
    datamodule.setup("fit")
    model = MatrixFactorizationLitModule(MatrixFactorizationLitConfig())
    model.configure_model()

    # train
    rich.print(model(*model.example_input_array))
    rich.print(model.compute_losses(next(iter(datamodule.train_dataloader()))))

    # validate
    model.user_processor = datamodule.user_processor
    model.user_processor.get_index()
    model.item_processor = datamodule.item_processor
    model.item_processor.get_index(model)
    rich.print(
        model.update_metrics(next(iter(datamodule.val_dataloader())), "val").compute()
    )

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
