import datetime
import pathlib
import tempfile
from typing import TYPE_CHECKING

import lightning as lp
import lightning.pytorch.callbacks as lp_callbacks
import lightning.pytorch.cli as lp_cli
import lightning.pytorch.loggers as lp_loggers

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient


class LoggerSaveConfigCallback(lp_cli.SaveConfigCallback):
    def save_config(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,  # noqa: ARG002
        stage: str,  # noqa: ARG002
    ) -> None:
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
    datetime_now = datetime.datetime.now(datetime.UTC).astimezone()
    return datetime_now.isoformat(timespec="seconds")


class LightningCLI:
    def __init__(
        self,
        lightning_module_cls: type[lp.LightningModule],
        data_module_cls: type[lp.LightningDataModule],
        *,
        experiment_name: str = "",
    ) -> None:
        self.lightning_module_cls = lightning_module_cls
        self.data_module_cls = data_module_cls
        self.experiment_name = experiment_name

    def main(
        self, args: lp_cli.ArgsType = None, *, run: bool = True, log_model: bool = True
    ) -> lp_cli.LightningCLI:
        import mlflow
        from jsonargparse import lazy_instance

        experiment_name = self.experiment_name
        run_name = time_now_isoformat()
        run_id = None
        if active_run := mlflow.active_run():
            experiment_name = mlflow.get_experiment(active_run.info.experiment_id).name
            run_name = active_run.info.run_name
            run_id = active_run.info.run_id

        tensorboard_logger = {
            "class_path": "TensorBoardLogger",
            "init_args": {
                "save_dir": "lightning_logs",
                "name": experiment_name,
                "version": run_name,
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
            self.lightning_module_cls,
            self.data_module_cls,
            save_config_callback=LoggerSaveConfigCallback,
            trainer_defaults=trainer_defaults,
            args=args,
            run=run,
        )
