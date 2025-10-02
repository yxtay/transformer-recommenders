import datetime
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any

import bentoml
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


class LightningCLI:
    def __init__(
        self,
        lightning_module_cls: type[lp.LightningModule],
        data_module_cls: type[lp.LightningDataModule],
        *,
        model_name: str = "",
    ) -> None:
        """Initialize the CLI helper for Lightning experiments.

        Args:
            lightning_module_cls (type[lp.LightningModule]): The Lightning
                module class used for training/evaluation. The class should
                be compatible with ``lightning.pytorch.cli.LightningCLI``.
            data_module_cls (type[lp.LightningDataModule]): The data module
                class providing data loaders and dataset preparation.
            model_name (str, optional): Default name for experiment logs
                and TensorBoard/MLflow grouping. If an active MLflow run is
                detected at runtime, its experiment name will override this
                value.
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
                arguments (e.g., a dict matching the structure expected by
                ``LightningCLI``). Default is ``None`` which lets
                ``LightningCLI`` parse from sys.argv.
            run (bool, optional): Whether to immediately run the CLI (i.e.
                execute training). Passed to ``LightningCLI`` unchanged.
            log_model (bool, optional): Whether the MLflow logger should
                track model artifacts. Defaults to True.

        Returns:
            lp_cli.LightningCLI: The constructed and optionally-run
            LightningCLI instance.
        """

        import mlflow
        from jsonargparse import lazy_instance

        experiment_name = self.model_name
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

    def load_args(self, ckpt_path: str) -> dict[str, Any]:
        """Load configuration mappings from a Lightning checkpoint.

        When `ckpt_path` is empty a minimal default configuration mapping is
        returned (useful for fast-dev runs). Otherwise the datamodule and
        lightning module are reconstructed from the checkpoint and their
        serialized configs are returned in a dict suitable for passing into
        the JSONArgparse-based CLI.

        Args:
            ckpt_path: Path to a Lightning checkpoint or empty string.

        Returns:
            A dictionary with `data` and `model` config mappings.
        """

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
        """Create and configure a Lightning Trainer optionally from a checkpoint.

        The behavior mirrors other deploy helpers: return a fast-dev-run trainer
        when no checkpoint is given, otherwise reconstruct configuration from the
        checkpoint and create a CPU-only trainer suitable for validation.

        Args:
            ckpt_path: Optional checkpoint path.
            stage: CLI stage to run (e.g., 'validate').
            fast_dev_run: Passed to the Trainer to control a fast run.

        Returns:
            A configured `Trainer` instance.
        """

        if not ckpt_path:
            args = {"trainer": {"fast_dev_run": True}}
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
        """Save the Lightning module and its artifacts into the BentoML store.

        Creates a BentoML model entry (using `self.model_name`) and instructs
        the Lightning module attached to the trainer to persist its artifacts
        into the Bento model path.

        Args:
            trainer: PyTorch Lightning Trainer holding the trained model.
        """

        with bentoml.models.create(self.model_name) as model_ref:
            trainer.model.save(model_ref.path)
