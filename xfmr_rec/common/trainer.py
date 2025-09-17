import datetime
from typing import TYPE_CHECKING

import lightning.pytorch as lp
import lightning.pytorch.cli as lp_cli

if TYPE_CHECKING:
    from mlflow.tracking import MlflowClient


class LoggerSaveConfigCallback(lp_cli.SaveConfigCallback):
    def save_config(
        self,
        trainer: lp.Trainer,
        pl_module: lp.LightningModule,  # noqa: ARG002
        stage: str,  # noqa: ARG002
    ) -> None:
        import pathlib
        import tempfile

        import lightning.pytorch.loggers as lp_loggers

        for logger in trainer.loggers:
            if not isinstance(logger, lp_loggers.MLFlowLogger):
                continue

            trainer.logger.log_hyperparams(vars(self.config.as_flat()))
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
    datetime_now = datetime.datetime.now(datetime.UTC).astimezone()
    return datetime_now.isoformat(timespec="seconds")
