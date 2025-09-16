from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import ray.train
import ray.tune

from xfmr_rec.params import METRIC

if TYPE_CHECKING:
    from lightning import Trainer
    from lightning.pytorch.cli import LightningCLI


def get_cli(config: dict[str, bool | float | int | str]) -> LightningCLI:
    from xfmr_rec.flaml import get_lightning_args
    from xfmr_rec.mf.trainer import cli_main

    experiment_name = ray.tune.get_context().get_experiment_name()
    trial_name = ray.tune.get_context().get_trial_name()
    tensorboard_logger = {
        "class_path": "TensorBoardLogger",
        "init_args": {
            "save_dir": config["tensorboard_save_dir"],
            "name": experiment_name,
            "version": trial_name,
            "default_hp_metric": False,
        },
    }
    mlflow_logger = {
        "class_path": "MLFlowLogger",
        "init_args": {
            "save_dir": config["mlflow_save_dir"],
            "experiment_name": experiment_name,
            "run_name": trial_name,
            "log_model": True,
        },
    }
    strategy = {"class_path": "ray.train.lightning.RayDDPStrategy"}
    callbacks = [{"class_path": "ray.train.lightning.RayTrainReportCallback"}]
    plugins = [{"class_path": "ray.train.lightning.RayLightningEnvironment"}]
    trainer_args = {
        "enable_model_summary": False,
        "strategy": strategy,
        "logger": [tensorboard_logger, mlflow_logger],
        "callbacks": callbacks,
        "plugins": plugins,
    }
    args = {"trainer": trainer_args, **get_lightning_args(config)}
    return cli_main(args, run=False, log_model=False)


def train_loop_per_worker(
    config: dict[str, bool | float | int | str],
) -> None:
    import numpy as np
    import ray.train.lightning as ray_lightning

    ckpt_path = None
    if checkpoint := ray.train.get_checkpoint():
        checkpoint_name = ray_lightning.RayTrainReportCallback.CHECKPOINT_NAME
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = pathlib.Path(ckpt_dir, checkpoint_name)

    config = {
        key: value.tolist() if isinstance(value, np.generic) else value
        for key, value in config.items()
    }

    cli = get_cli(config)
    trainer: Trainer = ray_lightning.prepare_trainer(cli.trainer)
    try:
        trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
    except SystemExit:
        for logger in trainer.loggers:
            logger.finalize()
        raise


def train_driver_fn(
    config: dict[str, bool | float | int | str] | None = None,
) -> None:
    import os

    import ray.train.torch as ray_torch

    from xfmr_rec.params import DATA_DIR, MLFLOW_DIR, TENSORBOARD_DIR

    config = config or {}
    train_loop_config = {
        # tracking
        "tensorboard_save_dir": pathlib.Path(TENSORBOARD_DIR).absolute().as_posix(),
        "mlflow_save_dir": pathlib.Path(MLFLOW_DIR).absolute().as_posix(),
        # datamodule
        "data_dir": pathlib.Path(DATA_DIR).absolute().as_posix(),
        "num_hashes": 2,
        # lightning module
        "train_loss": "PairwiseHingeLoss",
        "log_num_negatives": 0,
        "sigma": 1.0,
        "margin": 1.0,
        "learning_rate": 0.1,
    } | config.get("train_loop_config", {})
    scaling_config = ray.train.ScalingConfig(
        num_workers=1,
        resources_per_worker={"CPU": os.cpu_count() - 1},
    )
    checkpoint_config = ray.train.CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute=METRIC["name"],
        checkpoint_score_order=METRIC["mode"],
    )
    run_config = ray.train.RunConfig(
        storage_path=pathlib.Path("ray_results").absolute(),
        checkpoint_config=checkpoint_config,
    )
    trainer = ray_torch.TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    trainer.fit()


def get_tuner() -> ray.tune.Tuner:
    import flaml
    import ray.tune.schedulers
    import ray.tune.stopper as ray_stopper

    train_losses = [
        "PairwiseHingeLoss",
        "PairwiseLogisticLoss",
        "AlignmentContrastiveLoss",
        "MutualInformationNeuralEstimationLoss",
    ]
    point_to_evaluate = {
        "train_loss": "PairwiseHingeLoss",
        "log_num_negatives": 0,
        "sigma": 1.0,
        "margin": 1.0,
        "learning_rate": 0.1,
    }
    search_space = point_to_evaluate | {
        "train_loss": ray.tune.choice(train_losses),
        "log_num_negatives": ray.tune.randint(0, 6),
        "sigma": ray.tune.lograndint(1, 1000),
        "margin": ray.tune.quniform(-1.0, 1.0, 0.01),
        "learning_rate": ray.tune.loguniform(0.001, 0.1),
    }
    low_cost_partial_config = {
        # "train_loss": "PairwiseHingeLoss",
        "log_num_negatives": 0,
        # "sigma": 1.0,
        # "margin": 1.0,
        # "learning_rate": 0.1,
    }
    search_alg = flaml.BlendSearch(
        low_cost_partial_config={"train_loop_config": low_cost_partial_config},
        # points_to_evaluate=[{"train_loop_config": point_to_evaluate}],
    )
    scheduler = ray.tune.schedulers.AsyncHyperBandScheduler()
    tune_config = ray.tune.TuneConfig(
        metric=METRIC["name"],
        mode=METRIC["mode"],
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=-1,
        time_budget_s=60 * 60 * 1,
        max_concurrent_trials=1,
    )
    checkpoint_config = ray.tune.CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute=METRIC["name"],
        checkpoint_score_order=METRIC["mode"],
    )
    stopper = ray_stopper.ExperimentPlateauStopper(
        metric=METRIC["name"], mode=METRIC["mode"]
    )
    run_config = ray.tune.RunConfig(
        storage_path=pathlib.Path("ray_results").absolute(),
        checkpoint_config=checkpoint_config,
        stop=stopper,
    )
    return ray.tune.Tuner(
        train_driver_fn,
        param_space={"train_loop_config": search_space},
        tune_config=tune_config,
        run_config=run_config,
    )


def main() -> None:
    import rich

    tuner = get_tuner()
    results = tuner.fit()
    rich.print(results.experiment_path)
    best_results = results.get_best_result()
    rich.print(best_results.path)
    rich.print(best_results.metrics)


if __name__ == "__main__":
    main()
