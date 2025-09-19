from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import flaml.tune.tune


def get_lightning_args(
    config: dict[str, bool | float | int | str],
) -> dict[str, bool | float | int | str]:
    num_negatives = 2 ** config["log_num_negatives"]

    model_args = {
        "train_loss": config["train_loss"],
        "num_negatives": num_negatives,
        "sigma": config["sigma"],
        "margin": config["margin"],
        "learning_rate": config["learning_rate"],
    }
    data_args = {
        "data_dir": config["data_dir"],
    }
    return {"model": model_args, "data": data_args}


def evaluation_function(
    config: dict[str, bool | float | int | str],
) -> dict[str, float]:
    import numpy as np

    from xfmr_rec.mf.trainer import cli_main

    config = {
        key: value.tolist() if isinstance(value, np.generic) else value
        for key, value in config.items()
    }

    trainer_args = {"limit_train_batches": config["limit_train_batches"]}
    args = {"trainer": trainer_args, **get_lightning_args(config)}
    cli = cli_main(args, run=False, log_model=False)
    try:
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)
        return {
            key: cli.trainer.callback_metrics[key].tolist()
            for key in cli.model.metrics["val"]
        }
    except (StopIteration, SystemExit):
        for logger in cli.trainer.loggers:
            logger.finalize()
        return {}


def flaml_tune() -> flaml.tune.tune.ExperimentAnalysis:
    import flaml.tune

    from xfmr_rec.params import DATA_DIR, METRIC

    train_losses = [
        "PairwiseHingeLoss",
        "PairwiseLogisticLoss",
        "AlignmentContrastiveLoss",
        "MutualInformationNeuralEstimationLoss",
    ]
    point_to_evaluate = {
        "data_dir": DATA_DIR,
        "train_loss": "PairwiseHingeLoss",
        "log_num_negatives": 0,
        "sigma": 1.0,
        "margin": 1.0,
        "learning_rate": 0.001,
    }
    config = point_to_evaluate | {
        "train_loss": flaml.tune.choice(train_losses),
        "log_num_negatives": flaml.tune.lograndint(0, 6),
        "sigma": flaml.tune.lograndint(1, 1000),
        "margin": flaml.tune.quniform(-1.0, 1.0, 0.01),
        "learning_rate": flaml.tune.loguniform(0.0001, 0.01),
    }
    low_cost_partial_config = {
        # "train_loss": "PairwiseHingeLoss",
        # "log_num_negatives": 0,
        # "sigma": 1.0,
        # "margin": 1.0,
        # "learning_rate": 0.1,
    }
    return flaml.tune.run(
        evaluation_function,
        metric=METRIC["name"],
        mode=METRIC["mode"],
        config=config,
        low_cost_partial_config=low_cost_partial_config,
        points_to_evaluate=[point_to_evaluate],
        time_budget_s=60 * 60 * 24,
        num_samples=-1,
        resource_attr="limit_train_batches",
        min_resource=1 / 16,
        max_resource=1.0,
        reduction_factor=2,
    )


def main() -> None:
    flaml_tune()


if __name__ == "__main__":
    import rich

    analysis = flaml_tune()
    rich.print(analysis.best_result)
