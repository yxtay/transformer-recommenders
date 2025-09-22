from __future__ import annotations

import flaml.tune
import mlflow
import numpy as np

from xfmr_rec.mf import MODEL_NAME
from xfmr_rec.mf.trainer import cli_main
from xfmr_rec.params import METRIC
from xfmr_rec.trainer import time_now_isoformat

ArgsType = dict[str, bool | float | int | str]


def get_lightning_args(
    config: ArgsType,
    data_args: ArgsType | None = None,
    model_args: ArgsType | None = None,
) -> dict[str, dict[str, ArgsType]]:
    hidden_size = 2 ** config["log_hidden_size"]
    num_attention_heads = 2 ** config["log_num_attention_heads"]
    intermediate_size = int(hidden_size * 2 ** config["log_intermediate_size"])
    num_negatives = 2 ** config["log_num_negatives"] - 1

    data_args = (data_args or {}) | {}
    model_args = (model_args or {}) | {
        "hidden_size": hidden_size,
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": num_attention_heads,
        "intermediate_size": intermediate_size,
        "num_negatives": num_negatives,
        "margin": config["margin"],
        "train_loss": config["train_loss"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
    }
    return {"data": {"config": data_args}, "model": {"config": model_args}}


def evaluation_function(config: ArgsType) -> dict[str, float]:
    config = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in config.items()
    }

    trainer_args = {"max_epochs": config["max_epochs"]}
    args = {"trainer": trainer_args, **get_lightning_args(config)}

    with mlflow.start_run(run_name=time_now_isoformat(), nested=True):
        cli = cli_main({"fit": args}, log_model=False)
    # get validation metrics from "best" checkpoint
    metrics = cli.trainer.validate(datamodule=cli.datamodule)
    return metrics[0]


def flaml_tune() -> flaml.tune.tune.ExperimentAnalysis:
    train_losses = ["InfoNCELoss", "NCELoss", "PairwiseLogisticLoss"]

    point_to_evaluate = {
        "log_hidden_size": 5,
        "num_hidden_layers": 1,
        "log_num_attention_heads": 2,
        "log_intermediate_size": 1,
        "log_num_negatives": 0,
        "margin": 0.5,
        "train_loss": "InfoNCELoss",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
    }

    config = point_to_evaluate | {
        "log_hidden_size": flaml.tune.randint(4, 7),
        "num_hidden_layers": flaml.tune.randint(1, 4),
        "log_num_attention_heads": flaml.tune.randint(0, 4),
        "log_intermediate_size": flaml.tune.randint(-1, 3),
        "log_num_negatives": flaml.tune.randint(0, 11),
        "margin": flaml.tune.quniform(0.5, 1.5, 0.1),
        "train_loss": flaml.tune.choice(train_losses),
        # "learning_rate": flaml.tune.qloguniform(0.0001, 0.01, 0.0001),
        # "weight_decay": flaml.tune.qloguniform(0.001, 0.1, 0.001),
    }

    low_cost_partial_config = {
        "log_hidden_size": 3,
        "num_hidden_layers": 1,
        "log_num_attention_heads": 0,
        "log_intermediate_size": -1,
        # "log_num_negatives": 0,
        # "margin": 0.5,
        # "train_loss": "InfoNCELoss",
        # "learning_rate": 0.001,
        # "weight_decay": 0.01,
    }

    mlflow.set_experiment(experiment_name=MODEL_NAME)
    with mlflow.start_run(run_name=f"tune-{time_now_isoformat()}"):
        return flaml.tune.run(
            evaluation_function,
            metric=METRIC["name"],
            mode=METRIC["mode"],
            config=config,
            low_cost_partial_config=low_cost_partial_config,
            points_to_evaluate=[point_to_evaluate],
            time_budget_s=60 * 60 * 1,
            num_samples=-1,
            resource_attr="max_epochs",
            min_resource=1,
            max_resource=16,
            reduction_factor=2,
        )


def main() -> None:
    flaml_tune()


if __name__ == "__main__":
    import rich

    analysis = flaml_tune()
    rich.print(analysis.best_result)
