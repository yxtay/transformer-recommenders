from __future__ import annotations

import functools
import tempfile
from typing import TYPE_CHECKING

import bentoml
import pydantic
from jsonargparse import auto_cli

from xfmr_rec.deploy import test_bento
from xfmr_rec.seq import MODEL_NAME
from xfmr_rec.seq.data import SeqDataModule
from xfmr_rec.seq.service import Service
from xfmr_rec.seq.trainer import SeqRecLightningModule, cli_main
from xfmr_rec.service import (
    EXAMPLE_ITEM,
    EXAMPLE_USER,
    ItemCandidate,
    ItemQuery,
    UserQuery,
)

if TYPE_CHECKING:
    from typing import Any

    from lightning import Trainer


def load_args(ckpt_path: str) -> dict[str, Any]:
    """Load configuration arguments from a Lightning checkpoint.

    When no checkpoint is supplied, return a minimal default data config
    suitable for fast local runs. If a checkpoint path is provided, load the
    saved DataModule and LightningModule and return their serialized configs.

    Args:
        ckpt_path: Path to a Lightning checkpoint or empty string for defaults.

    Returns:
        A mapping containing `data` and `model` configuration dictionaries.
    """

    if not ckpt_path:
        return {"data": {"config": {"num_workers": 0}}}

    datamodule = SeqDataModule.load_from_checkpoint(ckpt_path)
    model = SeqRecLightningModule.load_from_checkpoint(ckpt_path)
    return {
        "data": {"config": datamodule.config.model_dump()},
        "model": {"config": model.config.model_dump()},
    }


def prepare_trainer(
    ckpt_path: str = "", stage: str = "validate", fast_dev_run: int = 0
) -> Trainer:
    """Prepare a Lightning Trainer for running validation or tests.

    If `ckpt_path` is empty, a minimal fast-dev-run trainer is returned for
    quick iterations. If a checkpoint is provided, the trainer and model/data
    configs are reconstructed from the checkpoint and a CPU-only trainer is
    created for deterministic validation.

    Args:
        ckpt_path: Optional path to a Lightning checkpoint.
        stage: CLI stage to execute (commonly 'validate').
        fast_dev_run: Passed into the Trainer to control quick runs.

    Returns:
        A PyTorch Lightning `Trainer` instance.
    """

    if not ckpt_path:
        args = {"trainer": {"fast_dev_run": True}}
        return cli_main({"fit": args}).trainer

    with tempfile.TemporaryDirectory() as tmp:
        trainer_args = {
            "accelerator": "cpu",
            "logger": False,
            "fast_dev_run": fast_dev_run,
            "enable_checkpointing": False,
            "default_root_dir": tmp,
        }
        args = {"trainer": trainer_args, "ckpt_path": ckpt_path, **load_args(ckpt_path)}
        return cli_main({stage: args}).trainer


def save_model(trainer: Trainer) -> None:
    """Persist the Seq model into the BentoML model store.

    Creates a BentoML model entry named by `MODEL_NAME` and delegates to the
    Lightning module's `save` method to write model artifacts into the
    created path.

    Args:
        trainer: A PyTorch Lightning `Trainer` containing the trained model.

    Raises:
        Any exception raised by BentoML or the Lightning module's save call.
    """

    with bentoml.models.create(MODEL_NAME) as model_ref:
        model: SeqRecLightningModule = trainer.model
        model.save(model_ref.path)


def test_queries() -> None:
    """Run example API queries on the Seq Bento service and validate outputs.

    Uses `test_bento` to call several endpoints and validates returned
    payloads against the project's canonical example objects. Raises
    ValueError when any check fails.
    """

    import rich

    example_item_data = test_bento(Service, "item_id", {"item_id": "1"})
    example_item = ItemQuery.model_validate(example_item_data)
    rich.print(example_item)
    exclude_fields = {"embedding"}
    if example_item.model_dump(exclude=exclude_fields) != EXAMPLE_ITEM.model_dump(
        exclude=exclude_fields
    ):
        msg = f"{example_item = } != {EXAMPLE_ITEM = }"
        raise ValueError(msg)

    example_user_data = test_bento(Service, "user_id", {"user_id": "1"})
    example_user = UserQuery.model_validate(example_user_data)
    rich.print(example_user)
    exclude_fields = {"history", "target"}
    if example_user.model_dump(exclude=exclude_fields) != EXAMPLE_USER.model_dump(
        exclude=exclude_fields
    ):
        msg = f"{example_user = } != {EXAMPLE_USER = }"
        raise ValueError(msg)

    top_k = 5
    item_recs = test_bento(
        Service, "recommend_with_item_id", {"item_id": "1", "top_k": top_k}
    )
    item_recs = pydantic.TypeAdapter(list[ItemCandidate]).validate_python(item_recs)
    rich.print(item_recs)
    if len(item_recs) != top_k:
        msg = f"{len(item_recs) = } != {top_k}"
        raise ValueError(msg)

    user_recs = test_bento(
        Service, "recommend_with_user_id", {"user_id": "1", "top_k": top_k}
    )
    user_recs = pydantic.TypeAdapter(list[ItemCandidate]).validate_python(user_recs)
    rich.print(user_recs)
    if len(user_recs) != top_k:
        msg = f"{len(user_recs) = } != {top_k}"
        raise ValueError(msg)


@functools.partial(auto_cli, as_positional=False)
def main(ckpt_path: str = "") -> None:
    """CLI entrypoint: prepare a trainer, save the model, and run smoke tests.

    Args:
        ckpt_path: Optional checkpoint path to load model/data configuration.
    """

    trainer = prepare_trainer(ckpt_path=ckpt_path)
    save_model(trainer=trainer)
    test_queries()


if __name__ == "__main__":
    main()
