from __future__ import annotations

import functools
import tempfile
from typing import TYPE_CHECKING

import bentoml
import pydantic
from jsonargparse import auto_cli

from xfmr_rec.deploy import test_bento
from xfmr_rec.mf import MODEL_NAME
from xfmr_rec.mf.data import MFDataModule
from xfmr_rec.mf.service import Service
from xfmr_rec.mf.trainer import MFRecLightningModule, cli_main
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

    When a checkpoint path is provided, this function loads the saved
    DataModule and LightningModule from the checkpoint and extracts their
    configuration dictionaries suitable for re-creating the data and model
    configuration via JSON-serializable mappings.

    Args:
        ckpt_path: Path to a Lightning checkpoint file. If empty, a minimal
            default configuration is returned (used for quick local runs).

    Returns:
        A dictionary containing `data` and `model` configuration mappings. The
        returned structure is intended to be passed into the CLI helpers that
        accept nested config dictionaries.

    Raises:
        Any exception raised by `load_from_checkpoint` if the checkpoint is
        invalid or missing required attributes.
    """

    if not ckpt_path:
        return {"data": {"config": {"num_workers": 0}}}

    datamodule = MFDataModule.load_from_checkpoint(ckpt_path)
    model = MFRecLightningModule.load_from_checkpoint(ckpt_path)
    return {
        "data": {"config": datamodule.config.model_dump()},
        "model": {"config": model.config.model_dump()},
    }


def prepare_trainer(
    ckpt_path: str = "", stage: str = "validate", fast_dev_run: int = 0
) -> Trainer:
    """Prepare a Lightning Trainer for validation or testing.

    If no checkpoint path is provided, a quick local trainer configured with
    `fast_dev_run` is returned for development. When a checkpoint path is
    given, the function constructs a trainer configured to run on CPU with
    checkpointing and logging disabled and uses the saved configuration from
    the checkpoint to instantiate the model and data module.

    Args:
        ckpt_path: Optional path to a Lightning checkpoint. When provided, the
            model and data configs will be loaded from it.
        stage: The CLI stage to run (e.g., 'validate' or 'test'). This value
            is forwarded to the project's `cli_main` helper.
        fast_dev_run: Controls Lightning's `fast_dev_run` flag when running
            from a checkpoint. Useful for fast smoke tests.

    Returns:
        A configured `Trainer` instance from PyTorch Lightning.

    Raises:
        Any exceptions raised by `cli_main` when constructing the Trainer or
        when loading checkpointed components.
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
    """Save the trained model artifacts into a BentoML model store.

    This function creates a new BentoML model entry named by `MODEL_NAME`
    and asks the Lightning `trainer.model` to save its internal state into
    the created model directory.

    Args:
        trainer: A configured PyTorch Lightning `Trainer` which has an
            attached `model` ready to be saved.

    Raises:
        Any exception raised by BentoML model creation or the Lightning
        module's `save` implementation.
    """
    with bentoml.models.create(MODEL_NAME) as model_ref:
        model: MFRecLightningModule = trainer.model
        model.save(model_ref.path)


def test_queries() -> None:
    """Run a set of example queries against the BentoML service to validate
    basic API behavior.

    The function uses `test_bento` to call several endpoints exposed by the
    BentoML `Service` and validates that returned payloads match the
    project's canonical example objects. This is useful as a quick smoke test
    after packaging a model with BentoML.

    Raises:
        ValueError: If any of the example responses do not match the expected
            example values or if a recommendation list does not contain the
            expected number of items.
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
    """CLI entry point used to prepare, save, and validate a deployed model.

    When invoked, this function prepares a Trainer (optionally loading a
    checkpoint), saves the model into the BentoML model store, and runs a
    set of smoke tests against the bundled Bento service.

    Args:
        ckpt_path: Optional path to a Lightning checkpoint to load.
    """

    trainer = prepare_trainer(ckpt_path=ckpt_path)
    save_model(trainer=trainer)
    test_queries()


if __name__ == "__main__":
    main()
