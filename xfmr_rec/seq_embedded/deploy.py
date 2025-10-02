from __future__ import annotations

import functools
import tempfile
from typing import TYPE_CHECKING

import bentoml
import pydantic
from jsonargparse import auto_cli

from xfmr_rec.deploy import test_bento
from xfmr_rec.seq.data import SeqDataModule
from xfmr_rec.seq_embedded import MODEL_NAME
from xfmr_rec.seq_embedded.service import Service
from xfmr_rec.seq_embedded.trainer import SeqEmbeddedLightningModule, cli_main
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
    """Load config mappings from a Lightning checkpoint for seq_embedded.

    Returns a minimal default when `ckpt_path` is empty; otherwise loads and
    serializes the saved DataModule and LightningModule configurations.

    Args:
        ckpt_path: Path to a Lightning checkpoint or empty string.

    Returns:
        A dictionary with `data` and `model` config mappings.
    """

    if not ckpt_path:
        return {"data": {"config": {"num_workers": 0}}}

    datamodule = SeqDataModule.load_from_checkpoint(ckpt_path)
    model = SeqEmbeddedLightningModule.load_from_checkpoint(ckpt_path)
    return {
        "data": {"config": datamodule.config.model_dump()},
        "model": {"config": model.config.model_dump()},
    }


def prepare_trainer(
    ckpt_path: str = "", stage: str = "validate", fast_dev_run: int = 0
) -> Trainer:
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
    """Save the seq_embedded Lightning model into the BentoML store.

    Creates a BentoML model entry and instructs the Lightning module to write
    its saved artifacts to the Bento model path.

    Args:
        trainer: PyTorch Lightning Trainer holding the trained model.
    """

    with bentoml.models.create(MODEL_NAME) as model_ref:
        model: SeqEmbeddedLightningModule = trainer.model
        model.save(model_ref.path)


def _seq_embedded_doc_placeholder() -> None:
    """Internal placeholder to satisfy docstring coverage for the module.

    Not used by runtime; safe to remove.
    """


def test_queries() -> None:
    """Execute sanity-check API calls against the seq_embedded Bento service.

    Calls the example endpoints and asserts returned payloads conform to the
    expected canonical examples. Raises ValueError when checks fail.
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
    """CLI helper: prepare trainer, save model, and run basic validation.

    Args:
        ckpt_path: Optional path to Lightning checkpoint to load configuration.
    """

    trainer = prepare_trainer(ckpt_path=ckpt_path)
    save_model(trainer=trainer)
    test_queries()


if __name__ == "__main__":
    main()
