from __future__ import annotations

from typing import TYPE_CHECKING

from xfmr_rec.seq_embedded.trainer import SeqEmbeddedRecLightningModule

if TYPE_CHECKING:
    from typing import Any

    import bentoml
    from lightning import Trainer


def load_args(ckpt_path: str) -> dict[str, Any]:
    from xfmr_rec.seq_embedded.data import SeqEmbeddedDataModule

    if not ckpt_path:
        return {"data": {"config": {"num_workers": 0}}}

    datamodule = SeqEmbeddedDataModule.load_from_checkpoint(ckpt_path)
    model = SeqEmbeddedRecLightningModule.load_from_checkpoint(ckpt_path)
    return {
        "data": {"config": datamodule.config.model_dump()},
        "model": {"config": model.config.model_dump()},
    }


def prepare_trainer(
    ckpt_path: str = "", stage: str = "validate", fast_dev_run: int = 0
) -> Trainer:
    import tempfile

    from xfmr_rec.seq_embedded.trainer import cli_main

    if not ckpt_path:
        args = {"trainer": {"fast_dev_run": True}}
        return cli_main({"fit": args}).trainer

    with tempfile.TemporaryDirectory() as tmp:
        trainer_args = {
            "logger": False,
            "fast_dev_run": fast_dev_run,
            "enable_checkpointing": False,
            "default_root_dir": tmp,
        }
        args = {"trainer": trainer_args, "ckpt_path": ckpt_path, **load_args(ckpt_path)}
        return cli_main({stage: args}).trainer


def save_model(trainer: Trainer) -> None:
    import bentoml

    from xfmr_rec.params import SEQ_EMBEDDED_MODEL_NAME

    with bentoml.models.create(SEQ_EMBEDDED_MODEL_NAME) as model_ref:
        model: SeqEmbeddedRecLightningModule = trainer.model
        model.save(model_ref.path)


def test_bento(
    service: type[bentoml.Service], api_name: str, api_input: dict[str, Any]
) -> dict[str, Any]:
    from starlette.testclient import TestClient

    # disable prometheus, which can cause duplicated metrics error with repeated runs
    service.config["metrics"] = {"enabled": False}

    asgi_app = service.to_asgi()
    with TestClient(asgi_app) as client:
        response = client.post(f"/{api_name}", json=api_input)
        response.raise_for_status()
        return response.json()


def test_queries() -> None:
    import pydantic
    import rich

    from xfmr_rec.seq_embedded.service import (
        EXAMPLE_ITEM,
        EXAMPLE_USER,
        ItemCandidate,
        ItemQuery,
        Service,
        UserQuery,
    )

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


def main(ckpt_path: str = "") -> None:
    trainer = prepare_trainer(ckpt_path=ckpt_path)
    save_model(trainer=trainer)
    test_queries()


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, as_positional=False)
