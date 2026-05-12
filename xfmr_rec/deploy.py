from __future__ import annotations

from typing import Any

import bentoml
import pydantic
from jsonargparse import auto_cli

from xfmr_rec.data import SeqDataModule
from xfmr_rec.service import (
    EXAMPLE_ITEM,
    EXAMPLE_USER,
    ItemCandidate,
    ItemQuery,
    Service,
    UserQuery,
)
from xfmr_rec.trainer import LightningCLI, RecommenderLightningModule


def test_bento(
    service: type[bentoml.Service[Any]], api_name: str, api_input: dict[str, Any]
) -> dict[str, Any]:
    from starlette.testclient import TestClient

    """Invoke a BentoML service endpoint using a Starlette test client."""
    # disable prometheus, which can cause duplicated metrics error with repeated runs
    service.config["metrics"] = {"enabled": False}

    asgi_app = service.to_asgi()
    with TestClient(asgi_app) as client:
        response = client.post(f"/{api_name}", json=api_input)
        response.raise_for_status()
        return response.json()


def test_queries(service: type[bentoml.Service[Any]]) -> None:
    """Execute a set of sanity-check API calls against a BentoML service."""
    import rich

    example_item_data = test_bento(service, "item_id", {"item_id": "1"})
    example_item = ItemQuery.model_validate(example_item_data)
    rich.print(example_item)
    exclude_fields = {"embedding"}
    assert example_item.model_dump(exclude=exclude_fields) == EXAMPLE_ITEM.model_dump(
        exclude=exclude_fields
    ), f"{example_item = } != {EXAMPLE_ITEM = }"

    example_user_data = test_bento(service, "user_id", {"user_id": "1"})
    example_user = UserQuery.model_validate(example_user_data)
    rich.print(example_user)
    exclude_fields = {"history", "target"}
    assert example_user.model_dump(exclude=exclude_fields) == EXAMPLE_USER.model_dump(
        exclude=exclude_fields
    ), f"{example_user = } != {EXAMPLE_USER = }"

    top_k = 5
    item_recs = test_bento(
        service, "recommend_with_item_id", {"item_id": "1", "top_k": top_k}
    )
    item_recs = pydantic.TypeAdapter(list[ItemCandidate]).validate_python(item_recs)
    rich.print(item_recs)
    assert len(item_recs) == top_k, f"{len(item_recs) = } != {top_k}"

    user_recs = test_bento(
        service, "recommend_with_user_id", {"user_id": "1", "top_k": top_k}
    )
    user_recs = pydantic.TypeAdapter(list[ItemCandidate]).validate_python(user_recs)
    rich.print(user_recs)
    assert len(user_recs) == top_k, f"{len(user_recs) = } != {top_k}"


def main(ckpt_path: str = "") -> None:
    """CLI helper: prepare trainer, save model, and run basic validation."""
    cli = LightningCLI(RecommenderLightningModule, SeqDataModule, model_name="xfmr_rec")
    trainer = cli.prepare_trainer(ckpt_path=ckpt_path)
    cli.save_model(trainer=trainer)
    test_queries(Service)


def cli_main() -> None:
    auto_cli(main, as_positional=False)


if __name__ == "__main__":
    cli_main()
