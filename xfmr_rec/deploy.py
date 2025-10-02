from typing import Any

import bentoml
import pydantic

from xfmr_rec.service import (
    EXAMPLE_ITEM,
    EXAMPLE_USER,
    ItemCandidate,
    ItemQuery,
    UserQuery,
)


def test_bento(
    service: type[bentoml.Service], api_name: str, api_input: dict[str, Any]
) -> dict[str, Any]:
    from starlette.testclient import TestClient

    """Invoke a BentoML service endpoint using a Starlette test client.

    This helper constructs an ASGI app from the provided BentoML service and
    issues a POST request to the given API name with the provided JSON payload.

    Args:
        service: A BentoML Service class (not an instance). The service's
            configuration will be modified to disable metrics to avoid
            duplicated Prometheus metrics during repeated test runs.
        api_name: The name of the BentoML service API (route) to call.
        api_input: A JSON-serializable dictionary to send as the request body.

    Returns:
        The parsed JSON response from the service as a Python dictionary.

    Raises:
        HTTPError: If the response status is not successful (raised by
            `response.raise_for_status()`).

    Notes:
        The function uses `starlette.testclient.TestClient` which runs the ASGI
        app in the same process and is intended for testing only.
    """

    # disable prometheus, which can cause duplicated metrics error with repeated runs
    service.config["metrics"] = {"enabled": False}

    asgi_app = service.to_asgi()
    with TestClient(asgi_app) as client:
        response = client.post(f"/{api_name}", json=api_input)
        response.raise_for_status()
        return response.json()


class TestService:
    def __init__(self, service_cls: type[bentoml.Service]) -> None:
        self.service_cls = service_cls

    def test_queries(self) -> None:
        """Execute sanity-check API calls against a BentoML service.

        Calls the example endpoints and asserts returned payloads conform to
        the canonical example schemas defined in the project's service
        datamodels. Raises ValueError when checks fail.
        """

        import rich

        example_item_data = test_bento(self.service_cls, "item_id", {"item_id": "1"})
        example_item = ItemQuery.model_validate(example_item_data)
        rich.print(example_item)
        exclude_fields = {"embedding"}
        if example_item.model_dump(exclude=exclude_fields) != EXAMPLE_ITEM.model_dump(
            exclude=exclude_fields
        ):
            msg = f"{example_item = } != {EXAMPLE_ITEM = }"
            raise ValueError(msg)

        example_user_data = test_bento(self.service_cls, "user_id", {"user_id": "1"})
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
            self.service_cls, "recommend_with_item_id", {"item_id": "1", "top_k": top_k}
        )
        item_recs = pydantic.TypeAdapter(list[ItemCandidate]).validate_python(item_recs)
        rich.print(item_recs)
        if len(item_recs) != top_k:
            msg = f"{len(item_recs) = } != {top_k}"
            raise ValueError(msg)

        user_recs = test_bento(
            self.service_cls, "recommend_with_user_id", {"user_id": "1", "top_k": top_k}
        )
        user_recs = pydantic.TypeAdapter(list[ItemCandidate]).validate_python(user_recs)
        rich.print(user_recs)
        if len(user_recs) != top_k:
            msg = f"{len(user_recs) = } != {top_k}"
            raise ValueError(msg)
