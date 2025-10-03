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
    service: type[bentoml.Service[Any]], api_name: str, api_input: dict[str, Any]
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


def test_queries(service: type[bentoml.Service[Any]]) -> None:
    """Execute a set of sanity-check API calls against a BentoML service.

    This function issues several POST requests against a BentoML Service's
    example endpoints using an in-process Starlette test client. Responses
    are validated against the project's pydantic data models. The checks
    exercise both single-object endpoints (item and user lookups) and the
    top-k recommendation endpoints for users and items.

    Args:
        service: A BentoML Service class (not an instance). The service's
            configuration will be modified temporarily to disable metrics to
            avoid duplicated Prometheus metrics during repeated test runs.

    Returns:
        None. The function raises on unexpected results.

    Raises:
        AssertionError: If any response does not match the expected example
            schema or if recommendation endpoints do not return the expected
            number of items. The function performs these checks using
            assertion statements.

    Notes:
        - Uses `test_bento` to construct an ASGI app and make requests with
            `starlette.testclient.TestClient` (runs the app in-process for tests).
        - Comparison of example objects excludes large or non-deterministic
            fields (for example, embeddings and user history) to focus the
            checks on canonical structural equality.
    """
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
