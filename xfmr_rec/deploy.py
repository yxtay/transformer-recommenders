from typing import Any

import bentoml

"""Helpers for testing and running BentoML services used by the project.

Provides a small convenience to invoke BentoML service endpoints in tests.
"""


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
