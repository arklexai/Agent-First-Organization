"""
HTTP request tool for external APIs in the Arklex framework.

This module defines a tool for making HTTP requests to external APIs and handling responses. It is designed to be registered and used within the Arklex framework's tool system, providing a flexible interface for API integrations.
"""

import inspect
from typing import Any

import requests

from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError
from arklex.utils.graph_state import HTTPParams
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


@register_tool(
    desc="Make HTTP requests to external APIs and handle responses",
    slots=[],
    outputs=["response"],
    isResponse=False,
)
def http_tool(**kwargs: dict[str, Any]) -> str:
    """Make an HTTP request and return the response"""
    func_name: str = inspect.currentframe().f_code.co_name
    try:
        params: HTTPParams = HTTPParams(**kwargs)
        log_context.info(
            f"Making a {params.method} request to {params.endpoint}, with body: {params.body} and params: {params.params}"
        )
        response: requests.Response = requests.request(
            method=params.method,
            url=params.endpoint,
            headers=params.headers,
            json=params.body,
            params=params.params,
        )
        response.raise_for_status()
        response_data: dict[str, Any] | list[Any] = response.json()
        log_context.info(
            f"Response from the {params.endpoint} for body: {params.body} and params: {params.params} is: {response_data}"
        )
        return str(response_data)

    except requests.exceptions.RequestException as e:
        log_context.error(f"Error making HTTP request: {str(e)}")
        raise ToolExecutionError(
            func_name, f"Error making HTTP request: {str(e)}"
        ) from e
    except Exception as e:
        log_context.error(f"Unexpected error in HTTPTool: {str(e)}")
        raise ToolExecutionError(func_name, f"Unexpected error: {str(e)}") from e


http_tool.__name__ = "http_tool"
