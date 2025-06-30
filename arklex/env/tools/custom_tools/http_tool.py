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
        slots = kwargs.get("slots")
        log_context.info(f"HTTPTool execution called with args: {kwargs}")
        if slots:
            # Process slots based on their target
            for slot in slots:
                slot_name = None
                slot_value = None
                slot_target = None
                
                if hasattr(slot, 'name') and hasattr(slot, 'value'):
                    slot_name = slot.name
                    slot_value = slot.value
                    slot_target = getattr(slot, 'target', None)
                elif isinstance(slot, dict):
                    slot_name = slot.get("name")
                    slot_value = slot.get("value")
                    slot_target = slot.get("target")
                
                if slot_name and slot_value is not None and slot_target:
                    if slot_target == "params":
                        # Add to params
                        if not params.params:
                            params.params = {}
                        params.params[slot_name] = slot_value
                        log_context.info(f"Added slot '{slot_name}' with value '{slot_value}' to params")
                    elif slot_target == "body":
                        # Add to body
                        if not params.body:
                            params.body = {}
                        params.body[slot_name] = slot_value
                        log_context.info(f"Added slot '{slot_name}' with value '{slot_value}' to body")
        
        # Remove any {{}} placeholders from params and body as these are optional parameters
        def remove_placeholders(data_dict: dict[str, Any] | None) -> None:
            if not data_dict:
                return
            keys_to_remove = []
            for key, value in data_dict.items():
                if isinstance(value, str) and value.startswith('{{') and value.endswith('}}'):
                    keys_to_remove.append(key)
                    log_context.info(f"Removing placeholder '{key}' with value '{value}'")
            for key in keys_to_remove:
                del data_dict[key]
        
        remove_placeholders(params.params)
        remove_placeholders(params.body)

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
