"""
HTTP request tool for external APIs in the Arklex framework.

This module defines a tool for making HTTP requests to external APIs and handling responses. It is designed to be registered and used within the Arklex framework's tool system, providing a flexible interface for API integrations.
"""

import requests
import inspect
from typing import Dict, Any, Union, List

from arklex.env.tools.tools import register_tool
from arklex.utils.graph_state import HTTPParams
from arklex.utils.exceptions import ToolExecutionError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)
slots = [
    {
        "name": "cat breed",
        "type": "str",
        "description": "Type of cat breed",
        "prompt": "hello, how are you?",
        "required": True,
    },
    {
        "name": "cat age",
        "type": "int",
        "description": "Age of the cat",
        "prompt": "what is the age of the cat?",
        "required": True,
    }
]

@register_tool(
    desc="Make HTTP requests to external APIs and handle responses",
    slots=slots,
    outputs=["response"],
    isResponse=False,
)
def http_tool(**kwargs: Dict[str, Any]) -> str:
    """Make an HTTP request and return the response"""
    func_name: str = inspect.currentframe().f_code.co_name
    try:
        params: HTTPParams = HTTPParams(**kwargs)
        slots = kwargs.get("slots")  # This should be a list of Slot objects or dicts
        log_context.info(f"Slots: {slots}")
        if slots:
            # Process slots based on their target
            for slot in slots:
                # Handle both Slot objects and dictionaries
                slot_name = None
                slot_value = None
                slot_target = None
                
                if hasattr(slot, 'name') and hasattr(slot, 'value'):
                    # Slot object (Pydantic model)
                    slot_name = slot.name
                    slot_value = slot.value
                    slot_target = getattr(slot, 'target', None)
                elif isinstance(slot, dict):
                    # Dictionary format
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
                    # If target is None, ignore the slot

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
        response_data: Union[Dict[str, Any], List[Any]] = response.json()
        log_context.info(
            f"Response from the {params.endpoint} for body: {params.body} and params: {params.params} is: {response_data}"
        )
        return str(response_data)

    except requests.exceptions.RequestException as e:
        log_context.error(f"Error making HTTP request: {str(e)}")
        raise ToolExecutionError(func_name, f"Error making HTTP request: {str(e)}")
    except Exception as e:
        log_context.error(f"Unexpected error in HTTPTool: {str(e)}")
        raise ToolExecutionError(func_name, f"Unexpected error: {str(e)}")


http_tool.__name__ = "http_tool"
