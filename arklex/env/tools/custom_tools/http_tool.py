"""
HTTP request tool for external APIs in the Arklex framework.

This module defines a tool for making HTTP requests to external APIs and handling responses. It is designed to be registered and used within the Arklex framework's tool system, providing a flexible interface for API integrations.
"""

import inspect
from typing import Any

import requests

from arklex.env.tools.tools import register_tool
from arklex.orchestrator.entities.msg_state_entities import HTTPParams
from arklex.utils.exceptions import ToolExecutionError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


def clean_json_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Clean JSON data by removing or replacing invalid values that could cause parsing errors.
    
    Args:
        data: Dictionary to clean
        
    Returns:
        Cleaned dictionary with valid JSON values
    """
    if not isinstance(data, dict):
        return data
    
    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned_data[key] = clean_json_data(value)
        elif isinstance(value, list):
            cleaned_data[key] = [clean_json_data(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, str):
            # Remove any remaining placeholders that might cause JSON parsing issues
            if "{{" in value and "}}" in value:
                # Replace with empty string to avoid JSON parsing errors
                cleaned_data[key] = ""
            else:
                cleaned_data[key] = value
        else:
            cleaned_data[key] = value
    
    return cleaned_data


def validate_request_body(body: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Validate and clean the request body to ensure it's valid JSON.
    
    Args:
        body: Request body to validate
        
    Returns:
        Cleaned and validated request body
    """
    if body is None:
        return None
    
    try:
        # First clean the data
        cleaned_body = clean_json_data(body)
        
        # Test JSON serialization to catch any remaining issues
        import json
        json.dumps(cleaned_body)
        
        return cleaned_body
    except (TypeError, ValueError) as e:
        log_context.error(f"Invalid request body after cleaning: {str(e)}")
        # Return a minimal valid body if cleaning fails
        return {"error": "Invalid request body", "details": str(e)}


def replace_placeholders(
    data: dict[str, object] | list[object] | str | object,
    slot_map: dict[str, object],
) -> dict[str, object] | list[object] | str | object:
    """
    Recursively replace {{slot_name}} in all string values in data with slot_map[slot_name].
    If the slot is not found, replace the placeholder with appropriate default values based on type.
    Only supports entire placeholder replacement (no partial placeholder support).
    """
    import re

    def handle_dict(d: dict[str, object]) -> dict[str, object]:
        return {k: replace_placeholders(v, slot_map) for k, v in d.items()}

    def handle_list(lst: list[object]) -> list[object]:
        return [replace_placeholders(item, slot_map) for item in lst]

    def handle_entire_placeholder(s: str) -> object:
        placeholder_pattern = r"^\{\{(\w+)\}\}$"
        match = re.match(placeholder_pattern, s)
        if match:
            slot_name = match.group(1)
            slot_info = slot_map.get(slot_name)
            if slot_info is not None:
                value = slot_info.get("value")
                slot_type = slot_info.get("type")
                if value is not None:
                    return value
                # Use type to determine default
                defaults = {
                    "list": [],
                    "str": "",
                    "string": "",
                    "int": 0,
                    "integer": 0,
                    "float": 0.0,
                    "bool": False,
                    "boolean": False,
                }
                return defaults.get(slot_type)
            else:
                # If slot not found in slot_map, return appropriate default
                return ""
        return None  # Not a full placeholder

    if isinstance(data, dict):
        return handle_dict(data)
    elif isinstance(data, list):
        return handle_list(data)
    elif isinstance(data, str):
        entire_placeholder_result = handle_entire_placeholder(data)
        if entire_placeholder_result is not None:
            return entire_placeholder_result
        else:
            return data
    else:
        return data


@register_tool(
    desc="Make HTTP requests to external APIs and handle responses",
    slots=[],
    outputs=["response"],
    isResponse=False,
)
def http_tool(
    slots: list[dict[str, Any]] | None = None, **kwargs: dict[str, Any]
) -> str:
    """Make an HTTP request and return the response"""
    func_name: str = inspect.currentframe().f_code.co_name
    try:
        params: HTTPParams = HTTPParams(**kwargs)
        log_context.info(
            f"HTTPTool execution called with args: {kwargs}, slots: {slots}"
        )
        if slots:
            # Process slots based on their target
            for slot in slots:
                slot_name = None
                slot_value = None
                slot_target = None

                if hasattr(slot, "name") and hasattr(slot, "value"):
                    slot_name = slot.name
                    slot_value = slot.value
                    slot_target = getattr(slot, "target", None)
                elif isinstance(slot, dict):
                    slot_name = slot.get("name")
                    slot_value = slot.get("value")
                    slot_target = slot.get("target")

                if (
                    slot_name
                    and slot_value is not None
                    and slot_target
                    and slot_target == "params"
                ):
                    # Add to params
                    if not params.params:
                        params.params = {}
                    params.params[slot_name] = slot_value
                    log_context.info(
                        f"Added slot '{slot_name}' with value '{slot_value}' to params"
                    )

            # Build slot_map once after all slots are processed
            slot_map = {}
            for slot in slots:
                if isinstance(slot, dict):
                    slot_map[slot.get("name")] = {
                        "value": slot.get("value"),
                        "type": slot.get("type"),
                        "description": slot.get("description"),
                    }
                else:
                    slot_map[getattr(slot, "name", None)] = {
                        "value": getattr(slot, "value", None),
                        "type": getattr(slot, "type", None),
                        "description": getattr(slot, "description", None),
                    }
            # Recursively replace placeholders in body
            params.body = replace_placeholders(params.body, slot_map)

        # Clean and validate JSON data to prevent parsing errors
        if params.body:
            params.body = validate_request_body(params.body)
        
        # Remove any {{}} placeholders from params and body as these are optional parameters
        def remove_placeholders(data_dict: dict[str, Any] | None) -> None:
            if not data_dict:
                return
            keys_to_remove = []
            for key, value in data_dict.items():
                if (
                    isinstance(value, str)
                    and value.startswith("{{")
                    and value.endswith("}}")
                ):
                    keys_to_remove.append(key)
                    log_context.info(
                        f"Removing placeholder '{key}' with value '{value}'"
                    )
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
        
        # Handle JSON parsing with better error handling
        try:
            response_data: dict[str, Any] | list[Any] = response.json()
        except ValueError as json_error:
            log_context.error(f"Failed to parse JSON response: {str(json_error)}")
            # Return the raw text if JSON parsing fails
            response_data = {"raw_response": response.text, "error": "JSON parsing failed"}
        
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
