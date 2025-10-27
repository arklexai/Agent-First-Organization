"""
HTTP request tool for external APIs in the Arklex framework.

This module defines a tool for making HTTP requests to external APIs and handling responses. It is designed to be registered and used within the Arklex framework's tool system, providing a flexible interface for API integrations.
"""

import inspect
from typing import Any

import requests
from pydantic import BaseModel, Field

from arklex.orchestrator.nlu.entities.slot_entities import (
    Slot,
)
from arklex.resources.tools.tools import register_tool
from arklex.utils.logging.exceptions import ToolExecutionError, ValidationError
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class HTTPParams(BaseModel):
    """Parameters for HTTP requests.

    This class defines the parameters needed for making HTTP requests,
    including endpoint, method, headers, and body data.

    The class provides:
    1. Endpoint management
    2. HTTP method handling
    3. Header management
    4. Body data handling
    5. URL parameter support
    6. Type-safe HTTP parameter management

    Attributes:
        endpoint (str): The API endpoint URL.
        method (str): HTTP method to use.
        headers (Dict[str, str]): HTTP headers.
        body (Optional[Any]): Request body.
        params (Optional[Dict[str, Any]]): URL parameters.
    """

    endpoint: str
    method: str = Field(default="GET")
    headers: dict[str, str] = Field(
        default_factory=lambda: {"Content-Type": "application/json"}
    )
    body: Any | None = Field(default=None)
    params: dict[str, Any] | None = Field(default=None)


def validate_required_slots(slots: list[Slot]) -> None:
    """
    Validate if all required slots are filled, including nested required fields.
    """
    try:
        for slot in slots:
            slot_name = slot.name
            slot_value = slot.value
            slot_required = slot.required
            slot_repeatable = slot.repeatable
            slot_schema = slot.slot_schema

            log_context.info(
                f"slot_name: {slot_name}, slot_value: {slot_value}, slot_value_type: {type(slot_value)}, slot_repeatable: {slot_repeatable}, slot_required: {slot_required}"
            )

            # Check if the slot is required and missing (can be an array, object, string, etc.)
            # Last check if for the case where the slot is boolean and is False
            if slot_required and not slot_value and slot_value is not False:
                log_context.warning(
                    f"Validation error: Required slot '{slot_name}' is missing"
                )
                raise ValidationError(f"Required slot '{slot_name}' is missing")

            if not slot_schema:
                continue

            # Check for nested fields, navigate to the actual field schema
            function_schema = slot_schema.get("function", {})
            parameters = function_schema.get("parameters", {})
            properties = parameters.get("properties", {})

            # Get the properties for this specific slot
            slot_properties = properties.get(slot_name, {})
            if slot_value is not None and slot_repeatable:
                if not isinstance(slot_value, list):
                    log_context.warning(
                        f"Validation error: Required array slot '{slot_name}' is not a list"
                    )
                    raise ValidationError(
                        f"Required array slot '{slot_name}' is not a list"
                    )
                # validate array items if slot has value and is a repeatable field
                _validate_array_items(slot_name, slot_value, slot_properties)
            elif slot_value is not None and isinstance(slot_value, dict):
                # validate object field if slot has value and is a dict
                required_fields = slot_properties.get("required", [])
                _validate_object_fields(slot_name, slot_value, required_fields)

    except ValidationError as e:
        log_context.warning(f"Validation error: {str(e)}")
        raise e
    except Exception as e:
        log_context.warning(f"Error validating required slots: {str(e)}")
        raise e


def _validate_array_items(
    slot_name: str, slot_value: list, field_schema: dict[str, Any]
) -> None:
    """
    Validate required fields in array items.
    """

    items_schema = field_schema.get("items", {})
    if items_schema.get("type") == "object":
        required_fields = items_schema.get("required", [])
        for item in slot_value:
            _validate_object_fields(slot_name, item, required_fields)
    else:
        for item in slot_value:
            if item is None or item == "":
                raise ValidationError(
                    f"Required array slot '{slot_name}' has empty item"
                )


def _validate_object_fields(
    slot_name: str, slot_value: dict, required_fields: list[str]
) -> None:
    """
    Validate required fields in object.
    """

    for required_field in required_fields:
        field_value = slot_value.get(required_field)
        if field_value is None or field_value == "":
            raise ValidationError(
                f"Required slot field '{slot_name}.{required_field}' is missing"
            )


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
            cleaned_data[key] = [
                clean_json_data(item) if isinstance(item, dict) else item
                for item in value
            ]
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
    description="Make HTTP requests to external APIs and handle responses",
    slots=[],
)
def http_tool(
    slots: list[Slot] | None = None,
    node_specific_data: dict[str, Any] | None = None,
    **kwargs: dict[str, Any],
) -> str:
    """Make an HTTP request and return the response"""
    func_name: str = inspect.currentframe().f_code.co_name
    try:
        params: HTTPParams = HTTPParams(**node_specific_data)
        log_context.info(
            f"HTTPTool execution called with params: {params}, slots: {slots}"
        )
        if slots:
            # check for required fields in slots before processing
            validate_required_slots(slots)
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
            # Recursively replace placeholders in params
            params.params = replace_placeholders(params.params, slot_map)

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
            response_data = {
                "raw_response": response.text,
                "error": "JSON parsing failed",
            }

        log_context.info(f"Response from http tool: {response_data}")
        return str(response_data)

    except ValidationError as e:
        log_context.warning(f"Validation error: {str(e)}")
        raise ToolExecutionError(
            func_name, f"Validation error in HTTPTool: {str(e)}"
        ) from e
    except requests.exceptions.RequestException as e:
        log_context.error(f"Error making HTTP request: {str(e)}")
        raise ToolExecutionError(
            func_name, f"Error making HTTP request: {str(e)}"
        ) from e
    except Exception as e:
        log_context.error(f"Unexpected error in HTTPTool: {str(e)}")
        raise ToolExecutionError(func_name, f"Unexpected error: {str(e)}") from e


http_tool.__name__ = "http_tool"
