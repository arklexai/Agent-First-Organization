"""Entities for slot management.

This module defines the entities used in the slot management system,
including Slot for tracking slot states and relationships.

Key Components:
- Slot: Represents a slot in the conversation system.
- SlotInput: Represents the input format for slot filling operations.
- SlotInputList: Represents a list of slot inputs.
- Verification: Represents the result of verifying a slot value.
- Slot processing utilities: Common functions for slot processing and valueSource logic.
"""

from typing import Any

from pydantic import BaseModel, Field

# Type conversion mapping for slot values
TYPE_CONVERTERS = {
    "int": int,
    "float": float,
    "bool": lambda v: v
    if isinstance(v, bool)
    else (v.lower() == "true" if isinstance(v, str) else bool(v)),
    "str": lambda v: v if isinstance(v, dict | list) else str(v),
}


def convert_value_for_type(
    value: str | int | float | bool | list | dict | None, type_str: str
) -> str | int | float | bool | list | dict | None:
    """Convert value to the specified type.

    Args:
        value: The value to convert
        type_str: The target type string

    Returns:
        The converted value
    """
    type_mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
    }

    internal_type = type_mapping.get(type_str, "str")
    converter = TYPE_CONVERTERS.get(internal_type, lambda x: x)

    try:
        return converter(value)
    except Exception:
        return value


def extract_fields_from_properties(
    properties: dict, fields: dict, path: str = ""
) -> None:
    """Extract fixed/default fields from properties, handling nested structures.

    Args:
        properties: Properties dictionary from schema
        fields: Dictionary to populate with field definitions
        path: Current path for nested fields
    """
    for field_name, field_def in properties.items():
        current_path = f"{path}.{field_name}" if path else field_name
        value_source = field_def.get("valueSource")

        if value_source in ["fixed", "default"] and "value" in field_def:
            fields[current_path] = {
                "value": field_def["value"],
                "type": field_def.get("type", "string"),
                "valueSource": value_source,
                "field_name": field_name,
            }

        # Handle nested objects and arrays
        if field_def.get("type") == "object":
            nested_props = field_def.get("properties", {})
            extract_fields_from_properties(nested_props, fields, current_path)
        elif field_def.get("type") == "array":
            items = field_def.get("items", {})
            if items.get("type") == "object":
                nested_props = items.get("properties", {})
                extract_fields_from_properties(nested_props, fields, current_path)


def extract_nested_fields_from_definition(
    field_def: dict[str, Any], fields: dict[str, dict[str, Any]], path: str = ""
) -> None:
    """Extract field definitions from a field definition, handling nested structures.

    Args:
        field_def: Field definition dictionary
        fields: Dictionary to populate with field definitions
        path: Current path for nested fields
    """
    field_name = field_def.get("name", "")
    current_path = f"{path}.{field_name}" if path else field_name

    # Add current field if it has valueSource
    value_source = field_def.get("valueSource")
    if value_source in ["fixed", "default"] and "value" in field_def:
        fields[current_path] = {
            "name": field_name,
            "type": field_def.get("type", "str"),
            "valueSource": value_source,
            "value": field_def.get("value"),
            "repeatable": field_def.get("repeatable", False),
        }


def extract_fields_from_openai_schema(
    schema: dict, slot_name: str, fields: dict[str, dict[str, Any]], base_path: str = ""
) -> None:
    """Extract field definitions from OpenAI function-style schema.

    Args:
        schema: OpenAI function schema dictionary
        slot_name: Name of the slot
        fields: Dictionary to populate with field definitions
        base_path: Base path for nested fields
    """
    if "function" not in schema:
        return

    function_block = schema.get("function", {})
    parameters = function_block.get("parameters", {})
    properties = parameters.get("properties", {})
    slot_prop = properties.get(slot_name)

    if not slot_prop:
        return

    # Handle array of objects
    if slot_prop.get("type") == "array":
        items = slot_prop.get("items", {})
        if items.get("type") == "object":
            extract_properties_recursively(
                items.get("properties", {}), fields, base_path
            )
    # Handle single object
    elif slot_prop.get("type") == "object":
        extract_properties_recursively(
            slot_prop.get("properties", {}), fields, base_path
        )


def extract_properties_recursively(
    properties: dict, fields: dict[str, dict[str, Any]], path: str = ""
) -> None:
    """Recursively extract field definitions from properties.

    Args:
        properties: Properties dictionary
        fields: Dictionary to populate with field definitions
        path: Current path for nested fields
    """
    for field_name, field_def in properties.items():
        current_path = f"{path}.{field_name}" if path else field_name
        value_source = field_def.get("valueSource")

        if value_source in ["fixed", "default"] and "value" in field_def:
            fields[current_path] = {
                "name": field_name,
                "type": field_def.get("type", "string"),
                "valueSource": value_source,
                "value": field_def.get("value"),
                "repeatable": field_def.get("type") == "array",
            }

        # Handle nested objects
        if field_def.get("type") == "object":
            nested_props = field_def.get("properties", {})
            extract_properties_recursively(nested_props, fields, current_path)
        # Handle arrays of objects
        elif field_def.get("type") == "array":
            items = field_def.get("items", {})
            if items.get("type") == "object":
                nested_props = items.get("properties", {})
                extract_properties_recursively(nested_props, fields, current_path)


def find_fixed_default_fields_recursive(schema: dict, slot_name: str) -> dict:
    """Recursively find all fields with valueSource='fixed' or 'default' at any nesting level.

    Args:
        schema: Slot schema dictionary
        slot_name: Name of the slot

    Returns:
        Dictionary mapping field paths to their values and types
    """
    fields = {}

    if isinstance(schema, dict) and "function" in schema:
        function_block = schema.get("function", {})
        parameters = function_block.get("parameters", {})
        properties = parameters.get("properties", {})
        slot_prop = properties.get(slot_name)

        if slot_prop:
            # Handle array of objects
            if slot_prop.get("type") == "array":
                items = slot_prop.get("items", {})
                if items.get("type") == "object":
                    extract_fields_from_properties(items.get("properties", {}), fields)
            # Handle single object
            elif slot_prop.get("type") == "object":
                extract_fields_from_properties(slot_prop.get("properties", {}), fields)

    return fields


def apply_fields_to_item_recursive(
    item: dict, fields: dict, schema: dict, slot_name: str
) -> None:
    """Apply fixed/default fields to an item, handling nested structures.

    Args:
        item: Dictionary to apply values to
        fields: Dictionary of field definitions with values
        schema: Schema for recursive processing
        slot_name: Name of the slot for context
    """
    for field_path, field_info in fields.items():
        # Split path to handle nested fields
        path_parts = field_path.split(".")
        current_obj = item

        # Navigate to the parent object of the target field
        for part in path_parts[:-1]:
            if part in current_obj:
                current_obj = current_obj[part]
            else:
                # If path doesn't exist, skip this field
                break
        else:
            # We found the parent object, now apply the value
            field_name = path_parts[-1]
            value_source = field_info["valueSource"]

            if value_source == "fixed":
                # Always override with fixed value
                converted_value = convert_value_for_type(
                    field_info["value"], field_info["type"]
                )

                # Handle arrays - apply to each item in the array
                if isinstance(current_obj, list):
                    for array_item in current_obj:
                        if isinstance(array_item, dict) and field_name in array_item:
                            array_item[field_name] = converted_value
                elif isinstance(current_obj, dict):
                    current_obj[field_name] = converted_value
            elif value_source == "default":
                # Apply default only if value is missing/empty/null
                converted_value = convert_value_for_type(
                    field_info["value"], field_info["type"]
                )

                # Handle arrays - apply to each item in the array
                if isinstance(current_obj, list):
                    for array_item in current_obj:
                        if isinstance(array_item, dict) and field_name in array_item:
                            current_value = array_item.get(field_name)
                            # Only apply default if value is truly missing/empty/null, not False
                            if (
                                current_value is None
                                or current_value == ""
                                or current_value == "null"
                            ):
                                array_item[field_name] = converted_value
                elif isinstance(current_obj, dict):
                    current_value = current_obj.get(field_name)
                    # Only apply default if value is truly missing/empty/null, not False
                    if (
                        current_value is None
                        or current_value == ""
                        or current_value == "null"
                    ):
                        current_obj[field_name] = converted_value


def apply_values_recursively(
    value: str | int | float | bool | list | dict | None, schema: dict, slot_name: str
) -> None:
    """Recursively apply fixed/default values to nested structures.

    Args:
        value: The value to process (can be dict, list, or primitive)
        schema: The schema containing field definitions
        slot_name: Name of the current slot for context
    """
    if isinstance(value, list):
        # Handle arrays - apply to each item
        for item in value:
            apply_values_recursively(item, schema, slot_name)
    elif isinstance(value, dict):
        # Handle objects - find and apply fixed/default values
        fixed_default_fields = find_fixed_default_fields_recursive(schema, slot_name)
        apply_fields_to_item_recursive(value, fixed_default_fields, schema, slot_name)


class Slot(BaseModel):
    """Represents a slot in the conversation system.

    A slot is a named container for a value of a specific type, with optional
    validation rules and metadata. Slots are used to capture and validate user
    input during conversations.

    The class provides:
    1. Type-safe value storage
    2. Value validation through enums
    3. Metadata for slot description and prompting
    4. Verification status tracking

    Attributes:
        name (str): The name of the slot.
        type (str): The type of the slot value (default: "str").
        value (Any): The current value of the slot (can be primitive, list, dict, or list of dicts).
        enum (Optional[List[Union[str, int, float, bool, None]]]): List of valid values.
        description (str): Description of the slot's purpose.
        prompt (str): Prompt to use when filling the slot.
        required (bool): Whether the slot must be filled.
        verified (bool): Whether the slot's value has been verified.
    """

    name: str
    type: str = Field(default="str")
    value: Any = Field(default=None)
    enum: list[str | int | float | bool | None] | None = Field(default=[])
    description: str = Field(default="")
    prompt: str = Field(default="")
    required: bool = Field(default=False)
    verified: bool = Field(default=False)
    repeatable: bool = Field(default=False)
    slot_schema: list[dict] | dict | None = None
    items: dict | None = None
    target: str | None = None
    valueSource: str | None = Field(default=None)

    # Allow field name 'schema' even though BaseModel defines method/attr with same name
    model_config = {"protected_namespaces": (), "arbitrary_types_allowed": True}

    def to_openai_schema(self) -> dict | None:
        """Build the OpenAI JSON schema for this slot's parameter definition.

        If an OpenAI-style schema dict is already attached to this slot under
        self.schema (containing a top-level key 'function' and nested
        parameters.properties), we will extract this slot's property from there.
        Otherwise, we will synthesize the property schema from this slot's fields.
        """

        def _get_type_map() -> dict[str, str]:
            """Get the mapping from internal types to OpenAI schema types."""
            return {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
            }

        # If a full OpenAI function schema is provided on this slot, prefer extracting from it
        if isinstance(self.slot_schema, dict) and "function" in self.slot_schema:
            try:
                properties = (
                    self.slot_schema.get("function", {})
                    .get("parameters", {})
                    .get("properties", {})
                )
                if self.name in properties:
                    return properties[self.name]
            except Exception:
                # Fall back to synthesizing schema if extraction fails
                pass
        elif (
            hasattr(self, "schema")
            and isinstance(self.schema, dict)
            and "function" in self.schema
        ):
            try:
                properties = (
                    self.schema.get("function", {})
                    .get("parameters", {})
                    .get("properties", {})
                )
                if self.name in properties:
                    return properties[self.name]
            except Exception:
                # Fall back to synthesizing schema if extraction fails
                pass

        def _build_primitive_schema() -> dict:
            """Build schema for primitive type fields."""
            type_map = _get_type_map()
            schema: dict[str, Any] = {
                "type": type_map.get(self.type, "string"),
                "description": getattr(self, "description", ""),
            }
            # Preserve prompt/valueSource/value for downstream use and validation
            if getattr(self, "prompt", None):
                schema["prompt"] = self.prompt
            if getattr(self, "valueSource", None):
                schema["valueSource"] = self.valueSource
            if getattr(self, "value", None) is not None:
                schema["value"] = self.value
            if getattr(self, "enum", None):
                schema["enum"] = list(self.enum or [])
            return schema

        # Handle repeatable fields
        if getattr(self, "repeatable", False):
            # For repeatable primitive types, define the item type
            return {
                "type": "array",
                "items": _build_primitive_schema(),
                "description": getattr(self, "description", ""),
            }
        else:
            # Primitive type
            return _build_primitive_schema()


class SlotInput(BaseModel):
    """Input structure for slot filling operations.

    This class represents the input format for slot filling operations,
    containing the essential information needed to process a slot.

    The class provides:
    1. Structured input format for slot filling
    2. Type-safe value handling
    3. Support for enum-based validation
    4. Descriptive metadata

    Attributes:
        name (str): The name of the slot.
        value (Union[str, int, float, bool, List[str], None]): The current value.
        enum (Optional[List[Union[str, int, float, bool, None]]]): Valid values.
        description (str): Description of the slot's purpose.
    """

    name: str
    value: str | int | float | bool | list[str] | None
    enum: list[str | int | float | bool | None] | None
    description: str


class SlotInputList(BaseModel):
    """Container for a list of slot inputs.

    This class serves as a container for multiple slot inputs that need to be
    processed together in a slot filling operation.

    The class provides:
    1. Batch processing of multiple slots
    2. Structured input format for slot filling operations
    3. Type-safe handling of multiple slot inputs

    Attributes:
        slot_input_list (List[SlotInput]): List of slot inputs to process.
    """

    slot_input_list: list[SlotInput]


class Verification(BaseModel):
    """Verification result for a slot value.

    This class represents the result of verifying a slot value, including
    the reasoning behind the verification decision and whether additional
    verification is needed.

    The class provides:
    1. Structured representation of verification results
    2. Reasoning for verification decisions
    3. Status tracking for additional verification needs

    Attributes:
        thought (str): Reasoning behind the verification decision.
        verification_needed (bool): Whether additional verification is required.
    """

    thought: str
    verification_needed: bool
