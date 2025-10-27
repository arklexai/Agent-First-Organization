"""Slot filling implementation for NLU.

This module provides the core implementation for slot filling functionality,
supporting both local model-based and remote API-based approaches. It implements
the BaseSlotFilling interface to provide a unified way of extracting and
verifying slot values from input text.

The module includes:
- SlotFiller: Main class for slot filling
- Support for both local and remote slot filling
- Integration with language models and APIs
"""

import json
from typing import Any

from arklex.models.model_service import ModelService
from arklex.orchestrator.nlu.entities.slot_entities import Slot
from arklex.orchestrator.nlu.utils.formatters import (
    format_verification_input as format_verification_input_formatter,
)
from arklex.utils.logging.exceptions import ModelError
from arklex.utils.logging.logging_utils import LogContext, handle_exceptions

log_context = LogContext(__name__)


class SlotFiller:
    """Slot filling implementation.

    This class provides functionality for extracting and verifying slot values
    from user input, supporting both local model-based and remote API-based
    approaches.
    """

    def __init__(
        self,
        model_service: ModelService,
    ) -> None:
        """Initialize the slot filler.

        Args:
            model_service: Service for local model-based slot filling

        Raises:
            ValidationError: If model_service is not provided
        """
        self.model_service = model_service
        log_context.info(
            "SlotFiller initialized successfully",
            extra={
                "mode": "local",
                "operation": "initialization",
            },
        )

    def format_slot_input(
        self, slots: list[dict[str, Any]], context: str, type: str = "chat"
    ) -> tuple[str, str]:
        """Format input for slot filling.

        Creates a prompt for the model to extract slot values from the given context.
        The prompt includes slot definitions and the context to analyze.

        Args:
            slots: List of slot definitions to fill (can be dict or Pydantic model)
            context: Input context to extract values from
            type: Type of slot filling operation (default: "chat")

        Returns:
            Tuple of (user_prompt, system_prompt)
        """
        # Format slot definitions
        slot_definitions = []
        for slot in slots:
            # Handle both dict and Pydantic model inputs
            if isinstance(slot, dict):
                slot_name = slot.get("name", "")
                slot_type = slot.get("type", "string")
                description = slot.get("description", "")
                required = "required" if slot.get("required", False) else "optional"
                items = slot.get("items", {})
            else:
                slot_name = getattr(slot, "name", "")
                slot_type = getattr(slot, "type", "string")
                description = getattr(slot, "description", "")
                required = (
                    "required" if getattr(slot, "required", False) else "optional"
                )
                items = getattr(slot, "items", {})

            slot_def = f"- {slot_name} ({slot_type}, {required}): {description}"
            if items:
                enum_values = (
                    items.get("enum", [])
                    if isinstance(items, dict)
                    else getattr(items, "enum", [])
                )
                if enum_values:
                    slot_def += f"\n  Possible values: {', '.join(enum_values)}"
            slot_definitions.append(slot_def)

        # Create the prompts
        system_prompt = (
            "You are a slot filling assistant. Your task is to extract specific "
            "information from the given context based on the slot definitions. "
            "Extract values for all slots when the information is present in the context, "
            "regardless of whether they are required or optional. "
            "Only set a slot to null if the information is truly not mentioned. "
            "Return the extracted values in JSON format only without any markdown formatting or code blocks."
        )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Slot definitions:\n" + "\n".join(slot_definitions) + "\n\n"
            "Please extract the values for the defined slots from the context. "
            "Extract values whenever the information is mentioned, whether the slot is required or optional. "
            "Set to null only if the information is not present in the context. "
            "Return the results in JSON format with slot names as keys and "
            "extracted values as values."
        )

        return user_prompt, system_prompt

    def process_slot_response(
        self, response: str | dict[str, Any], slots: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process the model's response for slot filling.

        Parses the model's response and updates the slot values accordingly.
        Handles both traditional slot structures and new slot_schema structures.

        Args:
            response: Model's response containing extracted slot values (can be string or dict)
            slots: Original slot definitions (can be dict or Pydantic model)

        Returns:
            Updated list of slots with extracted values

        Raises:
            ValueError: If response parsing fails
        """
        try:
            # If there are no slots to process, return empty list early
            if not slots:
                return []
            # Handle both string and dict responses
            if isinstance(response, str):
                # Parse the JSON response if it's a string
                extracted_values = json.loads(response)
            elif isinstance(response, dict):
                # Use the dict directly if it's already a dict
                extracted_values = response
            else:
                raise ValueError(f"Unsupported response type: {type(response)}")

            # Route to slot_schema handler if ANY slot provides a slot_schema
            has_any_slot_schema = any(getattr(s, "slot_schema", None) for s in slots)
            if has_any_slot_schema:
                # Handle new slot_schema structure
                return self._process_slot_schema_response(extracted_values, slots)
            else:
                # Handle traditional slot structure
                return self._process_traditional_slot_response(extracted_values, slots)

        except json.JSONDecodeError as e:
            log_context.error(f"Error parsing slot filling response: {str(e)}")
            raise ValueError(f"Failed to parse slot filling response: {str(e)}") from e
        except Exception as e:
            log_context.error(f"Error processing slot filling response: {str(e)}")
            raise ValueError(
                f"Failed to process slot filling response: {str(e)}"
            ) from e

    def _process_slot_schema_response(
        self, extracted_values: dict, slots: list
    ) -> list:
        """Process response for slot_schema structure.

        Args:
            extracted_values: Extracted values from model response
            slots: Original slot definitions

        Returns:
            Updated list of slots with extracted values
        """

        if isinstance(extracted_values, dict) and len(slots) >= 1:
            for s in slots:
                slot_name = (
                    s.get("name") if isinstance(s, dict) else getattr(s, "name", None)
                )
                if not slot_name:
                    continue
                value = extracted_values.get(slot_name)
                if value is not None:
                    if isinstance(s, dict):
                        s["value"] = value
                    else:
                        s.value = value
            return slots

        return slots

    def _process_traditional_slot_response(
        self, extracted_values: dict, slots: list
    ) -> list:
        """Process response for traditional slot structure.

        Args:
            extracted_values: Extracted values from model response
            slots: Original slot definitions

        Returns:
            Updated list of slots with extracted values
        """
        # Update slot values
        for slot in slots:
            # Handle both dict and Pydantic model inputs
            if isinstance(slot, dict):
                slot_name = slot.get("name", "")
                slot["value"] = extracted_values.get(slot_name)
            else:
                slot_name = getattr(slot, "name", "")
                slot.value = extracted_values.get(slot_name)

        return slots

    def format_verification_input(
        self, slot: dict[str, Any], chat_history_str: str
    ) -> str:
        """Format input for slot verification.

        Creates a prompt for the model to verify if a slot value is correct and valid.

        Args:
            slot: Slot definition with value to verify
            chat_history_str: Chat history context

        Returns:
            str: Formatted verification prompt
        """
        return format_verification_input_formatter(slot, chat_history_str)

    def process_verification_response(self, response: str) -> tuple[bool, str]:
        """Process the model's response for slot verification.

        Parses the model's response to determine if verification is needed.

        Args:
            response: Model's response for verification

        Returns:
            Tuple[bool, str]: (verification_needed, reason)
        """
        try:
            # Parse JSON response from formatters
            log_context.info(f"Verification response: {response}")
            response_data = json.loads(response)
            verification_needed = response_data.get("verification_needed", True)
            thought = response_data.get("thought", "No reasoning progivided")
            return verification_needed, thought
        except json.JSONDecodeError as e:
            log_context.error(f"Error parsing verification response: {str(e)}")
            # Default to needing verification if JSON parsing fails
            return True, f"Failed to parse verification response: {str(e)}"

    def format_slot_schema_input(
        self, function_def: dict[str, Any], context: str
    ) -> tuple[str, str]:
        """Format input for slot extraction using an OpenAI-style function schema.

        Args:
            function_def: Dict with keys 'name','description','parameters' matching OpenAI function schema
            context: Input context to extract values from

        Returns:
            Tuple of (user_prompt, system_prompt)
        """
        name = function_def.get("name", "tool")
        description = function_def.get("description", "")
        parameters = function_def.get("parameters", {})
        # Provide the parameters JSON exactly to the model
        schema_json = json.dumps(parameters, ensure_ascii=False)

        system_prompt = (
            "You are a precise information extraction assistant. "
            "Given a JSON Schema for function parameters and a conversation context, "
            "extract values strictly matching the schema. Return ONLY a JSON object that conforms to the 'parameters' schema. "
            "Do not include Markdown or explanations."
        )
        user_prompt = (
            f"Function: {name}\n"
            f"Description: {description}\n"
            f"Parameters JSON Schema:\n{schema_json}\n\n"
            f"Context:\n{context}\n\n"
            "Return a JSON object whose keys and value types exactly match the 'properties' under the parameters schema."
        )
        return user_prompt, system_prompt

    def _slots_to_openai_schema(self, slots: list[Slot]) -> dict[str, Any]:
        """Convert a list of Slot objects into a single OpenAI JSON schema.

        The returned schema is always a top-level object where each slot is a
        property whose value is that slot's own schema definition.

        Preference order per slot:
        1) slot.slot_schema (deep-copied and sanitized), else
        2) slot.to_openai_schema().
        """
        import copy

        properties: dict[str, Any] = {}
        required: list[str] = []

        for slot in slots:
            if getattr(slot, "valueSource", None) == "fixed":
                continue
            # Prefer explicit slot_schema if provided
            if getattr(slot, "slot_schema", None):
                slot_schema = copy.deepcopy(slot.slot_schema)
                # Unwrap legacy function wrapper to plain JSON Schema parameters if present
                if isinstance(slot_schema, dict) and "function" in slot_schema:
                    params = slot_schema.get("function", {}).get("parameters")
                    if isinstance(params, dict):
                        # Extract the specific property's schema for this slot
                        props = (
                            params.get("properties", {})
                            if isinstance(params.get("properties"), dict)
                            else {}
                        )
                        if slot.name in props and isinstance(
                            props.get(slot.name), dict
                        ):
                            slot_schema = props.get(slot.name)
                        else:
                            # Fallback to the parameters block (best-effort) if property missing
                            slot_schema = params
                # Sanitize custom fields to adhere to OpenAI JSON schema
                self._remove_non_openai_fields(slot_schema)
            else:
                # Fall back to the legacy per-slot schema generator
                slot_schema = slot.to_openai_schema()

            if slot_schema is None:
                # Skip slots that explicitly indicate no schema (e.g., fixed values)
                continue

            properties[slot.name] = slot_schema

            if getattr(slot, "required", False):
                required.append(slot.name)

        return {
            "title": "SlotFillingOutput",
            "description": "Structured output for slot filling",
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _remove_non_openai_fields(self, schema_obj: dict | list) -> None:
        """Recursively remove non-OpenAI standard fields from schema objects.

        Args:
            schema_obj: The schema object to clean (dict or list)
        """
        if isinstance(schema_obj, dict):
            # Remove non-OpenAI fields
            fields_to_remove = [
                "valueSource",
                "fixed",
                "default",
                "prompt",
                "verified",
                "required",
                "repeatable",
            ]
            for field in fields_to_remove:
                schema_obj.pop(field, None)

            # Recursively clean nested objects
            for _key, value in list(schema_obj.items()):
                if isinstance(value, dict | list):
                    self._remove_non_openai_fields(value)
        elif isinstance(schema_obj, list):
            # Recursively clean list items
            for item in schema_obj:
                if isinstance(item, dict | list):
                    self._remove_non_openai_fields(item)

    def _apply_fixed_or_default_for_simple(
        self, slot: Slot, schema_props: dict[str, Any]
    ) -> None:
        """Apply fixed or default value for a simple (non-array) slot using schema properties.
        This preserves current behavior: fixed overrides; default fills only when empty; marks provenance and verified.
        """
        try:
            field_schema = schema_props.get(slot.name)
            if not isinstance(field_schema, dict):
                return
            # Fixed overrides model output
            if field_schema.get("valueSource") == "fixed" and "value" in field_schema:
                slot.value = self._convert_value_to_type(
                    field_schema["value"], field_schema.get("type", "string")
                )
                try:
                    slot.valueSource = "fixed"
                    slot.verified = True
                except Exception:
                    pass
                return

            # Default applies only if no model value
            default_value = None
            if "default" in field_schema:
                default_value = field_schema["default"]
            elif (
                field_schema.get("valueSource") == "default" and "value" in field_schema
            ):
                default_value = field_schema["value"]

            if default_value is not None and (slot.value in (None, "", [])):
                slot.value = self._convert_value_to_type(
                    default_value, field_schema.get("type", "string")
                )
                try:
                    slot.valueSource = "default"
                    slot.verified = True
                except Exception:
                    pass
        except Exception as e:
            log_context.warning(
                f"Failed to apply schema defaults for simple slot {slot.name}: {e}",
                extra={"operation": "slot_filling_evaluation"},
            )

    @handle_exceptions()
    def _fill_slots(
        self,
        slots: list[Slot],
        context: str,
        model_config: dict[str, Any],
        type: str = "chat",
    ) -> list[Slot]:
        """Fill slots.

        Args:
            slots: List of slots to fill
            context: Input context to extract values from
            model_config: Model configuration
            type: Type of slot filling operation (default: "chat")

        Returns:
            List of filled slots

        Raises:
            ModelError: If slot filling fails
            ValidationError: If input validation fails
        """
        # Format input
        prompt, system_prompt = self.format_slot_input(slots, context, type)
        log_context.info(
            "Slot filling input prepared",
            extra={
                "prompt": prompt,
                "system_prompt": system_prompt,
                "operation": "slot_filling_local",
            },
        )

        # Generate OpenAI schema from slots (let LLM handle descriptions/prompts)
        schema = self._slots_to_openai_schema(slots)
        log_context.info(
            "OpenAI schema generated",
            extra={
                "schema": schema,
                "operation": "slot_filling_local",
            },
        )

        # If there are no properties to extract (all slots are fixed or none require LLM),
        # skip calling the model and only apply fixed values. Defaults must be sent to LLM first.
        if not schema.get("properties"):
            log_context.info(
                "No variable slots to extract; skipping LLM call and applying fixed values only",
                extra={
                    "operation": "slot_filling_local",
                },
            )

            # Start from the input slots and fill fixed values for simple slots
            filled_slots = list(slots)
            simple_fixed_slots = [
                s
                for s in slots
                if not getattr(s, "slot_schema", None)
                and getattr(s, "valueSource", None) == "fixed"
            ]
            if simple_fixed_slots:
                name_to_slot = {s.name: s for s in filled_slots}
                for fixed_slot in simple_fixed_slots:
                    if fixed_slot.name in name_to_slot:
                        fixed_value = getattr(fixed_slot, "fixed", None)
                        if fixed_value is not None:
                            target = name_to_slot[fixed_slot.name]
                            target.value = fixed_value
                            try:
                                target.valueSource = "fixed"
                                target.verified = True
                            except Exception:
                                pass

            log_context.info(
                "Slot filling completed without LLM (fixed only)",
                extra={
                    "filled_slots": [slot.name for slot in filled_slots],
                    "operation": "slot_filling_local",
                },
            )
            return filled_slots

        # Get model response (guard against None)
        response = self.model_service.get_response_with_structured_output(
            prompt, schema, system_prompt
        )
        log_context.info(
            "Model response received",
            extra={
                "prompt": prompt,
                "system_prompt": system_prompt,
                "raw_response": response,
                "operation": "slot_filling_local",
            },
        )

        # Process response
        try:
            # If model produced no structured output, skip processing and just apply fixed/defaults
            if response is None:
                log_context.warning(
                    "Model returned None; skipping response processing and applying fixed/default values",
                    extra={
                        "operation": "slot_filling_local",
                    },
                )
                filled_slots = list(slots)
            else:
                filled_slots = self.process_slot_response(response, slots)

            # Apply default/fixed values for any slots that use slot_schema
            if slots:
                schema_slots = [s for s in slots if getattr(s, "slot_schema", None)]
                for schema_slot in schema_slots:
                    filled_slots = self._evaluate_and_fill_slot_values(
                        filled_slots, schema_slot
                    )
                # Also apply fixed/default values for simple, non-nested slots that were skipped
                simple_fixed_slots = [
                    s
                    for s in slots
                    if not getattr(s, "slot_schema", None)
                    and getattr(s, "valueSource", None) == "fixed"
                ]
                if simple_fixed_slots:
                    name_to_slot = {s.name: s for s in filled_slots}
                    for fixed_slot in simple_fixed_slots:
                        if fixed_slot.name in name_to_slot:
                            # Prefer explicit fixed value, else default
                            fixed_value = getattr(fixed_slot, "fixed", None)
                            if fixed_value is None:
                                fixed_value = getattr(fixed_slot, "default", None)
                            if fixed_value is not None:
                                target = name_to_slot[fixed_slot.name]
                                target.value = fixed_value
                                # Mark provenance to avoid verification later
                                try:
                                    target.valueSource = "fixed"
                                    target.verified = True
                                except Exception:
                                    pass

                # Apply defaults for simple non-fixed slots if model didn't provide values
                defaultable_slots = [
                    s
                    for s in slots
                    if getattr(s, "valueSource", None) != "fixed"
                    and getattr(s, "default", None) is not None
                ]
                if defaultable_slots:
                    name_to_slot = {s.name: s for s in filled_slots}
                    for default_slot in defaultable_slots:
                        if default_slot.name in name_to_slot:
                            target = name_to_slot[default_slot.name]
                            if getattr(target, "value", None) in (None, "", []):
                                target.value = default_slot.default
                                # Mark provenance for defaults; verification not needed
                                try:
                                    target.valueSource = "default"
                                    target.verified = True
                                except Exception:
                                    pass

            log_context.info(
                "Slot filling completed",
                extra={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "raw_response": response,
                    "filled_slots": [slot.name for slot in filled_slots],
                    "operation": "slot_filling_local",
                },
            )
            return filled_slots
        except Exception as e:
            log_context.error(
                "Failed to process slot filling response",
                extra={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "raw_response": response,
                    "error": str(e),
                    "operation": "slot_filling_local",
                },
            )
            raise ModelError(
                "Failed to process slot filling response",
                details={
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "raw_response": response,
                    "error": str(e),
                    "operation": "slot_filling_local",
                },
            ) from e

    def _evaluate_and_fill_slot_values(
        self, filled_slots: list[Slot], original_slot: Slot
    ) -> list[Slot]:
        """Evaluate and fill back default and fixed values from slot_schema structure.

        Args:
            filled_slots: List of slots with model-extracted values
            original_slot: Original slot with slot_schema structure

        Returns:
            Updated list of slots with proper values filled
        """
        if not original_slot.slot_schema:
            return filled_slots

        # For the new slot_schema structure, we need to handle the nested array structure
        for slot in filled_slots:
            if (
                slot.name == original_slot.name
                and slot.value
                and isinstance(slot.value, list)
            ):
                # This is an array slot, we need to process each item
                updated_items = []
                for item in slot.value:
                    if isinstance(item, dict):
                        # Apply fixed values to each item in the array using direct field access
                        updated_item = self._apply_fixed_values_direct(
                            item, original_slot.slot_schema
                        )
                        updated_items.append(updated_item)
                    else:
                        updated_items.append(item)
                slot.value = updated_items

                log_context.info(
                    f"Applied fixed values to array slot {slot.name}",
                    extra={
                        "slot_name": slot.name,
                        "updated_value": slot.value,
                        "operation": "slot_filling_evaluation",
                    },
                )
            elif slot.name == original_slot.name and (slot.value in (None, "", [])):
                # Simple (non-array) slot: apply fixed/default from schema via helper
                props = (
                    original_slot.slot_schema.get("function", {})
                    .get("parameters", {})
                    .get("properties", {})
                )
                self._apply_fixed_or_default_for_simple(slot, props)

        return filled_slots

    def _apply_fixed_values_direct(self, item: dict, slot_schema: dict) -> dict:
        """Apply fixed values directly to an item using field-level access.

        Args:
            item: Dictionary item to update
            slot_schema: The slot schema containing field definitions

        Returns:
            Updated item with fixed values applied
        """
        try:
            # Get the array items schema directly
            slot_name = None
            for key in (
                slot_schema.get("function", {})
                .get("parameters", {})
                .get("properties", {})
            ):
                slot_name = key
                break

            if not slot_name:
                return item

            # Get the array items schema
            array_schema = (
                slot_schema.get("function", {})
                .get("parameters", {})
                .get("properties", {})
                .get(slot_name, {})
            )
            items_schema = array_schema.get("items", {})
            properties = items_schema.get("properties", {})

            # Apply fixed values recursively to the item
            updated_item = self._apply_fixed_values_recursive(item, properties)

            return updated_item

        except Exception as e:
            log_context.error(
                "Error applying fixed values to item",
                extra={
                    "error": str(e),
                    "item": item,
                    "operation": "slot_filling_evaluation",
                },
            )
            return item

    def _apply_fixed_values_recursive(
        self, item: dict, properties: dict, path: str = ""
    ) -> dict:
        """Recursively apply fixed values to an item and its nested structures.

        Args:
            item: Dictionary item to update
            properties: Properties schema containing field definitions
            path: Current path for nested fields

        Returns:
            Updated item with fixed values applied
        """
        updated_item = item.copy()

        for field_name, field_schema in properties.items():
            current_path = f"{path}.{field_name}" if path else field_name

            # Check if this field has a fixed or default
            if field_schema.get("valueSource") == "fixed" and "value" in field_schema:
                # Convert and apply the fixed value
                fixed_value = self._convert_value_to_type(
                    field_schema["value"], field_schema.get("type", "string")
                )
                updated_item[field_name] = fixed_value
                log_context.info(
                    f"Applied fixed value to field {current_path}",
                    extra={
                        "field_path": current_path,
                        "fixed_value": fixed_value,
                        "original_value": field_schema["value"],
                        "type": field_schema.get("type", "string"),
                        "operation": "slot_filling_evaluation",
                    },
                )

            elif "default" in field_schema:
                # Default only applies if the model didn't provide a value
                if field_name not in updated_item or updated_item[field_name] in (
                    None,
                    "",
                    [],
                ):
                    default_value = self._convert_value_to_type(
                        field_schema["default"], field_schema.get("type", "string")
                    )
                    updated_item[field_name] = default_value
                    log_context.info(
                        f"Applied default value to field {current_path}",
                        extra={
                            "field_path": current_path,
                            "default_value": default_value,
                            "operation": "slot_filling_evaluation",
                        },
                    )

            # Handle nested objects
            elif field_schema.get("type") == "object" and field_name in updated_item:
                nested_props = field_schema.get("properties", {})
                if nested_props and isinstance(updated_item[field_name], dict):
                    updated_item[field_name] = self._apply_fixed_values_recursive(
                        updated_item[field_name], nested_props, current_path
                    )

            # Handle arrays of objects
            elif field_schema.get("type") == "array" and field_name in updated_item:
                items_schema = field_schema.get("items", {})
                if items_schema.get("type") == "object" and isinstance(
                    updated_item[field_name], list
                ):
                    nested_props = items_schema.get("properties", {})
                    if nested_props:
                        updated_item[field_name] = [
                            self._apply_fixed_values_recursive(
                                item, nested_props, current_path
                            )
                            for item in updated_item[field_name]
                        ]

        return updated_item

    def _convert_value_to_type(
        self, value: str | int | float | bool | list | dict | None, target_type: str
    ) -> str | int | float | bool | list | dict | None:
        """Convert a value to the specified type.

        Args:
            value: The value to convert
            target_type: The target type string

        Returns:
            The converted value
        """
        try:
            if target_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            elif target_type == "integer":
                return int(value)
            elif target_type == "number":
                return float(value)
            else:  # string or unknown type
                return str(value)
        except (ValueError, TypeError) as e:
            log_context.warning(
                f"Failed to convert value {value} to type {target_type}",
                extra={
                    "value": value,
                    "target_type": target_type,
                    "error": str(e),
                    "operation": "slot_filling_type_conversion",
                },
            )
            return value

    @handle_exceptions()
    def _verify_slot_local(
        self,
        slot: dict[str, Any],
        chat_history_str: str,
        model_config: dict[str, Any],
    ) -> tuple[bool, str]:
        """Verify slot value using local model.

        Args:
            slot: Slot to verify
            chat_history_str: Formatted chat history
            model_config: Model configuration

        Returns:
            Tuple of (is_valid, reason)

        Raises:
            ModelError: If slot verification fails
            ValidationError: If input validation fails
        """
        log_context.info(
            "Using local model for slot verification",
            extra={
                "slot": slot.get("name", "unknown"),
                "operation": "slot_verification_local",
            },
        )

        # Format input
        prompt = self.format_verification_input(slot, chat_history_str)
        log_context.info(
            "Slot verification input prepared",
            extra={
                "prompt": prompt,
                "operation": "slot_verification_local",
            },
        )

        # Get model response
        response = self.model_service.get_response(prompt)
        log_context.info(
            "Model response received",
            extra={
                "response": response,
                "operation": "slot_verification_local",
            },
        )

        # Process response
        try:
            is_valid, reason = self.process_verification_response(response)
            log_context.info(
                "Slot verification completed",
                extra={
                    "is_valid": is_valid,
                    "reason": reason,
                    "operation": "slot_verification_local",
                },
            )
            return is_valid, reason
        except Exception as e:
            log_context.error(
                "Failed to process slot verification response",
                extra={
                    "error": str(e),
                    "response": response,
                    "operation": "slot_verification_local",
                },
            )
            raise ModelError(
                "Failed to process slot verification response",
                details={
                    "error": str(e),
                    "response": response,
                    "operation": "slot_verification_local",
                },
            ) from e

    @handle_exceptions()
    def verify_slot(
        self,
        slot: Slot | dict[str, Any],
        chat_history_str: str,
        model_config: dict[str, Any],
    ) -> tuple[bool, str]:
        """Verify slot value using local model.

        Args:
            slot: Slot to verify (can be Slot object or dict)
            chat_history_str: Formatted chat history
            model_config: Model configuration

        Returns:
            Tuple of (is_valid, reason)

        Raises:
            ModelError: If slot verification fails
            ValidationError: If input validation fails
            APIError: If API request fails
        """

        # Short-circuit verification for fixed or default-provisioned slots
        try:
            if hasattr(slot, "valueSource"):
                vs = getattr(slot, "valueSource", None)
                if vs in ("fixed", "default"):
                    return True, f"{vs.capitalize()} value; no verification required"
            if isinstance(slot, dict):
                vs = slot.get("valueSource")
                if vs in ("fixed", "default"):
                    return True, f"{vs.capitalize()} value; no verification required"
        except Exception:
            pass

        # Handle both Slot objects and dictionaries
        slot_name = slot.name if hasattr(slot, "name") else slot.get("name", "unknown")

        # Convert Slot object to dictionary if needed
        slot_dict = slot
        if hasattr(slot, "__dict__"):
            slot_dict = {
                "name": slot.name,
                "value": getattr(slot, "value", None),
                "type": getattr(slot, "type", None),
                "description": getattr(slot, "description", None),
                "enum": getattr(slot, "enum", None),
                "required": getattr(slot, "required", False),
                "repeatable": getattr(slot, "repeatable", False),
                "prompt": getattr(slot, "prompt", None),
                "valueSource": getattr(slot, "valueSource", None),
                "fixed": getattr(slot, "fixed", None),
                "default": getattr(slot, "default", None),
                "verified": getattr(slot, "verified", False),
            }

        log_context.info(
            "Starting slot verification",
            extra={
                "slot": slot_name,
                "mode": "local",
                "operation": "slot_verification",
            },
        )

        try:
            is_valid, reason = self._verify_slot_local(
                slot_dict, chat_history_str, model_config
            )

            log_context.info(
                "Slot verification completed",
                extra={
                    "is_valid": is_valid,
                    "reason": reason,
                    "operation": "slot_verification",
                },
            )
            return is_valid, reason
        except Exception as e:
            log_context.error(
                "Slot verification failed",
                extra={
                    "error": str(e),
                    "slot": slot_name,
                    "operation": "slot_verification",
                },
            )
            raise

    @handle_exceptions()
    def fill_slots(
        self,
        slots: list[Slot],
        context: str,
        model_config: dict[str, Any],
        type: str = "chat",
    ) -> list[Slot]:
        """Fill slots from input context.

        Args:
            slots: List of slots to fill
            context: Input context to extract values from
            model_config: Model configuration
            type: Type of slot filling operation (default: "chat")

        Returns:
            List of filled slots

        Raises:
            ModelError: If slot filling fails
            ValidationError: If input validation fails
            APIError: If API request fails
        """
        log_context.info(
            "Starting slot filling",
            extra={
                "slots": [slot.name for slot in slots],
                "context_length": len(context),
                "mode": "local",
                "operation": "slot_filling",
            },
        )

        try:
            filled_slots = self._fill_slots(slots, context, model_config, type)

            log_context.info(
                "Slot filling completed",
                extra={
                    "filled_slots": [slot.name for slot in filled_slots],
                    "operation": "slot_filling",
                },
            )
            return filled_slots
        except Exception as e:
            log_context.error(
                "Slot filling failed",
                extra={
                    "error": str(e),
                    "slots": [slot.name for slot in slots],
                    "operation": "slot_filling",
                },
            )
            raise
