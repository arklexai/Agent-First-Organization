"""Model interaction service for NLU operations.

This module provides services for interacting with language models,
handling model configuration, and processing model responses.
It manages the lifecycle of model interactions, including initialization,
message formatting, and response processing.
"""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from arklex.orchestrator.NLU.utils.formatters import (
    format_verification_input as format_verification_input_formatter,
)
from arklex.utils.llm_config import LLMConfig, load_llm
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class ModelService:
    """Service for interacting with language models.

    This class manages the interaction with language models, handling
    message formatting, response processing, and error handling.

    Key responsibilities:
    - Model initialization and configuration
    - Message formatting and prompt management
    - Response processing and validation
    - Error handling and logging

    Attributes:
        model_config: Configuration for the language model
        model: Initialized model instance
    """

    def __init__(self, llm_config: LLMConfig) -> None:
        """Initialize the model service.

        Args:
            llm_config: Configuration for the language model

        Raises:
            ModelError: If initialization fails
        """
        self.llm_config = llm_config
        self.model: BaseChatModel = load_llm(llm_config)

    def get_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Get response from the model.

        Sends a prompt to the model and returns its response as a string.
        Handles message formatting and response validation.

        Args:
            prompt: User prompt to send to the model
            system_prompt: Optional system prompt for model context

        Returns:
            Model response as string

        Raises:
            ValueError: If model response is invalid or empty
        """
        try:
            # Format messages with system prompt if provided
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            # Get response from model
            response = self.model.invoke(messages)
            if not response or not response.content:
                raise ValueError("Empty response from model")
            return response.content
        except Exception as e:
            log_context.error(f"Error getting model response: {str(e)}")
            raise ValueError(f"Failed to get model response: {str(e)}") from e

    def get_response_with_structured_output(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Get response from the model with structured output."""
        # Check if the model is an OpenAI model by checking the model_config
        is_openai_model = (
            self.llm_config.llm_provider.lower() == "openai"
            or "openai" in str(self.model).lower()
        )

        if is_openai_model:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            llm = self.model.with_structured_output(schema)
            return llm.invoke(messages)
        else:
            return self.get_response(prompt, system_prompt)

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
            has_any_slot_schema = any(
                getattr(s, "slot_schema", None) for s in slots
            )
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
                slot_name = s.get("name") if isinstance(s, dict) else getattr(s, "name", None)
                if not slot_name:
                    continue
                value = extracted_values.get(slot_name)
                if isinstance(s, dict):
                    s["value"] = value.get(slot_name)
                else:
                    s.value = value.get(slot_name)
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


class DummyModelService(ModelService):
    """A dummy model service for testing purposes.

    This class provides mock implementations of model service methods
    for use in testing scenarios.
    """

    def format_slot_input(
        self, slots: list[dict[str, Any]], context: str, type: str = "chat"
    ) -> tuple[str, str]:
        """Format slot input for testing.

        Args:
            slots: List of slot definitions
            context: Context string
            type: Type of input format (default: "chat")

        Returns:
            Tuple[str, str]: Formatted input and context
        """
        return super().format_slot_input(slots, context, type)

    def get_response(
        self,
        prompt: str,
        model_config: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        response_format: str | None = None,
        note: str | None = None,
    ) -> str:
        """Get a mock response for testing.

        Args:
            prompt: Input prompt
            model_config: Optional model configuration
            system_prompt: Optional system prompt
            response_format: Optional response format
            note: Optional note

        Returns:
            str: Mock response for testing
        """
        return "1) others"

    def process_slot_response(
        self, response: str, slots: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process mock slot response for testing.

        Args:
            response: Mock response string
            slots: List of slot definitions

        Returns:
            List[Dict[str, Any]]: Processed slot values
        """
        return super().process_slot_response(response, slots)

    def format_verification_input(
        self, slot: dict[str, Any], chat_history_str: str
    ) -> tuple[str, str]:
        """Format verification input for testing.

        Args:
            slot: Slot definition
            chat_history_str: Chat history string

        Returns:
            Tuple[str, str]: Formatted input and context
        """
        return super().format_verification_input(slot, chat_history_str)

    def process_verification_response(self, response: str) -> tuple[bool, str]:
        """Process mock verification response for testing.

        Args:
            response: Mock response string

        Returns:
            Tuple[bool, str]: Verification result and explanation
        """
        return super().process_verification_response(response)
