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
from arklex.utils.exceptions import ModelError, ValidationError
from arklex.utils.logging_utils import LOG_MESSAGES, LogContext, handle_exceptions
from arklex.utils.model_config import MODEL

from .model_config import ModelConfig

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

    def __init__(self, model_config: dict[str, Any]) -> None:
        """Initialize the model service.

        Args:
            model_config: Configuration for the language model

        Raises:
            ModelError: If initialization fails
            ValidationError: If configuration is invalid
        """
        self.model_config = model_config
        self._validate_config()
        try:
            self.model = self._initialize_model()
        except Exception as e:
            log_context.error(
                LOG_MESSAGES["ERROR"]["INITIALIZATION_ERROR"].format(
                    service="ModelService", error=str(e)
                ),
                extra={
                    "error": str(e),
                    "service": "ModelService",
                    "operation": "initialization",
                },
            )
            raise ModelError(
                "Failed to initialize model service",
                details={
                    "error": str(e),
                    "service": "ModelService",
                    "operation": "initialization",
                },
            ) from e

    def _validate_config(self) -> None:
        """Validate the model configuration.

        Raises:
            ValidationError: If the configuration is invalid
        """
        required_fields = ["model_name", "model_type_or_path"]
        missing_fields = [
            field for field in required_fields if field not in self.model_config
        ]
        if missing_fields:
            log_context.error(
                "Missing required field",
                extra={
                    "missing_fields": missing_fields,
                    "operation": "config_validation",
                },
            )
            raise ValidationError(
                "Missing required field",
                details={
                    "missing_fields": missing_fields,
                    "operation": "config_validation",
                },
            )

        # Ensure API key is provided and not set to None or empty
        if "api_key" not in self.model_config or not self.model_config["api_key"]:
            # Don't set a default value - require explicit API key
            log_context.error(
                "API key is missing or empty",
                extra={
                    "operation": "config_validation",
                },
            )
            raise ValidationError(
                "API key is missing or empty",
                details={
                    "operation": "config_validation",
                },
            )

        # Set endpoint if not provided
        if "endpoint" not in self.model_config:
            self.model_config["endpoint"] = MODEL["endpoint"]

        # Validate API key presence
        from arklex.utils.provider_utils import validate_api_key_presence

        try:
            validate_api_key_presence(
                self.model_config.get("llm_provider", ""),
                self.model_config.get("api_key", ""),
            )
        except ValueError as e:
            log_context.error(
                "API key validation failed",
                extra={
                    "error": str(e),
                    "operation": "config_validation",
                },
            )
            raise ValidationError(
                "API key validation failed",
                details={
                    "error": str(e),
                    "operation": "config_validation",
                },
            ) from e

    @handle_exceptions()
    def _initialize_model(self) -> BaseChatModel:
        """Initialize the language model.

        Creates and configures a new model instance based on the service
        configuration.

        Returns:
            Initialized model instance

        Raises:
            ModelError: If model initialization fails
        """
        try:
            model = ModelConfig.get_model_instance(self.model_config)
            return ModelConfig.configure_response_format(model, self.model_config)
        except Exception as e:
            raise ModelError(
                "Failed to initialize model",
                details={
                    "error": str(e),
                    "model_config": self.model_config,
                    "operation": "model_initialization",
                },
            ) from e

    def get_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
        note: str | None = None,
    ) -> str:
        """Get response from the model.

        Sends a prompt to the model and returns its response as a string.
        Handles message formatting and response validation.

        Args:
            prompt: User prompt to send to the model
            model_config: Optional model configuration parameters. If not provided,
                         uses the instance's model_config.
            system_prompt: Optional system prompt for model context
            response_format: Optional format specification for the response
            note: Optional note for logging purposes

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

            if note:
                log_context.info(f"Model response for {note}: {response.content}")

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
            self.model_config.get("llm_provider", "").lower() == "openai"
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

        Args:
            response: Model's response containing extracted slot values (can be string or dict)
            slots: Original slot definitions (can be dict or Pydantic model)

        Returns:
            Updated list of slots with extracted values

        Raises:
            ValueError: If response parsing fails
        """
        try:
            # Handle both string and dict responses
            if isinstance(response, str):
                # Parse the JSON response if it's a string
                extracted_values = json.loads(response)
            elif isinstance(response, dict):
                # Use the dict directly if it's already a dict
                extracted_values = response
            else:
                raise ValueError(f"Unsupported response type: {type(response)}")

            # Update slot values
            for slot in slots:
                # Handle both dict and Pydantic model inputs
                if isinstance(slot, dict):
                    slot_name = slot.get("name", "")
                    if slot_name in extracted_values:
                        slot["value"] = extracted_values[slot_name]
                    else:
                        slot["value"] = None
                else:
                    slot_name = getattr(slot, "name", "")
                    if slot_name in extracted_values:
                        slot.value = extracted_values[slot_name]
                    else:
                        slot.value = None

            return slots
        except json.JSONDecodeError as e:
            log_context.error(f"Error parsing slot filling response: {str(e)}")
            raise ValueError(f"Failed to parse slot filling response: {str(e)}") from e
        except Exception as e:
            log_context.error(f"Error processing slot filling response: {str(e)}")
            raise ValueError(
                f"Failed to process slot filling response: {str(e)}"
            ) from e

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
