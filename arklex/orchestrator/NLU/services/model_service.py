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

from arklex.orchestrator.NLU.core.base import (
    IntentResponse,
    SlotResponse,
    VerificationResponse,
)
from arklex.orchestrator.NLU.services.api_service import APIClientService
from arklex.orchestrator.NLU.utils.formatters import (
    format_verification_input as format_verification_input_formatter,
)
from arklex.orchestrator.NLU.utils.validators import (
    validate_intent_response,
    validate_slot_response,
    validate_verification_response,
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
            self.api_service = APIClientService(base_url=self.model_config["endpoint"])
            self.model = self._initialize_model()
            log_context.info(
                "ModelService initialized successfully",
                extra={
                    "model_name": model_config.get("model_name"),
                    "operation": "initialization",
                },
            )
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
    async def process_text(
        self, text: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Process text through the model.

        Args:
            text: Input text to process
            context: Optional context information

        Returns:
            Dict[str, Any]: Processed response from the model

        Raises:
            ValidationError: If input validation fails
            ModelError: If model processing fails
        """
        if not isinstance(text, str):
            log_context.error(
                "Invalid input text",
                extra={
                    "text": text,
                    "type": type(text).__name__,
                    "operation": "text_processing",
                },
            )
            raise ValidationError(
                "Invalid input text",
                details={
                    "text": text,
                    "type": type(text).__name__,
                    "operation": "text_processing",
                },
            )

        if not text or not text.strip():
            log_context.error(
                "Text cannot be empty or whitespace-only",
                extra={
                    "text": text,
                    "operation": "text_processing",
                },
            )
            raise ValidationError(
                "Text cannot be empty or whitespace-only",
                details={
                    "text": text,
                    "operation": "text_processing",
                },
            )

        try:
            response = await self._make_model_request(
                {
                    "text": text,
                    "context": context,
                    "model": self.model_config["model_name"],
                }
            )
            return response
        except Exception as e:
            log_context.error(
                str(e),
                extra={
                    "error": str(e),
                    "text": text,
                    "operation": "text_processing",
                },
            )
            raise ModelError(
                str(e),
                details={
                    "error": str(e),
                    "text": text,
                    "operation": "text_processing",
                },
            ) from e

    async def _make_model_request(self, text: str | dict[str, Any]) -> dict[str, Any]:
        """Make a request to the model.

        Args:
            text: Input text or dictionary to send to the model

        Returns:
            Dict[str, Any]: Model response

        Raises:
            ModelError: If the request fails
        """
        try:
            if isinstance(text, dict):
                prompt = text.get("text", "")
                context = text.get("context", {})
                text.get("model", self.model_config.get("model_name"))
                messages = self._format_messages(prompt, context)
            else:
                messages = self._format_messages(text)

            response = await self.model.agenerate([messages])
            return {"result": response.generations[0][0].text}
        except Exception as e:
            log_context.error(
                str(e),
                extra={
                    "error": str(e),
                    "text": text,
                    "operation": "model_request",
                },
            )
            raise ModelError(
                str(e),
                details={
                    "error": str(e),
                    "text": text,
                    "operation": "model_request",
                },
            ) from e

    @handle_exceptions()
    async def predict_intent(self, text: str) -> IntentResponse:
        """Predict intent from input text.

        Args:
            text: Input text to predict intent from

        Returns:
            IntentResponse: Predicted intent and confidence

        Raises:
            ValidationError: If input validation fails
            ModelError: If model prediction fails
        """
        # Validate input
        if not text or not isinstance(text, str):
            log_context.error(
                "Invalid input text",
                extra={
                    "text": text,
                    "type": type(text).__name__,
                    "operation": "intent_prediction",
                },
            )
            raise ValidationError(
                "Invalid input text",
                details={
                    "text": text,
                    "type": type(text).__name__,
                    "operation": "intent_prediction",
                },
            )

        # Get model response
        response = await self.model.invoke(text, response_format="intent")

        # Validate response
        if not response or not getattr(response, "content", None):
            log_context.error(
                "Empty response from model",
                extra={
                    "response": response,
                    "operation": "intent_prediction",
                },
            )
            raise ModelError(
                "Empty response from model",
                details={
                    "response": response,
                    "operation": "intent_prediction",
                },
            )

        # Parse and validate intent response
        try:
            intent_data = json.loads(response.content)

            # Validate that the response has the expected structure
            if not isinstance(intent_data, dict) or "intent" not in intent_data:
                log_context.error(
                    "Invalid intent response structure",
                    extra={
                        "response": response.content,
                        "operation": "intent_prediction",
                    },
                )
                raise ValidationError(
                    "Invalid intent response structure",
                    details={
                        "response": response.content,
                        "operation": "intent_prediction",
                    },
                )

            # For now, we'll create a simple mapping since we don't have the full context
            # In a real implementation, this would come from the intent definitions
            idx2intents_mapping = {"1": "test_intent"}  # Default mapping for testing

            # Convert the intent_data to a string for validation
            intent_str = str(intent_data["intent"])
            validated_response = validate_intent_response(
                intent_str, idx2intents_mapping
            )

            log_context.info(
                "Intent prediction successful",
                extra={
                    "intent": validated_response,
                    "operation": "intent_prediction",
                },
            )
            # Create a simple IntentResponse with the validated intent
            return IntentResponse(intent=validated_response, confidence=0.9)
        except json.JSONDecodeError as e:
            log_context.error(
                "Failed to parse model response",
                extra={
                    "error": str(e),
                    "response": response.content,
                    "operation": "intent_prediction",
                },
            )
            raise ModelError(
                "Failed to parse model response",
                details={
                    "error": str(e),
                    "response": response.content,
                    "operation": "intent_prediction",
                },
            ) from e
        except ValidationError as e:
            log_context.error(
                "Invalid intent response format",
                extra={
                    "error": str(e),
                    "response": response.content,
                    "operation": "intent_prediction",
                },
            )
            raise ValidationError(
                "Invalid intent response format",
                details={
                    "error": str(e),
                    "response": response.content,
                    "operation": "intent_prediction",
                },
            ) from e

    @handle_exceptions()
    async def fill_slots(self, text: str, intent: str) -> SlotResponse:
        """Fill slots based on input text and intent.

        Args:
            text: Input text to extract slots from
            intent: Intent to use for slot filling

        Returns:
            SlotResponse: Extracted slots and their values

        Raises:
            ValidationError: If input validation fails
            ModelError: If slot filling fails
        """
        # Validate inputs
        if not text or not isinstance(text, str):
            log_context.error(
                "Invalid input text",
                extra={
                    "text": text,
                    "type": type(text).__name__,
                    "operation": "slot_filling",
                },
            )
            raise ValidationError(
                "Invalid input text",
                details={
                    "text": text,
                    "type": type(text).__name__,
                    "operation": "slot_filling",
                },
            )
        if not intent or not isinstance(intent, str):
            log_context.error(
                "Invalid intent",
                extra={
                    "intent": intent,
                    "type": type(intent).__name__,
                    "operation": "slot_filling",
                },
            )
            raise ValidationError(
                "Invalid intent",
                details={
                    "intent": intent,
                    "type": type(intent).__name__,
                    "operation": "slot_filling",
                },
            )

        # Get model response
        response = await self.model.invoke(text, response_format="slots", intent=intent)

        # Validate response
        if not response or not getattr(response, "content", None):
            log_context.error(
                "Empty response from model",
                extra={
                    "response": response,
                    "operation": "slot_filling",
                },
            )
            raise ModelError(
                "Empty response from model",
                details={
                    "response": response,
                    "operation": "slot_filling",
                },
            )

        # Parse and validate slot response
        try:
            slot_data = json.loads(response.content)
            validated_response = validate_slot_response(slot_data)
            log_context.info(
                "Slot filling successful",
                extra={
                    "slots": validated_response.get("slots"),
                    "operation": "slot_filling",
                },
            )
            return SlotResponse(**validated_response)
        except json.JSONDecodeError as e:
            log_context.error(
                "Failed to parse slot response",
                extra={
                    "error": str(e),
                    "response": response.content,
                    "operation": "slot_filling",
                },
            )
            raise ModelError(
                "Failed to parse slot response",
                details={
                    "error": str(e),
                    "response": response.content,
                    "operation": "slot_filling",
                },
            ) from e
        except ValidationError as e:
            log_context.error(
                "Invalid slot response format",
                extra={
                    "error": str(e),
                    "response": response.content,
                    "operation": "slot_filling",
                },
            )
            raise ValidationError(
                "Invalid slot response format",
                details={
                    "error": str(e),
                    "response": response.content,
                    "operation": "slot_filling",
                },
            ) from e

    @handle_exceptions()
    async def verify_slots(
        self, text: str, slots: dict[str, Any]
    ) -> VerificationResponse:
        """Verify slots against input text.

        Args:
            text: Input text to verify slots against
            slots: Dictionary of slots to verify

        Returns:
            VerificationResponse: Verification results for each slot

        Raises:
            ValidationError: If input validation fails
            ModelError: If slot verification fails
        """
        if not text or not isinstance(text, str):
            log_context.error(
                "Invalid input text",
                extra={
                    "text": text,
                    "type": type(text).__name__,
                    "operation": "slot_verification",
                },
            )
            raise ValidationError(
                "Invalid input text",
                details={
                    "text": text,
                    "type": type(text).__name__,
                    "operation": "slot_verification",
                },
            )
        if not slots or not isinstance(slots, dict):
            log_context.error(
                "Invalid slots",
                extra={
                    "slots": slots,
                    "type": type(slots).__name__,
                    "operation": "slot_verification",
                },
            )
            raise ValidationError(
                "Invalid slots",
                details={
                    "slots": slots,
                    "type": type(slots).__name__,
                    "operation": "slot_verification",
                },
            )

        # Get model response
        response = await self.model.invoke(
            text, response_format="verification", slots=slots
        )

        # Validate response
        if not response or not getattr(response, "content", None):
            log_context.error(
                "Empty response from model",
                extra={
                    "response": response,
                    "operation": "slot_verification",
                },
            )
            raise ModelError(
                "Empty response from model",
                details={
                    "response": response,
                    "operation": "slot_verification",
                },
            )

        # Parse and validate verification response
        try:
            verification_data = json.loads(response.content)
            validated_response = validate_verification_response(verification_data)
            log_context.info(
                "Slot verification successful",
                extra={
                    "verification": validated_response,
                    "operation": "slot_verification",
                },
            )
            return VerificationResponse(**validated_response)
        except json.JSONDecodeError as e:
            log_context.error(
                "Failed to parse verification response",
                extra={
                    "error": str(e),
                    "response": response.content,
                    "operation": "slot_verification",
                },
            )
            raise ModelError(
                "Failed to parse verification response",
                details={
                    "error": str(e),
                    "response": response.content,
                    "operation": "slot_verification",
                },
            ) from e
        except ValidationError as e:
            log_context.error(
                "Invalid verification response format",
                extra={
                    "error": str(e),
                    "response": response.content,
                    "operation": "slot_verification",
                },
            )
            raise ValidationError(
                "Invalid verification response format",
                details={
                    "error": str(e),
                    "response": response.content,
                    "operation": "slot_verification",
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

    def _format_messages(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> list[HumanMessage | SystemMessage]:
        """Format messages for the model.

        Args:
            prompt: User prompt to send to the model
            context: Optional context information

        Returns:
            List[Union[HumanMessage, SystemMessage]]: Formatted messages
        """
        messages = []
        if context:
            system_prompt = f"Context: {json.dumps(context)}"
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages

    def _format_intent_definition(
        self, intent_name: str, definition: str, count: int
    ) -> str:
        """Format a single intent definition.

        Args:
            intent_name: Name of the intent
            definition: Intent definition text
            count: Intent number in sequence

        Returns:
            Formatted intent definition string
        """
        return f"{count}) {intent_name}: {definition}\n"

    def _format_intent_exemplars(
        self, intent_name: str, sample_utterances: list[str], count: int
    ) -> str:
        """Format sample utterances for an intent.

        Args:
            intent_name: Name of the intent
            sample_utterances: List of example utterances
            count: Intent number in sequence

        Returns:
            Formatted exemplars string
        """
        if not sample_utterances:
            return ""
        exemplars = "\n".join(sample_utterances)
        return f"{count}) {intent_name}: \n{exemplars}\n"

    def _process_intent(
        self,
        intent_k: str,
        intent_v: list[dict[str, Any]],
        count: int,
        idx2intents_mapping: dict[str, str],
    ) -> tuple[str, str, str, int]:
        """Process a single intent and its variations.

        Args:
            intent_k: Intent key/name
            intent_v: List of intent definitions
            count: Current count for numbering
            idx2intents_mapping: Mapping of indices to intent names

        Returns:
            Tuple containing:
                - definition_str: Formatted definitions
                - exemplars_str: Formatted exemplars
                - intents_choice: Formatted choices
                - new_count: Updated count
        """
        definition_str = ""
        exemplars_str = ""
        intents_choice = ""

        if len(intent_v) == 1:
            intent_name = intent_k
            idx2intents_mapping[str(count)] = intent_name
            definition = intent_v[0].get("attribute", {}).get("definition", "")
            sample_utterances = (
                intent_v[0].get("attribute", {}).get("sample_utterances", [])
            )

            if definition:
                definition_str += self._format_intent_definition(
                    intent_name, definition, count
                )
            if sample_utterances:
                exemplars_str += self._format_intent_exemplars(
                    intent_name, sample_utterances, count
                )
            intents_choice += f"{count}) {intent_name}\n"

            count += 1
        else:
            for idx, intent in enumerate(intent_v):
                intent_name = f"{intent_k}__<{idx}>"
                idx2intents_mapping[str(count)] = intent_name
                definition = intent.get("attribute", {}).get("definition", "")
                sample_utterances = intent.get("attribute", {}).get(
                    "sample_utterances", []
                )

                if definition:
                    definition_str += self._format_intent_definition(
                        intent_name, definition, count
                    )
                if sample_utterances:
                    exemplars_str += self._format_intent_exemplars(
                        intent_name, sample_utterances, count
                    )
                intents_choice += f"{count}) {intent_name}\n"

                count += 1

        return definition_str, exemplars_str, intents_choice, count

    def get_response(
        self,
        prompt: str,
        model_config: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        response_format: str | None = None,
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

    def get_json_response(
        self,
        prompt: str,
        model_config: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Get JSON response from the model.

        Sends a prompt to the model and returns its response as a parsed
        JSON object. Handles message formatting and JSON validation.

        Args:
            prompt: User prompt to send to the model
            model_config: Optional model configuration parameters. If not provided,
                         uses the instance's model_config.
            system_prompt: Optional system prompt for model context

        Returns:
            Parsed JSON response

        Raises:
            ValueError: If JSON parsing fails or response is invalid
        """
        try:
            response = self.get_response(prompt, model_config, system_prompt)
            return json.loads(response)
        except json.JSONDecodeError as e:
            log_context.error(f"Error parsing JSON response: {str(e)}")
            raise ValueError(f"Failed to parse JSON response: {str(e)}") from e
        except Exception as e:
            log_context.error(f"Error getting JSON response: {str(e)}")
            raise ValueError(f"Failed to get JSON response: {str(e)}") from e

    def format_intent_input(
        self, intents: dict[str, list[dict[str, Any]]], chat_history_str: str
    ) -> tuple[str, dict[str, str]]:
        """Format input for intent detection.

        Creates a formatted prompt for intent detection based on the
        provided intents and chat history. Also generates a mapping
        from indices to intent names.

        Args:
            intents: Dictionary of intents containing:
                - intent_name: List of intent definitions
                - attribute: Intent attributes (definition, sample_utterances)
            chat_history_str: Formatted chat history

        Returns:
            Tuple containing:
                - formatted_prompt: Formatted prompt for intent detection
                - idx2intents_mapping: Mapping from indices to intent names
        """
        definition_str = ""
        exemplars_str = ""
        intents_choice = ""
        idx2intents_mapping: dict[str, str] = {}
        count = 1

        for intent_k, intent_v in intents.items():
            def_str, ex_str, choice_str, new_count = self._process_intent(
                intent_k, intent_v, count, idx2intents_mapping
            )
            definition_str += def_str
            exemplars_str += ex_str
            intents_choice += choice_str
            count = new_count

        prompt = f"""Given the following intents and their definitions, determine the most appropriate intent for the user's input.

Intent Definitions:
{definition_str}

Sample Utterances:
{exemplars_str}

Available Intents:
{intents_choice}

Chat History:
{chat_history_str}

Please choose the most appropriate intent by providing the corresponding intent number and intent name in the format of 'intent_number) intent_name'."""

        return prompt, idx2intents_mapping

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
        self, response: str, slots: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Process the model's response for slot filling.

        Parses the model's response and updates the slot values accordingly.

        Args:
            response: Model's response containing extracted slot values
            slots: Original slot definitions (can be dict or Pydantic model)

        Returns:
            Updated list of slots with extracted values

        Raises:
            ValueError: If response parsing fails
        """
        try:
            # Parse the JSON response
            extracted_values = json.loads(response)

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

    def get_json_response(
        self,
        prompt: str,
        model_config: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Get a mock JSON response for testing.

        Args:
            prompt: Input prompt
            model_config: Optional model configuration
            system_prompt: Optional system prompt

        Returns:
            dict[str, Any]: Mock JSON response for testing
        """
        # Handle None or empty prompts
        if prompt is None:
            prompt = ""

        # Return appropriate mock JSON responses based on the input
        return {"result": "mock_response", "status": "success"}
