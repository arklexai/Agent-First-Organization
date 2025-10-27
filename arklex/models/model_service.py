"""Model interaction service for NLU operations.

This module provides services for interacting with language models,
handling model configuration, and processing model responses.
It manages the lifecycle of model interactions, including initialization,
message formatting, and response processing.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from arklex.models.llm_config import LLMConfig, load_llm
from arklex.utils.logging.logging_utils import LogContext

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
