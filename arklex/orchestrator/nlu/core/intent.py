"""Intent detection implementation for NLU.

This module provides the core implementation for intent detection functionality,
supporting both local model-based and remote API-based approaches. It implements
the BaseNLU interface to provide a unified way of detecting user intents from
input text.

The module includes:
- IntentDetector: Main class for intent detection
- Support for both local and remote intent detection
- Integration with language models and APIs
"""

from typing import Any

from arklex.models.model_service import ModelService
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


DEFAULT_INTENT_NAME = "others"


class IntentDetector:
    """Intent detection implementation.

    This class provides functionality for detecting intents from user input,
    supporting both local model-based and remote API-based approaches. It
    implements the BaseNLU interface and can be configured to use either
    a local language model or a remote API service.

    Key features:
    - Dual-mode operation (local/remote)
    - Integration with language models
    - Support for chat history context
    - Intent mapping and validation

    Attributes:
        model_service: Service for local model-based intent detection
    """

    def __init__(
        self,
        model_service: ModelService,
    ) -> None:
        """Initialize the intent detector.

        Args:
            model_service: Service for local model-based intent detection

        Raises:
            ValidationError: If model_service is not provided
        """
        self.model_service = model_service

    def _format_intent_input(
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

        prompt = f"""Given the following intents and their definitions, determine the most appropriate intent for the user's last input.

Intent Definitions:
{definition_str}

Sample Utterances:
{exemplars_str}

Available Intents:
{intents_choice}

Chat History:
{chat_history_str}
"""

        return prompt, idx2intents_mapping

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
                definition_str += self._process_intent_definition(
                    intent_name, definition, count
                )
            if sample_utterances:
                exemplars_str += self._process_intent_exemplars(
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
                    definition_str += self._process_intent_definition(
                        intent_name, definition, count
                    )
                if sample_utterances:
                    exemplars_str += self._process_intent_exemplars(
                        intent_name, sample_utterances, count
                    )
                intents_choice += f"{count}) {intent_name}\n"

                count += 1

        return definition_str, exemplars_str, intents_choice, count

    def _process_intent_definition(
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

    def _process_intent_exemplars(
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

    def _intent_to_openai_schema(
        self, idx2intents_mapping: dict[str, str]
    ) -> dict[str, Any]:
        """Convert intents to OpenAI schema for structured output."""
        return {
            "title": "IntentDetectionOutput",
            "description": "Structured output for intent detection",
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "The detected intent name",
                    "enum": list(idx2intents_mapping.values()),
                },
            },
            "required": ["intent"],
        }

    def _detect_intent(self, prompt: str, idx2intents_mapping: dict[str, str]) -> str:
        """Detect intent.

        Args:
            intents: Dictionary of available intents
            chat_history_str: Formatted chat history

        Returns:
            Predicted intent name

        Raises:
            ModelError: If intent detection fails
            ValidationError: If input validation fails
        """
        # Get model response
        response = self.model_service.get_response(prompt)
        log_context.info(
            f"Model response received:\nResponse: {response}",
            extra={
                "prompt": prompt,
                "raw_response": response,
                "operation": "intent_detection",
            },
        )
        _, pred_intent = [i.strip() for i in response.split(")", 1)]
        return pred_intent

    def _detect_intent_with_structured_output(
        self,
        prompt: str,
        idx2intents_mapping: dict[str, str],
    ) -> str:
        """Detect intent with structured output."""
        schema = self._intent_to_openai_schema(idx2intents_mapping)
        response = self.model_service.get_response_with_structured_output(
            prompt, schema
        )
        log_context.info(
            f"Model response received:\nResponse: {response}",
            extra={
                "prompt": prompt,
                "raw_response": response,
                "operation": "intent_detection",
            },
        )
        pred_intent = response.get("intent", DEFAULT_INTENT_NAME)
        return pred_intent

    def predict_intent(
        self,
        intents: dict[str, list[dict[str, Any]]],
        chat_history_str: str,
    ) -> str:
        """Predict intent from input text.

        Analyzes the input text to determine the most likely intent based on
        the available intent definitions and chat history context. Can operate
        in either local model-based or remote API-based mode.

        Args:
            intents: Dictionary mapping intent names to their definitions and attributes
            chat_history_str: Formatted chat history providing conversation context
            model_config: Configuration parameters for the language model

        Returns:
            The predicted intent name as a string

        Raises:
            ModelError: If intent detection fails
            ValidationError: If input validation fails
            APIError: If API request fails
        """

        try:
            # Format input and get mapping
            prompt, idx2intents_mapping = self._format_intent_input(
                intents, chat_history_str
            )
            log_context.info(
                f"Intent detection input prepared:\nPrompt: {prompt}\n\nMapping: {idx2intents_mapping}",
                extra={
                    "prompt": prompt,
                    "mapping": idx2intents_mapping,
                    "operation": "intent_detection",
                },
            )
            intent = self._detect_intent_with_structured_output(
                prompt, idx2intents_mapping
            )
            if intent not in idx2intents_mapping.values():
                log_context.warning(
                    f"Predicted intent not in mapping:\nPredicted intent: {intent}\n\nAvailable intents: {list(idx2intents_mapping.values())}",
                )
                intent = DEFAULT_INTENT_NAME
            log_context.info(f"Predicted intent: {intent}")
            return intent
        except Exception as e:
            log_context.error(
                f"Intent prediction failed: {str(e)}",
                details={
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                    "operation": "intent_prediction",
                },
            )
            return DEFAULT_INTENT_NAME

    def execute(
        self,
        intents: dict[str, list[dict[str, Any]]],
        chat_history_str: str,
    ) -> str:
        """Execute intent detection.

        This method is an alias for predict_intent, implementing the BaseNLU
        interface. It provides the same functionality as predict_intent.

        Args:
            text: Input text to analyze
            intents: Dictionary of available intents
            chat_history_str: Formatted chat history
            model_config: Model configuration

        Returns:
            Predicted intent name

        Raises:
            ModelError: If intent detection fails
            ValidationError: If input validation fails
            APIError: If API request fails
        """
        return self.predict_intent(intents, chat_history_str)
