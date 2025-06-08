"""Model interaction service for NLU operations.

This module provides services for interacting with language models,
handling model configuration, and processing model responses.
It manages the lifecycle of model interactions, including initialization,
message formatting, and response processing.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from .model_config import ModelConfig

logger = logging.getLogger(__name__)


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

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initialize the model service.

        Creates a new model service instance with the specified configuration
        and initializes the language model.

        Args:
            model_config: Model configuration dictionary containing:
                - model_type_or_path: Model identifier
                - llm_provider: Provider name

        Raises:
            ValueError: If model initialization fails
        """
        self.model_config = model_config
        self.model = self._initialize_model()

    def _initialize_model(self) -> BaseChatModel:
        """Initialize the language model.

        Creates and configures a new model instance based on the service
        configuration.

        Returns:
            Initialized model instance

        Raises:
            ValueError: If model initialization fails
        """
        try:
            model = ModelConfig.get_model_instance(self.model_config)
            return ModelConfig.configure_response_format(model, self.model_config)
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise ValueError(f"Failed to initialize model: {str(e)}")

    def _format_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> List[Union[HumanMessage, SystemMessage]]:
        """Format messages for model input.

        Creates a list of formatted messages for model input, including
        an optional system prompt and the user prompt.

        Args:
            prompt: User prompt to send to the model
            system_prompt: Optional system prompt for model context

        Returns:
            List of formatted messages for model input
        """
        messages = []
        if system_prompt:
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
        self, intent_name: str, sample_utterances: List[str], count: int
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
        intent_v: List[Dict[str, Any]],
        count: int,
        idx2intents_mapping: Dict[str, str],
    ) -> Tuple[str, str, str, int]:
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
        model_config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = None,
        note: Optional[str] = None,
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
            # Use instance model_config if none provided
            config = model_config if model_config is not None else self.model_config

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
                logger.info(f"Model response for {note}: {response.content}")

            return response.content
        except Exception as e:
            logger.error(f"Error getting model response: {str(e)}")
            raise ValueError(f"Failed to get model response: {str(e)}")

    def get_json_response(
        self,
        prompt: str,
        model_config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            logger.error(f"Error parsing JSON response: {str(e)}")
            raise ValueError(f"Failed to parse JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting JSON response: {str(e)}")
            raise ValueError(f"Failed to get JSON response: {str(e)}")

    def format_intent_input(
        self, intents: Dict[str, List[Dict[str, Any]]], chat_history_str: str
    ) -> Tuple[str, Dict[str, str]]:
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
        idx2intents_mapping: Dict[str, str] = {}
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

Please choose the most appropriate intent by providing only the corresponding number."""

        return prompt, idx2intents_mapping

    def format_slot_input(
        self, slots: List[Dict[str, Any]], context: str, type: str = "chat"
    ) -> Tuple[str, str]:
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
            "Return the extracted values in JSON format."
        )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Slot definitions:\n" + "\n".join(slot_definitions) + "\n\n"
            "Please extract the values for the defined slots from the context. "
            "Return the results in JSON format with slot names as keys and "
            "extracted values as values. If a slot value cannot be found, "
            "set its value to null."
        )

        return user_prompt, system_prompt

    def process_slot_response(
        self, response: str, slots: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
                        setattr(slot, "value", extracted_values[slot_name])
                    else:
                        setattr(slot, "value", None)

            return slots
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing slot filling response: {str(e)}")
            raise ValueError(f"Failed to parse slot filling response: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing slot filling response: {str(e)}")
            raise ValueError(f"Failed to process slot filling response: {str(e)}")
