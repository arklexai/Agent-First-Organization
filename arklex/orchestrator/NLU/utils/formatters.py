"""Input/output formatting utilities for NLU.

This module provides utilities for formatting inputs and outputs for NLU
operations, ensuring consistent and structured data presentation. It
handles the formatting of prompts, responses, and intermediate data
for various NLU tasks.

Key features:
- Intent detection prompt formatting
- Slot filling prompt formatting
- Slot verification prompt formatting
- Response structure formatting
"""

from typing import Dict, List, Any, Tuple
from arklex.utils.slot import SlotInputList


def _format_slot_description(slot: Any) -> str:
    """Format a single slot description.

    Args:
        slot: Slot object containing name, description, required flag, and type

    Returns:
        Formatted slot description string
    """
    description = f"- {slot.name}: {slot.description}"
    if slot.required:
        description += " (required)"
    if slot.type:
        description += f" (type: {slot.type})"
    return description


def _format_slot_prompt(context: str, slot_str: str) -> str:
    """Format the slot filling prompt template.

    Args:
        context: Input context to extract values from
        slot_str: Formatted slot descriptions

    Returns:
        Formatted prompt string
    """
    return f"""Given the following context and slot requirements, extract the values for each slot.

Context:
{context}

Required Slots:
{slot_str}

Please provide the values in JSON format with the following structure:
{{
    "slots": [
        {{
            "name": "slot_name",
            "value": "extracted_value"
        }}
    ]
}}"""


def format_intent_input(
    intents: Dict[str, List[Dict[str, Any]]], chat_history_str: str
) -> Tuple[str, Dict[str, str]]:
    """Format input for intent detection.

    This function formats the input data for intent detection, creating
    a structured prompt that includes intent definitions, sample utterances,
    and chat history. It also generates a mapping from indices to intent names.

    Args:
        intents: Dictionary mapping intent names to their definitions and attributes:
            - intent_name: List of intent definitions
            - attribute: Intent attributes (definition, sample_utterances)
        chat_history_str: Formatted chat history providing conversation context

    Returns:
        A tuple containing:
            - str: Formatted prompt for intent detection
            - Dict[str, str]: Mapping from indices to intent names

    Note:
        The function handles both single and multiple intent definitions
        per intent name, creating appropriate mappings for each case.
    """
    intents_choice = ""
    definition_str = ""
    exemplars_str = ""
    idx2intents_mapping: Dict[str, str] = {}
    count = 1

    for intent_k, intent_v in intents.items():
        if len(intent_v) == 1:
            intent_name = intent_k
            idx2intents_mapping[str(count)] = intent_name
            definition = intent_v[0].get("attribute", {}).get("definition", "")
            sample_utterances = (
                intent_v[0].get("attribute", {}).get("sample_utterances", [])
            )

            if definition:
                definition_str += f"{count}) {intent_name}: {definition}\n"
            if sample_utterances:
                exemplars = "\n".join(sample_utterances)
                exemplars_str += f"{count}) {intent_name}: \n{exemplars}\n"
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
                    definition_str += f"{count}) {intent_name}: {definition}\n"
                if sample_utterances:
                    exemplars = "\n".join(sample_utterances)
                    exemplars_str += f"{count}) {intent_name}: \n{exemplars}\n"
                intents_choice += f"{count}) {intent_name}\n"

                count += 1

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


def format_slot_input(slots: SlotInputList, context: str, type: str = "chat") -> str:
    """Format input for slot filling.

    This function formats the input data for slot filling, creating a
    structured prompt that includes slot requirements and context.
    It supports different types of slot filling operations.

    Args:
        slots: List of slots to fill, each containing:
            - name: Slot name
            - description: Slot description
            - required: Whether the slot is required
            - type: Slot value type
        context: Input context to extract values from
        type: Type of slot filling operation (default: "chat")

    Returns:
        Formatted prompt for slot filling

    Note:
        The function generates different prompts based on the operation type,
        but currently uses the same format for all types.
    """
    slot_descriptions = [_format_slot_description(slot) for slot in slots]
    slot_str = "\n".join(slot_descriptions)
    return _format_slot_prompt(context, slot_str)


def format_verification_input(slot: Dict[str, Any], chat_history_str: str) -> str:
    """Format input for slot verification.

    This function formats the input data for slot verification, creating
    a structured prompt that includes slot information and chat history.
    It guides the model to make a verification decision with reasoning.

    Args:
        slot: Dictionary containing slot information:
            - name: Slot name
            - description: Slot description
            - value: Current slot value
            - type: Slot value type
        chat_history_str: Formatted chat history providing conversation context

    Returns:
        Formatted prompt for slot verification

    Note:
        The function generates a prompt that requests a JSON response
        containing the verification decision and reasoning.
    """
    prompt = f"""Given the following slot and chat history, determine if the slot value needs verification.

Slot:
- Name: {slot["name"]}
- Description: {slot["description"]}
- Value: {slot.get("value", "Not provided")}
- Type: {slot.get("type", "Not specified")}

Chat History:
{chat_history_str}

Please provide your response in JSON format:
{{
    "verification_needed": true/false,
    "thought": "reasoning for the decision"
}}"""

    return prompt
