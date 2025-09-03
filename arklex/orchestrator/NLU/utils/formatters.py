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

from typing import Any


def format_verification_input(slot: dict[str, Any], chat_history_str: str) -> str:
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
            - enum: Fixed choice of value for the slot
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
- Enum: {slot.get("enum", "Not provided")}

Chat History:
{chat_history_str}

Please provide your response in JSON format Only without any markdown formatting or code blocks:
{{
    "verification_needed": true/false,
    "thought": "reasoning for the decision"
}}"""

    return prompt
