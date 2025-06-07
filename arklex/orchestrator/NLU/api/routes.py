"""FastAPI routes for NLU services.

This module defines the FastAPI routes for NLU services, providing
HTTP endpoints for intent detection and slot filling operations.
It handles request processing, model interaction, and response
formatting for the NLU API.

The module includes:
- Intent detection endpoint (/nlu/predict)
- Slot filling endpoint (/slotfill/predict)
- Slot verification endpoint (/slotfill/verify)
- Error handling and logging
"""

import logging
from typing import Dict, List, Any
from fastapi import FastAPI, Response
from arklex.utils.slot import Slot, Verification
from arklex.orchestrator.NLU.services.model_service import ModelService

logger = logging.getLogger(__name__)
app = FastAPI()
model_service = ModelService()


@app.post("/nlu/predict")
def predict_intent(data: Dict[str, Any], res: Response) -> Dict[str, str]:
    """Predict intent from input text.

    This endpoint processes a request to detect the intent from input text,
    using the provided intents, chat history, and model configuration.
    It handles the entire intent detection pipeline, from input formatting
    to response processing.

    Args:
        data: Request data containing:
            - text: Input text to analyze
            - intents: Dictionary of available intents
            - chat_history_str: Formatted chat history
            - model: Model configuration
        res: FastAPI response object for status code handling

    Returns:
        Dictionary containing:
            - intent: The predicted intent name

    Note:
        In case of errors, returns "others" as the intent and sets
        status code to 500.
    """
    try:
        text = data["text"]
        intents = data["intents"]
        chat_history_str = data["chat_history_str"]
        model_config = data["model"]

        prompt, idx2intents_mapping = model_service.format_intent_input(
            intents, chat_history_str
        )

        response = model_service.get_model_response(
            prompt, model_config, note="intent detection"
        )

        pred_intent = idx2intents_mapping.get(response.strip(), "others")
        return {"intent": pred_intent}
    except Exception as e:
        logger.error(f"Error in intent prediction: {str(e)}")
        res.status_code = 500
        return {"intent": "others"}


@app.post("/slotfill/predict")
def predict_slots(data: Dict[str, Any], res: Response) -> List[Slot]:
    """Fill slots from input context.

    This endpoint processes a request to fill slots from input context,
    using the provided slot definitions and model configuration.
    It handles the entire slot filling pipeline, from input formatting
    to response processing.

    Args:
        data: Request data containing:
            - slots: List of slots to fill
            - context: Input context to extract values from
            - model: Model configuration
            - type: Type of slot filling operation (default: "chat")
        res: FastAPI response object for status code handling

    Returns:
        List of filled slots, each containing extracted values and metadata

    Note:
        In case of errors, returns the original unfilled slots and sets
        status code to 500.
    """
    try:
        slots = [Slot(**slot) for slot in data["slots"]]
        context = data["context"]
        model_config = data["model"]
        type = data.get("type", "chat")

        # Format input for slot filling
        prompt = model_service.format_slot_input(slots, context, type)

        # Get model response
        response = model_service.get_model_response(
            prompt, model_config, response_format="json", note="slot filling"
        )

        # Process response and update slots
        filled_slots = model_service.process_slot_response(response, slots)
        return filled_slots
    except Exception as e:
        logger.error(f"Error in slot filling: {str(e)}")
        res.status_code = 500
        return data["slots"]


@app.post("/slotfill/verify")
def verify_slot(data: Dict[str, Any], res: Response) -> Verification:
    """Verify if slot value needs confirmation.

    This endpoint processes a request to verify a slot value,
    using the provided slot, chat history, and model configuration.
    It handles the entire verification pipeline, from input formatting
    to response processing.

    Args:
        data: Request data containing:
            - slot: The slot to verify
            - chat_history_str: Formatted chat history
            - model: Model configuration
        res: FastAPI response object for status code handling

    Returns:
        Verification result containing:
            - verification_needed: Whether verification is needed
            - thought: Reasoning for the verification decision

    Note:
        In case of errors, returns a default verification result
        (no verification needed) and sets status code to 500.
    """
    try:
        slot = data["slot"]
        chat_history_str = data["chat_history_str"]
        model_config = data["model"]

        # Format input for verification
        prompt = model_service.format_verification_input(slot, chat_history_str)

        # Get model response
        response = model_service.get_model_response(
            prompt, model_config, note="slot verification"
        )

        # Process response
        verification_needed, thought = model_service.process_verification_response(
            response
        )
        return Verification(verification_needed=verification_needed, thought=thought)
    except Exception as e:
        logger.error(f"Error in slot verification: {str(e)}")
        res.status_code = 500
        return Verification(verification_needed=False, thought="No need to verify")
