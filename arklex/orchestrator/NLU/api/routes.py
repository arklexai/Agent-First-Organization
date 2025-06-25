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

from typing import Dict, List, Any
from fastapi import FastAPI, Response, APIRouter, Depends
from arklex.utils.slot import Slot, Verification
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.utils.logging_utils import LogContext, handle_exceptions, LOG_MESSAGES
from arklex.utils.exceptions import ModelError, ValidationError
from arklex.orchestrator.NLU.core.base import (
    IntentResponse,
    SlotResponse,
    VerificationResponse,
)

log_context = LogContext(__name__)
app = FastAPI()
# Placeholder/default model configuration for ModelService
DEFAULT_MODEL_CONFIG = {
    "model_name": "default-nlu-model",
    "api_key": "dummy-key",
    "endpoint": "https://dummy-endpoint.com/api",
    "model_type_or_path": "gpt-3.5-turbo",
    "llm_provider": "openai",
    "temperature": 0.1,
    "max_tokens": 1000,
    "response_format": "json",
}

router = APIRouter()


@handle_exceptions()
async def get_model_service() -> ModelService:
    """Get or create a ModelService instance.

    Returns:
        ModelService: Initialized model service instance

    Raises:
        ModelError: If model service initialization fails
    """
    try:
        return ModelService(model_config=DEFAULT_MODEL_CONFIG)
    except Exception as e:
        log_context.error(
            LOG_MESSAGES["MODEL_INIT_ERROR"],
            extra={
                "error": str(e),
                "operation": "model_service_initialization",
            },
        )
        raise ModelError(
            "Failed to initialize model service",
            details={"error": str(e), "operation": "initialization"},
        )


@router.post("/predict_intent", response_model=IntentResponse)
@handle_exceptions()
async def predict_intent(
    text: str, model_service: ModelService = Depends(get_model_service)
) -> IntentResponse:
    """Predict intent from input text.

    Args:
        text: Input text to analyze
        model_service: Model service instance

    Returns:
        IntentResponse: Predicted intent and confidence

    Raises:
        ValidationError: If input validation fails
        ModelError: If model prediction fails
    """
    log_context.info(
        "Processing intent prediction request",
        extra={"text": text, "operation": "intent_prediction"},
    )
    response = await model_service.predict_intent(text)
    log_context.info(
        "Intent prediction successful",
        extra={"response": response, "operation": "intent_prediction"},
    )
    return response


@router.post("/fill_slots", response_model=SlotResponse)
@handle_exceptions()
async def fill_slots(
    text: str, intent: str, model_service: ModelService = Depends(get_model_service)
) -> SlotResponse:
    """Fill slots based on input text and intent.

    Args:
        text: Input text to analyze
        intent: Detected intent
        model_service: Model service instance

    Returns:
        SlotResponse: Filled slots and metadata

    Raises:
        ValidationError: If input validation fails
        ModelError: If slot filling fails
    """
    log_context.info(
        "Processing slot filling request",
        extra={"text": text, "intent": intent, "operation": "slot_filling"},
    )
    response = await model_service.fill_slots(text, intent)
    log_context.info(
        "Slot filling successful",
        extra={"response": response, "operation": "slot_filling"},
    )
    return response


@router.post("/verify_slots", response_model=VerificationResponse)
@handle_exceptions()
async def verify_slots(
    text: str,
    slots: Dict[str, Any],
    model_service: ModelService = Depends(get_model_service),
) -> VerificationResponse:
    """Verify slots against input text.

    Args:
        text: Input text to verify against
        slots: Slots to verify
        model_service: Model service instance

    Returns:
        VerificationResponse: Verification results

    Raises:
        ValidationError: If input validation fails
        ModelError: If slot verification fails
    """
    log_context.info(
        "Processing slot verification request",
        extra={"text": text, "slots": slots, "operation": "slot_verification"},
    )
    response = await model_service.verify_slots(text, slots)
    log_context.info(
        "Slot verification successful",
        extra={"response": response, "operation": "slot_verification"},
    )
    return response


@app.post("/nlu/predict")
@handle_exceptions()
def predict_intent(
    data: Dict[str, Any],
    res: Response,
    model_service: ModelService = Depends(get_model_service),
) -> Dict[str, str]:
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
        model_service: Model service instance

    Returns:
        Dictionary containing:
            - intent: The predicted intent name

    Raises:
        ValidationError: If input validation fails
        ModelError: If model prediction fails
        APIError: If API interaction fails
    """
    try:
        text = data["text"]
        intents = data["intents"]
        chat_history_str = data["chat_history_str"]
        model_config = data["model"]
        log_context.info(
            "Processing intent prediction request",
            extra={
                "text": text,
                "intents": intents,
                "operation": "intent_prediction",
            },
        )
        prompt, idx2intents_mapping = model_service.format_intent_input(
            intents, chat_history_str
        )
        response = model_service.get_model_response(
            prompt, model_config, note="intent detection"
        )
        pred_intent = idx2intents_mapping.get(response.strip(), "others")
        log_context.info(
            "Intent prediction successful",
            extra={
                "predicted_intent": pred_intent,
                "operation": "intent_prediction",
            },
        )
        return {"intent": pred_intent}
    except KeyError as e:
        log_context.error(
            "Missing required field in request",
            extra={
                "error": str(e),
                "operation": "intent_prediction",
                "required_field": str(e),
            },
        )
        raise ValidationError(
            "Missing required field in request",
            details={"field": str(e), "operation": "intent_prediction"},
        )
    except Exception as e:
        log_context.error(
            "Error in intent prediction",
            extra={
                "error": str(e),
                "operation": "intent_prediction",
            },
        )
        raise ModelError(
            "Failed to predict intent",
            details={"error": str(e), "operation": "intent_prediction"},
        )


@app.post("/slotfill/predict")
@handle_exceptions()
def predict_slots(
    data: Dict[str, Any],
    res: Response,
    model_service: ModelService = Depends(get_model_service),
) -> List[Slot]:
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
        model_service: Model service instance

    Returns:
        List of filled slots, each containing extracted values and metadata

    Raises:
        ValidationError: If input validation fails
        ModelError: If slot filling fails
        APIError: If API interaction fails
    """
    try:
        slots = [Slot(**slot) for slot in data["slots"]]
        context = data["context"]
        model_config = data["model"]
        type = data.get("type", "chat")
        log_context.info(
            "Processing slot filling request",
            extra={
                "context": context,
                "slots": slots,
                "type": type,
                "operation": "slot_filling",
            },
        )
        prompt = model_service.format_slot_input(slots, context, type)
        response = model_service.get_model_response(
            prompt, model_config, response_format="json", note="slot filling"
        )
        filled_slots = model_service.process_slot_response(response, slots)
        log_context.info(
            "Slot filling successful",
            extra={
                "filled_slots": filled_slots,
                "operation": "slot_filling",
            },
        )
        return filled_slots
    except KeyError as e:
        log_context.error(
            "Missing required field in request",
            extra={
                "error": str(e),
                "operation": "slot_filling",
                "required_field": str(e),
            },
        )
        raise ValidationError(
            "Missing required field in request",
            details={"field": str(e), "operation": "slot_filling"},
        )
    except Exception as e:
        log_context.error(
            "Error in slot filling",
            extra={
                "error": str(e),
                "operation": "slot_filling",
            },
        )
        raise ModelError(
            "Failed to fill slots",
            details={"error": str(e), "operation": "slot_filling"},
        )


@app.post("/slotfill/verify")
@handle_exceptions()
def verify_slot(
    data: Dict[str, Any],
    res: Response,
    model_service: ModelService = Depends(get_model_service),
) -> Verification:
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
        model_service: Model service instance

    Returns:
        Verification result containing:
            - verification_needed: Whether verification is needed
            - thought: Reasoning for the verification decision

    Raises:
        ValidationError: If input validation fails
        ModelError: If slot verification fails
        APIError: If API interaction fails
    """
    try:
        slot = data["slot"]
        chat_history_str = data["chat_history_str"]
        model_config = data["model"]
        log_context.info(
            "Processing slot verification request",
            extra={
                "slot": slot,
                "operation": "slot_verification",
            },
        )
        prompt = model_service.format_verification_input(slot, chat_history_str)
        response = model_service.get_model_response(
            prompt, model_config, note="slot verification"
        )
        verification = model_service.process_verification_response(response)
        log_context.info(
            "Slot verification successful",
            extra={
                "verification": verification,
                "operation": "slot_verification",
            },
        )
        return verification
    except KeyError as e:
        log_context.error(
            "Missing required field in request",
            extra={
                "error": str(e),
                "operation": "slot_verification",
                "required_field": str(e),
            },
        )
        raise ValidationError(
            "Missing required field in request",
            details={"field": str(e), "operation": "slot_verification"},
        )
    except Exception as e:
        log_context.error(
            "Error in slot verification",
            extra={
                "error": str(e),
                "operation": "slot_verification",
            },
        )
        raise ModelError(
            "Failed to verify slot",
            details={"error": str(e), "operation": "slot_verification"},
        )
