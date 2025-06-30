from unittest.mock import MagicMock, patch

import pytest

from arklex.orchestrator.NLU.core.intent import IntentDetector
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.utils.exceptions import ModelError, ValidationError


@pytest.fixture
def dummy_model_service() -> ModelService:
    return ModelService(
        {
            "model_name": "dummy",
            "api_key": "dummy",
            "endpoint": "http://dummy",
            "model_type_or_path": "dummy-path",
            "llm_provider": "dummy",
        }
    )


def test_intent_detector_local_detection(dummy_model_service: ModelService) -> None:
    detector = IntentDetector(model_service=dummy_model_service)
    intents = {
        "greeting": [
            {
                "attribute": {
                    "definition": "A greeting",
                    "sample_utterances": ["hello", "hi"],
                }
            }
        ]
    }
    chat_history_str = "User: hello"
    model_config = {"temperature": 0.7}

    # Mock the model service's get_response method
    with patch.object(dummy_model_service, "get_response", return_value="0) greeting"):
        result = detector._detect_intent_local(intents, chat_history_str, model_config)
        assert result == "greeting"


def test_intent_detector_invalid_response_format(
    dummy_model_service: ModelService,
) -> None:
    detector = IntentDetector(model_service=dummy_model_service)
    intents = {
        "greeting": [
            {
                "attribute": {
                    "definition": "A greeting",
                    "sample_utterances": ["hello", "hi"],
                }
            }
        ]
    }
    chat_history_str = "User: hello"
    model_config = {"temperature": 0.7}

    # Mock the model service's get_response method to return invalid format
    with (
        patch.object(
            dummy_model_service, "get_response", return_value="invalid_response"
        ),
        pytest.raises(ValidationError, match="Invalid response format"),
    ):
        detector._detect_intent_local(intents, chat_history_str, model_config)


def test_intent_detector_model_error(dummy_model_service: ModelService) -> None:
    detector = IntentDetector(model_service=dummy_model_service)
    intents = {
        "greeting": [
            {
                "attribute": {
                    "definition": "A greeting",
                    "sample_utterances": ["hello", "hi"],
                }
            }
        ]
    }
    chat_history_str = "User: hello"
    model_config = {"temperature": 0.7}

    # Mock the model service's get_response method to raise ModelError
    with (
        patch.object(
            dummy_model_service, "get_response", side_effect=ModelError("Model error")
        ),
        pytest.raises(ModelError, match="Model error"),
    ):
        detector._detect_intent_local(intents, chat_history_str, model_config)


def test_intent_detector_remote_detection(dummy_model_service: ModelService) -> None:
    detector = IntentDetector(
        model_service=dummy_model_service, api_service=MagicMock()
    )
    intents = {
        "greeting": [
            {
                "attribute": {
                    "definition": "A greeting",
                    "sample_utterances": ["hello", "hi"],
                }
            }
        ]
    }
    chat_history_str = "User: hello"
    model_config = {"temperature": 0.7}
    text = "hello"
    # Mock the api_service's predict_intent method for remote detection
    detector.api_service.predict_intent.return_value = "greeting"
    result = detector._detect_intent_remote(
        text, intents, chat_history_str, model_config
    )
    assert result == "greeting"


def test_intent_detector_predict_intent(dummy_model_service: ModelService) -> None:
    detector = IntentDetector(model_service=dummy_model_service)
    intents = {
        "greeting": [
            {
                "attribute": {
                    "definition": "A greeting",
                    "sample_utterances": ["hello", "hi"],
                }
            }
        ]
    }
    chat_history_str = "User: hello"
    model_config = {"temperature": 0.7}
    text = "hello"
    # Mock the model service's get_response method
    with patch.object(dummy_model_service, "get_response", return_value="0) greeting"):
        result = detector.predict_intent(text, intents, chat_history_str, model_config)
        assert result == "greeting"
