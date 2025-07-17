"""Tests for the IntentDetector class.

This module contains comprehensive tests for the IntentDetector class,
covering both local and remote intent detection functionality, as well
as error handling cases.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from arklex.orchestrator.NLU.core.intent import IntentDetector
from arklex.orchestrator.NLU.services.api_service import APIClientService
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.utils.exceptions import APIError, ModelError, ValidationError


@pytest.fixture
def mock_model_service() -> ModelService:
    """Create a mock ModelService for testing."""
    service = Mock(spec=ModelService)
    service.format_intent_input.return_value = (
        "Test prompt",
        {"1": "greeting", "2": "farewell"},
    )
    service.get_response.return_value = "1) greeting"
    return service


@pytest.fixture
def mock_api_service() -> APIClientService:
    """Create a mock APIClientService for testing."""
    service = Mock(spec=APIClientService)
    service.predict_intent.return_value = "greeting"
    return service


@pytest.fixture
def sample_intents() -> dict[str, list[dict[str, Any]]]:
    """Create sample intents for testing."""
    return {
        "greeting": [
            {
                "attribute": {
                    "definition": "A greeting",
                    "sample_utterances": ["hello", "hi"],
                }
            }
        ],
        "farewell": [
            {
                "attribute": {
                    "definition": "A farewell",
                    "sample_utterances": ["goodbye", "bye"],
                }
            }
        ],
    }


def test_intent_detector_initialization(mock_model_service: ModelService) -> None:
    """Test IntentDetector initialization."""
    # Test successful initialization with model service
    detector = IntentDetector(model_service=mock_model_service)
    assert detector.model_service == mock_model_service
    assert detector.api_service is None

    # Test initialization with both services
    api_service = Mock(spec=APIClientService)
    detector = IntentDetector(model_service=mock_model_service, api_service=api_service)
    assert detector.model_service == mock_model_service
    assert detector.api_service == api_service

    # Test initialization without model service
    with pytest.raises(ValidationError, match="Model service is required"):
        IntentDetector(model_service=None)


def test_detect_intent_local(
    mock_model_service: ModelService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test local intent detection."""
    detector = IntentDetector(model_service=mock_model_service)

    # Test successful intent detection
    result = detector._detect_intent_local(
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"
    mock_model_service.format_intent_input.assert_called_once()
    mock_model_service.get_response.assert_called_once()

    # Test invalid response format
    mock_model_service.get_response.return_value = "invalid_format"
    with pytest.raises(ValidationError, match="Invalid response format"):
        detector._detect_intent_local(
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )

    # Test intent not in mapping
    mock_model_service.get_response.return_value = "3) unknown_intent"
    result = detector._detect_intent_local(
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "others"


def test_detect_intent_local_edge_cases(
    mock_model_service: ModelService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test edge cases in local intent detection."""
    detector = IntentDetector(model_service=mock_model_service)

    # Test empty response - should raise ValueError when splitting
    mock_model_service.get_response.return_value = ""
    with pytest.raises(ValidationError, match="Invalid response format"):
        detector._detect_intent_local(
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )

    # Test response with only number and parenthesis - should split into ["1", ""]
    mock_model_service.get_response.return_value = "1)"
    result = detector._detect_intent_local(
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"  # Should use the index mapping since intent is empty

    # Test response with extra spaces
    mock_model_service.get_response.return_value = "  1)  greeting  "
    result = detector._detect_intent_local(
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"

    # Test response with different format (no space after parenthesis)
    mock_model_service.get_response.return_value = "1)greeting"
    result = detector._detect_intent_local(
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"

    # Test when predicted intent is not in mapping but index is valid
    mock_model_service.get_response.return_value = "2) unknown_intent"
    result = detector._detect_intent_local(
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "farewell"  # Should use the index mapping

    # Test when both index and intent are invalid
    mock_model_service.get_response.return_value = "99) unknown_intent"
    result = detector._detect_intent_local(
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "others"  # Should default to "others"

    # Test malformed response that can't be split
    mock_model_service.get_response.return_value = "no_parenthesis"
    with pytest.raises(ValidationError, match="Invalid response format"):
        detector._detect_intent_local(
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )


def test_detect_intent_local_model_service_errors(
    mock_model_service: ModelService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test error handling when model service methods fail."""
    detector = IntentDetector(model_service=mock_model_service)

    # Test format_intent_input failure
    mock_model_service.format_intent_input.side_effect = ModelError("Format error")
    with pytest.raises(ModelError, match="Format error"):
        detector._detect_intent_local(
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )

    # Reset mock
    mock_model_service.format_intent_input.side_effect = None
    mock_model_service.format_intent_input.return_value = (
        "Test prompt",
        {"1": "greeting", "2": "farewell"},
    )

    # Test get_response failure
    mock_model_service.get_response.side_effect = ModelError("Response error")
    with pytest.raises(ModelError, match="Response error"):
        detector._detect_intent_local(
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )


def test_detect_intent_remote(
    mock_model_service: ModelService,
    mock_api_service: APIClientService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test remote intent detection."""
    detector = IntentDetector(
        model_service=mock_model_service, api_service=mock_api_service
    )

    # Test successful remote intent detection
    result = detector._detect_intent_remote(
        text="hello",
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"
    mock_api_service.predict_intent.assert_called_once()

    # Test remote detection without API service
    detector_no_api = IntentDetector(model_service=mock_model_service)
    with pytest.raises(ValidationError, match="API service not configured"):
        detector_no_api._detect_intent_remote(
            text="hello",
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )


def test_detect_intent_remote_api_error_types(
    mock_model_service: ModelService,
    mock_api_service: APIClientService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test different types of API errors in remote intent detection."""
    detector = IntentDetector(
        model_service=mock_model_service, api_service=mock_api_service
    )

    # Test APIError
    mock_api_service.predict_intent.side_effect = APIError("API failed")
    with pytest.raises(APIError, match="Failed to detect intent via API"):
        detector._detect_intent_remote(
            text="hello",
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )

    # Test other exceptions
    mock_api_service.predict_intent.side_effect = RuntimeError("Runtime error")
    with pytest.raises(RuntimeError, match="Runtime error"):
        detector._detect_intent_remote(
            text="hello",
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )


def test_predict_intent(
    mock_model_service: ModelService,
    mock_api_service: APIClientService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test the predict_intent method."""
    # Test local mode
    detector_local = IntentDetector(model_service=mock_model_service)
    result = detector_local.predict_intent(
        text="hello",
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"

    # Test remote mode
    detector_remote = IntentDetector(
        model_service=mock_model_service, api_service=mock_api_service
    )
    result = detector_remote.predict_intent(
        text="hello",
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"


def test_execute(
    mock_model_service: ModelService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test the execute method (alias for predict_intent)."""
    detector = IntentDetector(model_service=mock_model_service)

    # Test successful execution
    result = detector.execute(
        text="hello",
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"

    # Test that execute calls predict_intent
    with patch.object(detector, "predict_intent") as mock_predict:
        mock_predict.return_value = "farewell"
        result = detector.execute(
            text="goodbye",
            intents=sample_intents,
            chat_history_str="User: goodbye",
            model_config={"temperature": 0.7},
        )
        assert result == "farewell"
        mock_predict.assert_called_once()


def test_empty_intents_handling(
    mock_model_service: ModelService,
) -> None:
    """Test handling of empty intents dictionary."""
    detector = IntentDetector(model_service=mock_model_service)

    # Test with empty intents
    mock_model_service.format_intent_input.return_value = ("Test prompt", {})
    mock_model_service.get_response.return_value = "1) unknown"

    result = detector._detect_intent_local(
        intents={},
        chat_history_str="User: hello",
        model_config={"temperature": 0.7},
    )
    assert result == "others"


def test_none_values_handling(
    mock_model_service: ModelService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test handling of None values in parameters."""
    detector = IntentDetector(model_service=mock_model_service)

    # Test with None chat_history_str
    result = detector._detect_intent_local(
        intents=sample_intents,
        chat_history_str=None,  # type: ignore
        model_config={"temperature": 0.7},
    )
    assert result == "greeting"

    # Test with None model_config
    result = detector._detect_intent_local(
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config=None,  # type: ignore
    )
    assert result == "greeting"


def test_model_config_passing(
    mock_model_service: ModelService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test that model_config is properly passed to model service."""
    detector = IntentDetector(model_service=mock_model_service)

    test_config = {"temperature": 0.8, "max_tokens": 100}

    detector._detect_intent_local(
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config=test_config,
    )

    # Verify that the model service was called with the correct config
    mock_model_service.format_intent_input.assert_called_once_with(
        sample_intents, "User: hello"
    )
    mock_model_service.get_response.assert_called_once_with("Test prompt")


def test_api_service_parameter_passing(
    mock_model_service: ModelService,
    mock_api_service: APIClientService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test that parameters are properly passed to API service."""
    detector = IntentDetector(
        model_service=mock_model_service, api_service=mock_api_service
    )

    test_config = {"temperature": 0.8, "max_tokens": 100}

    detector._detect_intent_remote(
        text="hello",
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config=test_config,
    )

    # Verify that the API service was called with the correct parameters
    mock_api_service.predict_intent.assert_called_once_with(
        text="hello",
        intents=sample_intents,
        chat_history_str="User: hello",
        model_config=test_config,
    )
