"""Error handling tests for the IntentDetector class.

This module contains tests specifically for error handling in the IntentDetector class.
These tests are separated to avoid conflicts with the global mock in conftest.py.
"""

from typing import Any
from unittest.mock import Mock, patch

import pytest

from arklex.orchestrator.NLU.core.intent import IntentDetector
from arklex.orchestrator.NLU.services.api_service import APIClientService
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.utils.exceptions import APIError, ArklexError, ModelError, ValidationError


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


@pytest.mark.no_intent_mock
def test_predict_intent_error_handling(
    mock_model_service: ModelService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test error handling in predict_intent method."""
    detector = IntentDetector(model_service=mock_model_service)

    # Test when local detection raises ValidationError
    mock_model_service.format_intent_input.side_effect = ValidationError(
        "Invalid input"
    )
    with pytest.raises(ArklexError, match="Intent prediction failed"):
        detector.predict_intent(
            text="hello",
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

    # Test when local detection raises ModelError
    mock_model_service.get_response.side_effect = ModelError("Model failed")
    with pytest.raises(ArklexError, match="Intent prediction failed"):
        detector.predict_intent(
            text="hello",
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )

    # Test when local detection raises unexpected exception
    mock_model_service.get_response.side_effect = RuntimeError("Unexpected error")
    with pytest.raises(ArklexError, match="Intent prediction failed"):
        detector.predict_intent(
            text="hello",
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )


@pytest.mark.no_intent_mock
def test_predict_intent_remote_error_handling(
    mock_model_service: ModelService,
    mock_api_service: APIClientService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test error handling in predict_intent method for remote mode."""
    detector = IntentDetector(
        model_service=mock_model_service, api_service=mock_api_service
    )

    # Test when remote detection raises APIError
    mock_api_service.predict_intent.side_effect = APIError("API failed")
    with pytest.raises(ArklexError, match="Intent prediction failed"):
        detector.predict_intent(
            text="hello",
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )


@pytest.mark.no_intent_mock
def test_logging_behavior(
    mock_model_service: ModelService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test that logging is called appropriately."""
    detector = IntentDetector(model_service=mock_model_service)

    # Test that predict_intent logs appropriately
    with patch("arklex.orchestrator.NLU.core.intent.log_context") as mock_log:
        detector.predict_intent(
            text="hello",
            intents=sample_intents,
            chat_history_str="User: hello",
            model_config={"temperature": 0.7},
        )

        # Verify logging calls
        assert (
            mock_log.info.call_count >= 3
        )  # At least start, method call, and completion
        assert mock_log.error.call_count == 0  # No errors should occur


@pytest.mark.no_intent_mock
def test_logging_behavior_with_errors(
    mock_model_service: ModelService,
    sample_intents: dict[str, list[dict[str, Any]]],
) -> None:
    """Test logging behavior when errors occur."""
    detector = IntentDetector(model_service=mock_model_service)

    # Make model service fail
    mock_model_service.format_intent_input.side_effect = ValidationError("Test error")

    with patch("arklex.orchestrator.NLU.core.intent.log_context") as mock_log:
        with pytest.raises(ArklexError):
            detector.predict_intent(
                text="hello",
                intents=sample_intents,
                chat_history_str="User: hello",
                model_config={"temperature": 0.7},
            )

        # Verify error logging
        assert mock_log.error.call_count >= 1
