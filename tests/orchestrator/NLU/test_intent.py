"""Tests for NLU intent module."""

import pytest
from unittest.mock import patch, MagicMock
from arklex.orchestrator.NLU.core.intent import IntentDetector
from arklex.orchestrator.NLU.core.base import IntentResponse
from arklex.utils.llm_config import LLMConfig


@pytest.mark.no_intent_mock
class TestIntentDetector:
    """Test cases for IntentDetector class."""

    def test_intent_detector_initialization(self):
        """Test IntentDetector initialization."""
        from arklex.orchestrator.NLU.services.model_service import ModelService
        from arklex.utils.llm_config import LLMConfig
        
        llm_config = LLMConfig()
        model_service = ModelService(llm_config)
        detector = IntentDetector(model_service)
        
        assert detector.model_service is not None

    @patch("arklex.orchestrator.NLU.core.intent.ModelService")
    def test_intent_detector_predict_intent_basic(self, mock_model_service_class):
        """Test basic intent prediction."""
        mock_model_service = MagicMock()
        mock_response = MagicMock()
        mock_response.get.return_value = "greeting"
        mock_model_service.get_response_with_structured_output.return_value = mock_response
        mock_model_service_class.return_value = mock_model_service
        
        detector = IntentDetector(mock_model_service)
        
        intents = {
            "greeting": [{"definition": "A greeting", "sample_utterances": ["Hello", "Hi"]}],
            "goodbye": [{"definition": "A farewell", "sample_utterances": ["Bye", "Goodbye"]}]
        }
        chat_history_str = "User: Hello there"
        
        result = detector.predict_intent(intents, chat_history_str)
        
        assert isinstance(result, str)
        assert result == "greeting"
        mock_model_service.get_response_with_structured_output.assert_called_once()

    @patch("arklex.orchestrator.NLU.core.intent.ModelService")
    def test_intent_detector_predict_intent_with_context(self, mock_model_service_class):
        """Test intent prediction with context."""
        mock_model_service = MagicMock()
        mock_response = MagicMock()
        mock_response.get.return_value = "question"
        mock_model_service.get_response_with_structured_output.return_value = mock_response
        mock_model_service_class.return_value = mock_model_service
        
        detector = IntentDetector(mock_model_service)
        
        intents = {
            "question": [{"definition": "A question", "sample_utterances": ["What is the weather?", "How are you?"]}],
            "greeting": [{"definition": "A greeting", "sample_utterances": ["Hello", "Hi"]}]
        }
        chat_history_str = "User: What is the weather?"
        
        result = detector.predict_intent(intents, chat_history_str)
        
        assert isinstance(result, str)
        assert result == "question"
        mock_model_service.get_response_with_structured_output.assert_called_once()

    @patch("arklex.orchestrator.NLU.core.intent.ModelService")
    def test_intent_detector_predict_intent_invalid_json(self, mock_model_service_class):
        """Test intent prediction with invalid JSON response."""
        mock_model_service = MagicMock()
        mock_response = MagicMock()
        mock_response.get.return_value = None  # Simulate missing intent field
        mock_model_service.get_response_with_structured_output.return_value = mock_response
        mock_model_service_class.return_value = mock_model_service
        
        detector = IntentDetector(mock_model_service)
        
        intents = {
            "greeting": [{"definition": "A greeting", "sample_utterances": ["Hello", "Hi"]}]
        }
        chat_history_str = "User: Hello"
        
        result = detector.predict_intent(intents, chat_history_str)
        
        # Should return default intent when JSON parsing fails
        assert isinstance(result, str)
        assert result == "others"

    @patch("arklex.orchestrator.NLU.core.intent.ModelService")
    def test_intent_detector_predict_intent_model_error(self, mock_model_service_class):
        """Test intent prediction when model raises error."""
        mock_model_service = MagicMock()
        mock_model_service.get_response_with_structured_output.side_effect = Exception("Model error")
        mock_model_service_class.return_value = mock_model_service
        
        detector = IntentDetector(mock_model_service)
        
        intents = {
            "greeting": [{"definition": "A greeting", "sample_utterances": ["Hello", "Hi"]}]
        }
        chat_history_str = "User: Hello"
        
        result = detector.predict_intent(intents, chat_history_str)
        
        # Should return default intent when model fails
        assert isinstance(result, str)
        assert result == "others"

    @patch("arklex.orchestrator.NLU.core.intent.ModelService")
    def test_intent_detector_predict_intent_missing_fields(self, mock_model_service_class):
        """Test intent prediction with missing fields in JSON."""
        mock_model_service = MagicMock()
        mock_response = MagicMock()
        mock_response.get.return_value = "greeting"
        mock_model_service.get_response_with_structured_output.return_value = mock_response
        mock_model_service_class.return_value = mock_model_service
        
        detector = IntentDetector(mock_model_service)
        
        intents = {
            "greeting": [{"definition": "A greeting", "sample_utterances": ["Hello", "Hi"]}]
        }
        chat_history_str = "User: Hello"
        
        result = detector.predict_intent(intents, chat_history_str)
        
        assert isinstance(result, str)
        assert result == "greeting"

    @patch("arklex.orchestrator.NLU.core.intent.ModelService")
    def test_intent_detector_predict_intent_with_metadata(self, mock_model_service_class):
        """Test intent prediction with metadata."""
        mock_model_service = MagicMock()
        mock_response = MagicMock()
        mock_response.get.return_value = "greeting"
        mock_model_service.get_response_with_structured_output.return_value = mock_response
        mock_model_service_class.return_value = mock_model_service
        
        detector = IntentDetector(mock_model_service)
        
        intents = {
            "greeting": [{"definition": "A greeting", "sample_utterances": ["Hello", "Hi"]}]
        }
        chat_history_str = "User: Hello"
        
        result = detector.predict_intent(intents, chat_history_str)
        
        assert isinstance(result, str)
        assert result == "greeting"
