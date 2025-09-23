"""Tests for NLU model_service module."""

import pytest
from unittest.mock import patch, MagicMock
from arklex.orchestrator.NLU.services.model_service import ModelService
from arklex.utils.llm_config import LLMConfig


class TestModelService:
    """Test cases for ModelService class."""

    def test_model_service_initialization(self):
        """Test ModelService initialization."""
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        with patch("arklex.orchestrator.NLU.services.model_service.load_llm") as mock_load_llm:
            mock_model = MagicMock()
            mock_load_llm.return_value = mock_model
            
            service = ModelService(llm_config)
            
            assert service.llm_config == llm_config
            assert service.model == mock_model
            mock_load_llm.assert_called_once_with(llm_config)

    def test_model_service_get_response_basic(self):
        """Test basic get_response functionality."""
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        with patch("arklex.orchestrator.NLU.services.model_service.load_llm") as mock_load_llm:
            mock_model = MagicMock()
            mock_model.invoke.return_value.content = "Test response"
            mock_load_llm.return_value = mock_model
            
            service = ModelService(llm_config)
            result = service.get_response("Test prompt")
            
            assert result == "Test response"
            mock_model.invoke.assert_called_once()

    def test_model_service_get_response_with_system_prompt(self):
        """Test get_response with system prompt."""
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        with patch("arklex.orchestrator.NLU.services.model_service.load_llm") as mock_load_llm:
            mock_model = MagicMock()
            mock_model.invoke.return_value.content = "Test response"
            mock_load_llm.return_value = mock_model
            
            service = ModelService(llm_config)
            result = service.get_response("Test prompt", "System prompt")
            
            assert result == "Test response"
            mock_model.invoke.assert_called_once()

    def test_model_service_get_response_without_system_prompt(self):
        """Test get_response without system prompt."""
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        with patch("arklex.orchestrator.NLU.services.model_service.load_llm") as mock_load_llm:
            mock_model = MagicMock()
            mock_model.invoke.return_value.content = "Test response"
            mock_load_llm.return_value = mock_model
            
            service = ModelService(llm_config)
            result = service.get_response("Test prompt", None)
            
            assert result == "Test response"
            mock_model.invoke.assert_called_once()

    def test_model_service_get_response_model_error(self):
        """Test get_response when model raises error."""
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        with patch("arklex.orchestrator.NLU.services.model_service.load_llm") as mock_load_llm:
            mock_model = MagicMock()
            mock_model.invoke.side_effect = Exception("Model error")
            mock_load_llm.return_value = mock_model
            
            service = ModelService(llm_config)
            with pytest.raises(ValueError, match="Failed to get model response"):
                service.get_response("Test prompt")

    def test_model_service_get_response_empty_response(self):
        """Test get_response with empty response."""
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        with patch("arklex.orchestrator.NLU.services.model_service.load_llm") as mock_load_llm:
            mock_model = MagicMock()
            mock_model.invoke.return_value.content = ""
            mock_load_llm.return_value = mock_model
            
            service = ModelService(llm_config)
            with pytest.raises(ValueError, match="Failed to get model response"):
                service.get_response("Test prompt")

    def test_model_service_get_response_none_response(self):
        """Test get_response with None response."""
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        with patch("arklex.orchestrator.NLU.services.model_service.load_llm") as mock_load_llm:
            mock_model = MagicMock()
            mock_model.invoke.return_value.content = None
            mock_load_llm.return_value = mock_model
            
            service = ModelService(llm_config)
            with pytest.raises(ValueError, match="Failed to get model response"):
                service.get_response("Test prompt")

    def test_model_service_initialization_error(self):
        """Test ModelService initialization error."""
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        with patch("arklex.orchestrator.NLU.services.model_service.load_llm") as mock_load_llm:
            mock_load_llm.side_effect = Exception("Load error")
            
            with pytest.raises(Exception):
                ModelService(llm_config)
