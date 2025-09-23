"""Tests for env/tools/utils module."""

import pytest
from unittest.mock import patch, MagicMock
from arklex.env.tools.utils import (
    get_prompt_template,
    ToolGenerator,
)
from arklex.orchestrator.entities.orchestrator_state_entities import (
    BotConfig,
    ConvoMessage,
    OrchestratorState,
    StreamType,
)
from arklex.types.stream_types import EventType


class TestGetPromptTemplate:
    """Test cases for get_prompt_template function."""

    @patch("arklex.env.tools.utils.load_prompts")
    def test_get_prompt_template_speech_non_chinese(self, mock_load_prompts):
        """Test getting prompt template for speech stream type (non-Chinese)."""
        mock_prompts = {
            "test_prompt": "Regular prompt",
            "test_prompt_speech": "Speech prompt"
        }
        mock_load_prompts.return_value = mock_prompts
        
        bot_config = BotConfig(
            language="EN",
            llm_config={},
            taskgraph={}
        )
        state = OrchestratorState(
            stream_type=StreamType.SPEECH,
            bot_config=bot_config,
            user_message=ConvoMessage(history="", message="test"),
            event_type=EventType.TEXT
        )
        
        result = get_prompt_template(state, "test_prompt")
        
        assert result is not None
        mock_load_prompts.assert_called_once_with(bot_config)

    @patch("arklex.env.tools.utils.load_prompts")
    def test_get_prompt_template_speech_chinese(self, mock_load_prompts):
        """Test getting prompt template for speech stream type (Chinese)."""
        mock_prompts = {
            "test_prompt": "Regular prompt",
            "test_prompt_speech": "Speech prompt"
        }
        mock_load_prompts.return_value = mock_prompts
        
        bot_config = BotConfig(
            language="CN",
            llm_config={},
            taskgraph={}
        )
        state = OrchestratorState(
            stream_type=StreamType.SPEECH,
            bot_config=bot_config,
            user_message=ConvoMessage(history="", message="test"),
            event_type=EventType.TEXT
        )
        
        result = get_prompt_template(state, "test_prompt")
        
        assert result is not None
        mock_load_prompts.assert_called_once_with(bot_config)

    @patch("arklex.env.tools.utils.load_prompts")
    def test_get_prompt_template_non_speech(self, mock_load_prompts):
        """Test getting prompt template for non-speech stream type."""
        mock_prompts = {
            "test_prompt": "Regular prompt",
            "test_prompt_speech": "Speech prompt"
        }
        mock_load_prompts.return_value = mock_prompts
        
        bot_config = BotConfig(
            language="EN",
            llm_config={},
            taskgraph={}
        )
        state = OrchestratorState(
            stream_type=StreamType.TEXT,
            bot_config=bot_config,
            user_message=ConvoMessage(history="", message="test"),
            event_type=EventType.TEXT
        )
        
        result = get_prompt_template(state, "test_prompt")
        
        assert result is not None
        mock_load_prompts.assert_called_once_with(bot_config)


class TestToolGenerator:
    """Test cases for ToolGenerator class."""

    @patch("arklex.env.tools.utils.ModelService")
    def test_tool_generator_generate_basic(self, mock_model_service_class):
        """Test basic ToolGenerator.generate functionality."""
        mock_model_service = MagicMock()
        mock_model_service.get_response.return_value = "Generated response"
        mock_model_service_class.return_value = mock_model_service
        
        bot_config = BotConfig(
            language="EN",
            llm_config={"provider": "openai", "model": "gpt-3.5-turbo"},
            taskgraph={},
        )
        state = OrchestratorState(
            stream_type=StreamType.TEXT,
            bot_config=bot_config,
            user_message=ConvoMessage(history="", message="test message"),
            event_type=EventType.TEXT
        )
        
        result = ToolGenerator.generate(state)
        
        assert result == "Generated response"
        mock_model_service_class.assert_called_once()
        mock_model_service.get_response.assert_called_once()

    @patch("arklex.env.tools.utils.ModelService")
    def test_tool_generator_generate_with_different_config(self, mock_model_service_class):
        """Test ToolGenerator.generate with different config."""
        mock_model_service = MagicMock()
        mock_model_service.get_response.return_value = "Generated response"
        mock_model_service_class.return_value = mock_model_service
        
        bot_config = BotConfig(
            language="CN",
            llm_config={"provider": "anthropic", "model": "claude-3"},
            taskgraph={},
        )
        state = OrchestratorState(
            stream_type=StreamType.SPEECH,
            bot_config=bot_config,
            user_message=ConvoMessage(history="", message="test message"),
            event_type=EventType.TEXT
        )
        
        result = ToolGenerator.generate(state)
        
        assert result == "Generated response"
        mock_model_service_class.assert_called_once()
        mock_model_service.get_response.assert_called_once()

    @patch("arklex.env.tools.utils.ModelService")
    def test_tool_generator_generate_model_error(self, mock_model_service_class):
        """Test ToolGenerator.generate when model raises error."""
        mock_model_service = MagicMock()
        mock_model_service.get_response.side_effect = Exception("Model error")
        mock_model_service_class.return_value = mock_model_service
        
        bot_config = BotConfig(
            language="EN",
            llm_config={"provider": "openai", "model": "gpt-3.5-turbo"},
            taskgraph={},
        )
        state = OrchestratorState(
            stream_type=StreamType.TEXT,
            bot_config=bot_config,
            user_message=ConvoMessage(history="", message="test message"),
            event_type=EventType.TEXT
        )
        
        with pytest.raises(Exception, match="Model error"):
            ToolGenerator.generate(state)

    @patch("arklex.env.tools.utils.ModelService")
    def test_tool_generator_generate_empty_response(self, mock_model_service_class):
        """Test ToolGenerator.generate with empty response."""
        mock_model_service = MagicMock()
        mock_model_service.get_response.return_value = ""
        mock_model_service_class.return_value = mock_model_service
        
        bot_config = BotConfig(
            language="EN",
            llm_config={"provider": "openai", "model": "gpt-3.5-turbo"},
            taskgraph={},
        )
        state = OrchestratorState(
            stream_type=StreamType.TEXT,
            bot_config=bot_config,
            user_message=ConvoMessage(history="", message="test message"),
            event_type=EventType.TEXT
        )
        
        result = ToolGenerator.generate(state)
        
        assert result == ""

    @patch("arklex.env.tools.utils.ModelService")
    def test_tool_generator_generate_none_response(self, mock_model_service_class):
        """Test ToolGenerator.generate with None response."""
        mock_model_service = MagicMock()
        mock_model_service.get_response.return_value = None
        mock_model_service_class.return_value = mock_model_service
        
        bot_config = BotConfig(
            language="EN",
            llm_config={"provider": "openai", "model": "gpt-3.5-turbo"},
            taskgraph={},
        )
        state = OrchestratorState(
            stream_type=StreamType.TEXT,
            bot_config=bot_config,
            user_message=ConvoMessage(history="", message="test message"),
            event_type=EventType.TEXT
        )
        
        result = ToolGenerator.generate(state)
        
        assert result is None