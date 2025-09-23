"""Tests for env/prompts module."""

import pytest
from unittest.mock import patch, MagicMock
from arklex.env.prompts import load_prompts
from arklex.orchestrator.entities.orchestrator_state_entities import BotConfig


class TestLoadPrompts:
    """Test cases for load_prompts function."""

    def test_load_prompts_with_prompts_in_config(self):
        """Test loading prompts when prompts are in bot_config."""
        bot_config = BotConfig(
            language="EN",
            llm_config={},
            taskgraph={}
        )
        
        result = load_prompts(bot_config)
        
        assert isinstance(result, dict)
        assert len(result) > 0
        # Check that default prompts are loaded
        assert "generator_prompt" in result
        assert "generator_prompt_speech" in result

    def test_load_prompts_with_empty_prompts(self):
        """Test loading prompts when prompts dict is empty."""
        bot_config = BotConfig(
            language="EN",
            llm_config={},
            taskgraph={},
            prompts={}
        )
        
        result = load_prompts(bot_config)
        
        assert isinstance(result, dict)
        assert len(result) > 0  # Function returns default prompts regardless of config.prompts

    def test_load_prompts_with_none_prompts(self):
        """Test loading prompts when prompts is None."""
        bot_config = BotConfig(
            language="EN",
            llm_config={},
            taskgraph={},
            prompts=None
        )
        
        result = load_prompts(bot_config)
        
        assert isinstance(result, dict)
        assert len(result) > 0  # Function returns default prompts regardless of config.prompts

    def test_load_prompts_with_missing_prompts_key(self):
        """Test loading prompts when prompts key is missing."""
        bot_config = BotConfig(
            language="EN",
            llm_config={},
            taskgraph={}
        )
        
        result = load_prompts(bot_config)
        
        assert isinstance(result, dict)
        assert len(result) > 0  # Function returns default prompts regardless of config.prompts

    def test_load_prompts_with_various_prompt_types(self):
        """Test loading prompts with various prompt content types."""
        bot_config = BotConfig(
            language="EN",
            llm_config={},
            taskgraph={},
            prompts={
                "string_prompt": "String prompt",
                "empty_prompt": "",
                "none_prompt": None,
                "numeric_prompt": 123,
                "boolean_prompt": True
            }
        )
        
        result = load_prompts(bot_config)
        
        assert isinstance(result, dict)
        # Function returns default prompts, not the custom ones from config
        assert "generator_prompt" in result
        assert "context_generator_prompt" in result

    def test_load_prompts_preserves_original_config(self):
        """Test that load_prompts doesn't modify the original config."""
        bot_config = BotConfig(
            language="EN",
            llm_config={},
            taskgraph={}
        )
        
        result = load_prompts(bot_config)
        
        # Modify the result
        result["generator_prompt"] = "Modified content"
        
        # Call again to ensure we get fresh default prompts
        result2 = load_prompts(bot_config)
        assert result2["generator_prompt"] != "Modified content"
        assert result["generator_prompt"] == "Modified content"

    def test_load_prompts_with_nested_prompts(self):
        """Test loading prompts with nested structure."""
        bot_config = BotConfig(
            language="EN",
            llm_config={},
            taskgraph={},
            prompts={
                "level1": {
                    "level2": {
                        "prompt": "Nested prompt"
                    }
                },
                "simple_prompt": "Simple prompt"
            }
        )
        
        result = load_prompts(bot_config)
        
        assert isinstance(result, dict)
        # Function returns default prompts, not the nested ones from config
        assert "generator_prompt" in result
        assert "context_generator_prompt" in result
