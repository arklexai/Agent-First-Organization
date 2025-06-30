"""Comprehensive tests for the prompts module.

This module provides comprehensive test coverage for the prompts module,
ensuring all functionality is properly tested including all language variants and edge cases.
"""

import pytest

from arklex.env.prompts import BotConfig, load_prompts


class TestBotConfig:
    """Test the BotConfig dataclass."""

    def test_bot_config_initialization(self) -> None:
        """Test BotConfig initialization."""
        config = BotConfig(language="EN")
        assert config.language == "EN"

    def test_bot_config_different_language(self) -> None:
        """Test BotConfig with different language."""
        config = BotConfig(language="CN")
        assert config.language == "CN"

    def test_bot_config_equality(self) -> None:
        """Test BotConfig equality."""
        config1 = BotConfig(language="EN")
        config2 = BotConfig(language="EN")
        assert config1 == config2

    def test_bot_config_inequality(self) -> None:
        """Test BotConfig inequality."""
        config1 = BotConfig(language="EN")
        config2 = BotConfig(language="CN")
        assert config1 != config2

    def test_bot_config_repr(self) -> None:
        """Test BotConfig string representation."""
        config = BotConfig(language="EN")
        assert "BotConfig" in repr(config)
        assert "EN" in repr(config)


class TestLoadPromptsEnglish:
    """Test loading English prompts."""

    def test_load_prompts_english(self) -> None:
        """Test loading English prompts."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        # Check that all expected prompt keys are present
        expected_keys = [
            "generator_prompt",
            "generator_prompt_speech",
            "context_generator_prompt",
            "context_generator_prompt_speech",
            "message_generator_prompt",
            "message_generator_prompt_speech",
            "message_flow_generator_prompt",
            "message_flow_generator_prompt_speech",
            "retrieve_contextualize_q_prompt",
            "retrieval_needed_prompt",
            "choose_worker_prompt",
            "database_action_prompt",
            "database_slot_prompt",
            "regenerate_response",
        ]

        for key in expected_keys:
            assert key in prompts, f"Missing prompt key: {key}"
            assert isinstance(prompts[key], str), f"Prompt {key} is not a string"
            assert len(prompts[key]) > 0, f"Prompt {key} is empty"

    def test_english_generator_prompt_content(self) -> None:
        """Test English generator prompt content."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        prompt = prompts["generator_prompt"]
        assert "{sys_instruct}" in prompt
        assert "{formatted_chat}" in prompt
        assert "assistant:" in prompt
        assert "Never repeat verbatim" in prompt

    def test_english_speech_prompt_content(self) -> None:
        """Test English speech prompt content."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        prompt = prompts["generator_prompt_speech"]
        assert "{sys_instruct}" in prompt
        assert "{formatted_chat}" in prompt
        assert "assistant (for speech):" in prompt
        assert "voice assistant" in prompt
        assert "spoken" in prompt

    def test_english_context_prompt_content(self) -> None:
        """Test English context prompt content."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        prompt = prompts["context_generator_prompt"]
        assert "{sys_instruct}" in prompt
        assert "{formatted_chat}" in prompt
        assert "{context}" in prompt
        assert "assistant:" in prompt

    def test_english_message_prompt_content(self) -> None:
        """Test English message prompt content."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        prompt = prompts["message_generator_prompt"]
        assert "{sys_instruct}" in prompt
        assert "{formatted_chat}" in prompt
        assert "{message}" in prompt
        assert "embed the following message" in prompt

    def test_english_message_flow_prompt_content(self) -> None:
        """Test English message flow prompt content."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        prompt = prompts["message_flow_generator_prompt"]
        assert "{sys_instruct}" in prompt
        assert "{formatted_chat}" in prompt
        assert "{context}" in prompt
        assert "{message}" in prompt

    def test_english_rag_prompt_content(self) -> None:
        """Test English RAG prompt content."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        prompt = prompts["retrieve_contextualize_q_prompt"]
        assert "{chat_history}" in prompt
        assert "formulate a standalone question" in prompt

    def test_english_retrieval_needed_prompt_content(self) -> None:
        """Test English retrieval needed prompt content."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        prompt = prompts["retrieval_needed_prompt"]
        assert "{formatted_chat}" in prompt
        assert "yes or no" in prompt

    def test_english_worker_prompt_content(self) -> None:
        """Test English worker prompt content."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        prompt = prompts["choose_worker_prompt"]
        assert "{workers_info}" in prompt
        assert "{task}" in prompt
        assert "{formatted_chat}" in prompt
        assert "{workers_name}" in prompt

    def test_english_database_prompts_content(self) -> None:
        """Test English database prompts content."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        # Test database action prompt
        action_prompt = prompts["database_action_prompt"]
        assert "{actions_info}" in action_prompt
        assert "{user_intent}" in action_prompt
        assert "{actions_name}" in action_prompt

        # Test database slot prompt
        slot_prompt = prompts["database_slot_prompt"]
        assert "{slot}" in slot_prompt
        assert "{value}" in slot_prompt
        assert "{value_list}" in slot_prompt

    def test_english_regenerate_prompt_content(self) -> None:
        """Test English regenerate prompt content."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        prompt = prompts["regenerate_response"]
        assert "{original_answer}" in prompt
        assert "Rephrase the Original Answer" in prompt
        assert "fluency or coherence issues" in prompt


class TestLoadPromptsChinese:
    """Test loading Chinese prompts."""

    def test_load_prompts_chinese(self) -> None:
        """Test loading Chinese prompts."""
        config = BotConfig(language="CN")
        prompts = load_prompts(config)

        # Check that all expected prompt keys are present for Chinese
        expected_keys = [
            "generator_prompt",
            "context_generator_prompt",
            "message_generator_prompt",
            "message_flow_generator_prompt",
            "retrieve_contextualize_q_prompt",
            "retrieval_needed_prompt",
            "choose_worker_prompt",
            "database_action_prompt",
            "database_slot_prompt",
        ]

        for key in expected_keys:
            assert key in prompts, f"Missing prompt key: {key}"
            assert isinstance(prompts[key], str), f"Prompt {key} is not a string"
            assert len(prompts[key]) > 0, f"Prompt {key} is empty"

    def test_chinese_generator_prompt_content(self) -> None:
        """Test Chinese generator prompt content."""
        config = BotConfig(language="CN")
        prompts = load_prompts(config)

        prompt = prompts["generator_prompt"]
        assert "{sys_instruct}" in prompt
        assert "{formatted_chat}" in prompt
        assert "助手：" in prompt
        assert "不要凭空想象" in prompt

    def test_chinese_context_prompt_content(self) -> None:
        """Test Chinese context prompt content."""
        config = BotConfig(language="CN")
        prompts = load_prompts(config)

        prompt = prompts["context_generator_prompt"]
        assert "{sys_instruct}" in prompt
        assert "{formatted_chat}" in prompt
        assert "{context}" in prompt
        assert "上下文：" in prompt

    def test_chinese_message_prompt_content(self) -> None:
        """Test Chinese message prompt content."""
        config = BotConfig(language="CN")
        prompts = load_prompts(config)

        prompt = prompts["message_generator_prompt"]
        assert "{sys_instruct}" in prompt
        assert "{formatted_chat}" in prompt
        assert "{message}" in prompt
        assert "加入以下消息" in prompt

    def test_chinese_message_flow_prompt_content(self) -> None:
        """Test Chinese message flow prompt content."""
        config = BotConfig(language="CN")
        prompts = load_prompts(config)

        prompt = prompts["message_flow_generator_prompt"]
        assert "{sys_instruct}" in prompt
        assert "{formatted_chat}" in prompt
        assert "{context}" in prompt
        assert "{message}" in prompt

    def test_chinese_rag_prompt_content(self) -> None:
        """Test Chinese RAG prompt content."""
        config = BotConfig(language="CN")
        prompts = load_prompts(config)

        prompt = prompts["retrieve_contextualize_q_prompt"]
        assert "{chat_history}" in prompt
        assert "构造一个可以独立理解的问题" in prompt

    def test_chinese_worker_prompt_content(self) -> None:
        """Test Chinese worker prompt content."""
        config = BotConfig(language="CN")
        prompts = load_prompts(config)

        prompt = prompts["choose_worker_prompt"]
        assert "{workers_info}" in prompt
        assert "{task}" in prompt
        assert "{formatted_chat}" in prompt
        assert "{workers_name}" in prompt
        assert "工具" in prompt

    def test_chinese_database_prompts_content(self) -> None:
        """Test Chinese database prompts content."""
        config = BotConfig(language="CN")
        prompts = load_prompts(config)

        # Test database action prompt
        action_prompt = prompts["database_action_prompt"]
        assert "{actions_info}" in action_prompt
        assert "{user_intent}" in action_prompt
        assert "{actions_name}" in action_prompt
        assert "操作" in action_prompt

        # Test database slot prompt
        slot_prompt = prompts["database_slot_prompt"]
        assert "{slot}" in slot_prompt
        assert "{value}" in slot_prompt
        assert "{value_list}" in slot_prompt
        assert "重新构造" in slot_prompt


class TestLoadPromptsErrorHandling:
    """Test error handling in load_prompts function."""

    def test_load_prompts_unsupported_language(self) -> None:
        """Test load_prompts with unsupported language."""
        config = BotConfig(language="FR")

        with pytest.raises(ValueError, match="Unsupported language: FR"):
            load_prompts(config)

    def test_load_prompts_empty_language(self) -> None:
        """Test load_prompts with empty language."""
        config = BotConfig(language="")

        with pytest.raises(ValueError, match="Unsupported language: "):
            load_prompts(config)

    def test_load_prompts_none_language(self) -> None:
        """Test load_prompts with None language."""
        config = BotConfig(language=None)

        with pytest.raises(ValueError, match="Unsupported language: None"):
            load_prompts(config)


class TestPromptFormatting:
    """Test prompt formatting and structure."""

    def test_english_prompts_have_consistent_formatting(self) -> None:
        """Test that English prompts have consistent formatting."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        # Check that generator prompts have consistent structure
        generator_prompts = [
            "generator_prompt",
            "context_generator_prompt",
            "message_generator_prompt",
            "message_flow_generator_prompt",
        ]

        for prompt_key in generator_prompts:
            prompt = prompts[prompt_key]
            assert "----------------" in prompt  # Section separators
            assert "assistant" in prompt.lower()  # Assistant identifier

    def test_chinese_prompts_have_consistent_formatting(self) -> None:
        """Test that Chinese prompts have consistent formatting."""
        config = BotConfig(language="CN")
        prompts = load_prompts(config)

        # Check that generator prompts have consistent structure
        generator_prompts = [
            "generator_prompt",
            "context_generator_prompt",
            "message_generator_prompt",
            "message_flow_generator_prompt",
        ]

        for prompt_key in generator_prompts:
            prompt = prompts[prompt_key]
            assert "----------------" in prompt  # Section separators
            assert "助手" in prompt  # Assistant identifier

    def test_speech_prompts_contain_speech_indicators(self) -> None:
        """Test that speech prompts contain appropriate speech indicators."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        speech_prompts = [
            "generator_prompt_speech",
            "context_generator_prompt_speech",
            "message_generator_prompt_speech",
            "message_flow_generator_prompt_speech",
        ]

        for prompt_key in speech_prompts:
            prompt = prompts[prompt_key]
            assert "voice assistant" in prompt
            assert "speech" in prompt
            assert "SSML" in prompt or "spoken" in prompt

    def test_prompts_contain_required_placeholders(self) -> None:
        """Test that prompts contain required placeholders."""
        config = BotConfig(language="EN")
        prompts = load_prompts(config)

        # Check that prompts contain expected placeholders
        assert "{sys_instruct}" in prompts["generator_prompt"]
        assert "{formatted_chat}" in prompts["generator_prompt"]
        assert "{context}" in prompts["context_generator_prompt"]
        assert "{message}" in prompts["message_generator_prompt"]
        assert "{workers_info}" in prompts["choose_worker_prompt"]
        assert "{task}" in prompts["choose_worker_prompt"]


class TestPromptLanguageDifferences:
    """Test differences between language variants."""

    def test_english_vs_chinese_prompt_lengths(self) -> None:
        """Test that English and Chinese prompts have reasonable lengths."""
        en_config = BotConfig(language="EN")
        cn_config = BotConfig(language="CN")

        en_prompts = load_prompts(en_config)
        cn_prompts = load_prompts(cn_config)

        # Check that prompts are not empty
        for key in ["generator_prompt", "context_generator_prompt"]:
            assert len(en_prompts[key]) > 100, f"English prompt {key} too short"
            assert len(cn_prompts[key]) > 100, f"Chinese prompt {key} too short"

    def test_english_has_more_prompts_than_chinese(self) -> None:
        """Test that English has more prompt variants than Chinese."""
        en_config = BotConfig(language="EN")
        cn_config = BotConfig(language="CN")

        en_prompts = load_prompts(en_config)
        cn_prompts = load_prompts(cn_config)

        # English should have speech variants that Chinese doesn't have
        assert len(en_prompts) > len(cn_prompts)

        # Check for speech-specific prompts in English
        speech_prompts = [key for key in en_prompts if "speech" in key]
        assert len(speech_prompts) > 0, "English should have speech prompts"

        # Chinese should not have speech prompts
        cn_speech_prompts = [key for key in cn_prompts if "speech" in key]
        assert len(cn_speech_prompts) == 0, "Chinese should not have speech prompts"
