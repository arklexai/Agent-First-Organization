"""Test utilities for the Arklex framework."""

from unittest.mock import Mock, patch

from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState
from arklex.resources.tools.utils import trace


class TestGetPromptTemplate:
    """Test cases for get_prompt_template function."""

    @patch("arklex.resources.tools.utils.load_prompts")
    def test_get_prompt_template_speech_non_chinese(
        self, mock_load_prompts: Mock
    ) -> None:
        """Test get_prompt_template for speech non-Chinese."""
        # Setup
        state = Mock(spec=OrchestratorState)
        state.stream_type = "speech"
        state.bot_config = Mock()
        state.bot_config.language = "EN"

        mock_prompts = {
            "test_prompt": "Regular prompt",
            "test_prompt_speech": "Speech prompt",
        }
        mock_load_prompts.return_value = mock_prompts

        # Execute
        from arklex.resources.tools.utils import get_prompt_template

        result = get_prompt_template(state, "test_prompt")

        # Assert
        assert result.template == "Speech prompt"
        mock_load_prompts.assert_called_once_with(state.bot_config.language)

    @patch("arklex.resources.tools.utils.load_prompts")
    def test_get_prompt_template_speech_chinese(self, mock_load_prompts: Mock) -> None:
        """Test get_prompt_template for speech Chinese."""
        # Setup
        state = Mock(spec=OrchestratorState)
        state.stream_type = "speech"
        state.bot_config = Mock()
        state.bot_config.language = "CN"

        mock_prompts = {
            "test_prompt": "Regular prompt",
            "test_prompt_speech": "Speech prompt",
        }
        mock_load_prompts.return_value = mock_prompts

        # Execute
        from arklex.resources.tools.utils import get_prompt_template

        result = get_prompt_template(state, "test_prompt")

        # Assert
        assert result.template == "Regular prompt"
        mock_load_prompts.assert_called_once_with(state.bot_config.language)

    @patch("arklex.resources.tools.utils.load_prompts")
    def test_get_prompt_template_non_speech(self, mock_load_prompts: Mock) -> None:
        """Test get_prompt_template for non-speech."""
        # Setup
        state = Mock(spec=OrchestratorState)
        state.stream_type = "text"
        state.bot_config = Mock()
        state.bot_config.language = "EN"

        mock_prompts = {
            "test_prompt": "Regular prompt",
            "test_prompt_speech": "Speech prompt",
        }
        mock_load_prompts.return_value = mock_prompts

        # Execute
        from arklex.resources.tools.utils import get_prompt_template

        result = get_prompt_template(state, "test_prompt")

        # Assert
        assert result.template == "Regular prompt"
        mock_load_prompts.assert_called_once_with(state.bot_config.language)


class TestTrace:
    """Test cases for trace function."""

    def test_trace(self) -> None:
        """Test trace function."""
        # Setup
        state = Mock(spec=OrchestratorState)
        state.trajectory = [[Mock()]]
        state.trajectory[-1][-1].steps = []

        # Execute
        result = trace("test input", "test_function", state)

        # Assert
        assert result is state  # trace returns the modified state object
        assert len(state.trajectory[-1][-1].steps) == 1
        assert state.trajectory[-1][-1].steps[0] == {"test_function": "test input"}
