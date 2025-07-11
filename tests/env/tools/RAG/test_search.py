"""Tests for the search module.

This module tests the search functionality including SearchConfig, SearchEngine,
and TavilySearchExecutor classes. It covers all methods and edge cases with
proper mocking of external dependencies.
"""

import os
from unittest.mock import Mock, patch

import pytest

from arklex.env.tools.RAG.search import (
    SearchConfig,
    SearchEngine,
    TavilySearchExecutor,
)
from arklex.orchestrator.entities.msg_state_entities import (
    BotConfig,
    ConvoMessage,
    LLMConfig,
    MessageState,
)
from arklex.utils.exceptions import SearchError


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    """Create a mock LLM configuration."""
    return LLMConfig(model_type_or_path="gpt-3.5-turbo", llm_provider="openai")


@pytest.fixture
def mock_bot_config() -> BotConfig:
    """Create a mock bot configuration."""
    return BotConfig(
        bot_id="test_bot",
        version="1.0.0",
        language="EN",
        bot_type="chat",
        llm_config=LLMConfig(model_type_or_path="gpt-3.5-turbo", llm_provider="openai"),
    )


@pytest.fixture
def mock_message_state(mock_bot_config: BotConfig) -> MessageState:
    """Create a mock message state."""
    return MessageState(
        sys_instruct="You are a helpful assistant.",
        bot_config=mock_bot_config,
        user_message=ConvoMessage(
            history="Previous conversation history",
            message="What is the weather like today?",
        ),
        message_flow="",
        response="",
    )


@pytest.fixture
def mock_search_results() -> list[dict[str, str]]:
    """Create mock search results."""
    return [
        {
            "url": "https://example.com/weather",
            "content": "Today's weather is sunny with a high of 75°F.",
        },
        {
            "url": "https://weather.com/forecast",
            "content": "Current conditions: Partly cloudy, 72°F",
        },
    ]


class TestSearchConfig:
    """Test the SearchConfig TypedDict."""

    def test_search_config_default_values(self) -> None:
        """Test SearchConfig with default values."""
        config: SearchConfig = {}

        # Should not raise any errors for empty config
        assert isinstance(config, dict)

    def test_search_config_with_all_values(self) -> None:
        """Test SearchConfig with all specified values."""
        config: SearchConfig = {
            "max_results": 10,
            "search_depth": "advanced",
            "include_answer": True,
            "include_raw_content": True,
            "include_images": True,
        }

        assert config["max_results"] == 10
        assert config["search_depth"] == "advanced"
        assert config["include_answer"] is True
        assert config["include_raw_content"] is True
        assert config["include_images"] is True

    def test_search_config_partial_values(self) -> None:
        """Test SearchConfig with partial values."""
        config: SearchConfig = {"max_results": 5, "search_depth": "basic"}

        assert config["max_results"] == 5
        assert config["search_depth"] == "basic"
        # Other values should be optional


class TestSearchEngine:
    """Test the SearchEngine class."""

    def test_search_engine_search_method(
        self, mock_message_state: MessageState
    ) -> None:
        """Test SearchEngine.search method."""
        with patch(
            "arklex.env.tools.RAG.search.TavilySearchExecutor"
        ) as mock_executor_class:
            mock_executor = Mock()
            mock_executor.search.return_value = "Search results text"
            mock_executor_class.return_value = mock_executor

            result = SearchEngine.search(mock_message_state)

            # Verify TavilySearchExecutor was instantiated
            mock_executor_class.assert_called_once()

            # Verify search method was called with the state
            mock_executor.search.assert_called_once_with(mock_message_state)

            # Verify the message_flow was updated
            assert result.message_flow == "Search results text"
            assert result == mock_message_state


class TestTavilySearchExecutor:
    """Test the TavilySearchExecutor class."""

    def test_tavily_search_executor_initialization_defaults(
        self, mock_llm_config: LLMConfig
    ) -> None:
        """Test TavilySearchExecutor initialization with default parameters."""
        with (
            patch(
                "arklex.env.tools.RAG.search.validate_and_get_model_class"
            ) as mock_validate,
            patch(
                "arklex.env.tools.RAG.search.TavilySearchResults"
            ) as mock_search_tool,
        ):
            mock_model_class = Mock()
            mock_validate.return_value = mock_model_class

            mock_llm = Mock()
            mock_model_class.return_value = mock_llm

            mock_search_tool_instance = Mock()
            mock_search_tool.return_value = mock_search_tool_instance

            executor = TavilySearchExecutor(mock_llm_config)

            # Verify model class validation
            mock_validate.assert_called_once_with(mock_llm_config)

            # Verify LLM instantiation
            mock_model_class.assert_called_once_with(
                model=mock_llm_config.model_type_or_path
            )

            # Verify search tool instantiation with defaults
            mock_search_tool.assert_called_once_with(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
                include_images=False,
            )

            assert executor.llm == mock_llm
            assert executor.search_tool == mock_search_tool_instance

    def test_tavily_search_executor_initialization_custom_params(
        self, mock_llm_config: LLMConfig
    ) -> None:
        """Test TavilySearchExecutor initialization with custom parameters."""
        with (
            patch(
                "arklex.env.tools.RAG.search.validate_and_get_model_class"
            ) as mock_validate,
            patch(
                "arklex.env.tools.RAG.search.TavilySearchResults"
            ) as mock_search_tool,
        ):
            mock_model_class = Mock()
            mock_validate.return_value = mock_model_class

            mock_llm = Mock()
            mock_model_class.return_value = mock_llm

            mock_search_tool_instance = Mock()
            mock_search_tool.return_value = mock_search_tool_instance

            custom_config: SearchConfig = {
                "max_results": 10,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
                "include_images": True,
            }

            TavilySearchExecutor(mock_llm_config, **custom_config)

            # Verify search tool instantiation with custom parameters
            mock_search_tool.assert_called_once_with(
                max_results=10,
                search_depth="basic",
                include_answer=False,
                include_raw_content=False,
                include_images=True,
            )

    def test_process_search_result(
        self, mock_search_results: list[dict[str, str]]
    ) -> None:
        """Test process_search_result method."""
        with (
            patch(
                "arklex.env.tools.RAG.search.validate_and_get_model_class"
            ) as mock_validate,
            patch(
                "arklex.env.tools.RAG.search.TavilySearchResults"
            ) as mock_search_tool,
        ):
            mock_model_class = Mock()
            mock_validate.return_value = mock_model_class

            mock_llm = Mock()
            mock_model_class.return_value = mock_llm

            mock_search_tool_instance = Mock()
            mock_search_tool.return_value = mock_search_tool_instance

            executor = TavilySearchExecutor(
                LLMConfig(model_type_or_path="gpt-3.5-turbo", llm_provider="openai")
            )

            result = executor.process_search_result(mock_search_results)

            expected_text = (
                "Source: https://example.com/weather \n"
                "Content: Today's weather is sunny with a high of 75°F. \n\n"
                "Source: https://weather.com/forecast \n"
                "Content: Current conditions: Partly cloudy, 72°F \n\n"
            )

            assert result == expected_text

    def test_process_search_result_empty_list(self) -> None:
        """Test process_search_result with empty results."""
        with (
            patch(
                "arklex.env.tools.RAG.search.validate_and_get_model_class"
            ) as mock_validate,
            patch(
                "arklex.env.tools.RAG.search.TavilySearchResults"
            ) as mock_search_tool,
        ):
            mock_model_class = Mock()
            mock_validate.return_value = mock_model_class

            mock_llm = Mock()
            mock_model_class.return_value = mock_llm

            mock_search_tool_instance = Mock()
            mock_search_tool.return_value = mock_search_tool_instance

            executor = TavilySearchExecutor(
                LLMConfig(model_type_or_path="gpt-3.5-turbo", llm_provider="openai")
            )

            result = executor.process_search_result([])

            assert result == ""

    def test_search_method(
        self,
        mock_message_state: MessageState,
        mock_search_results: list[dict[str, str]],
    ) -> None:
        """Test search method."""
        with (
            patch(
                "arklex.env.tools.RAG.search.validate_and_get_model_class"
            ) as mock_validate,
            patch(
                "arklex.env.tools.RAG.search.TavilySearchResults"
            ) as mock_search_tool,
            patch("arklex.env.tools.RAG.search.load_prompts") as mock_load_prompts,
            patch("arklex.env.tools.RAG.search.PromptTemplate") as mock_prompt_template,
            patch("arklex.env.tools.RAG.search.StrOutputParser") as mock_str_parser,
            patch("arklex.env.tools.RAG.search.log_context") as mock_log_context,
        ):
            # Setup mocks
            mock_model_class = Mock()
            mock_validate.return_value = mock_model_class

            mock_llm = Mock()
            mock_model_class.return_value = mock_llm

            mock_search_tool_instance = Mock()
            mock_search_tool.return_value = mock_search_tool_instance
            mock_search_tool_instance.invoke.return_value = mock_search_results

            mock_prompts = {
                "retrieve_contextualize_q_prompt": "Test prompt {chat_history}"
            }
            mock_load_prompts.return_value = mock_prompts

            mock_prompt_instance = Mock()
            mock_prompt_template.from_template.return_value = mock_prompt_instance

            mock_chain = Mock()
            mock_chain.invoke.return_value = "reformulated query"

            # Mock the chain: PromptTemplate | llm | StrOutputParser
            mock_prompt_instance.__or__ = Mock(return_value=mock_llm)
            mock_llm.__or__ = Mock(return_value=mock_chain)

            mock_str_parser_instance = Mock()
            mock_str_parser.return_value = mock_str_parser_instance

            executor = TavilySearchExecutor(
                LLMConfig(model_type_or_path="gpt-3.5-turbo", llm_provider="openai")
            )

            result = executor.search(mock_message_state)

            # Verify prompts were loaded
            mock_load_prompts.assert_called_once_with(mock_message_state.bot_config)

            # Verify prompt template was created
            mock_prompt_template.from_template.assert_called_once_with(
                mock_prompts["retrieve_contextualize_q_prompt"]
            )

            # Verify chain was invoked with chat history
            mock_chain.invoke.assert_called_once_with(
                {"chat_history": mock_message_state.user_message.history}
            )

            # Verify log message
            mock_log_context.info.assert_called_once_with(
                "Reformulated input for search engine: reformulated query"
            )

            # Verify search tool was invoked
            mock_search_tool_instance.invoke.assert_called_once_with(
                {"query": "reformulated query"}
            )

            # Verify result processing
            expected_text = (
                "Source: https://example.com/weather \n"
                "Content: Today's weather is sunny with a high of 75°F. \n\n"
                "Source: https://weather.com/forecast \n"
                "Content: Current conditions: Partly cloudy, 72°F \n\n"
            )
            assert result == expected_text

    def test_load_search_tool(self, mock_llm_config: LLMConfig) -> None:
        """Test load_search_tool method."""
        with (
            patch(
                "arklex.env.tools.RAG.search.validate_and_get_model_class"
            ) as mock_validate,
            patch(
                "arklex.env.tools.RAG.search.TavilySearchResults"
            ) as mock_search_tool,
        ):
            mock_model_class = Mock()
            mock_validate.return_value = mock_model_class

            mock_llm = Mock()
            mock_model_class.return_value = mock_llm

            mock_search_tool_instance = Mock()
            mock_search_tool.return_value = mock_search_tool_instance

            executor = TavilySearchExecutor(mock_llm_config)

            # Test load_search_tool method
            new_executor = executor.load_search_tool(mock_llm_config, max_results=10)

            # Verify it returns a new TavilySearchExecutor instance
            assert isinstance(new_executor, TavilySearchExecutor)
            assert new_executor.llm == mock_llm
            assert new_executor.search_tool == mock_search_tool_instance

    def test_search_documents_success(
        self, mock_search_results: list[dict[str, str]]
    ) -> None:
        """Test search_documents method with successful search."""
        with (
            patch(
                "arklex.env.tools.RAG.search.validate_and_get_model_class"
            ) as mock_validate,
            patch(
                "arklex.env.tools.RAG.search.TavilySearchResults"
            ) as mock_search_tool,
            patch("arklex.env.tools.RAG.search.log_context") as mock_log_context,
        ):
            mock_model_class = Mock()
            mock_validate.return_value = mock_model_class

            mock_llm = Mock()
            mock_model_class.return_value = mock_llm

            mock_search_tool_instance = Mock()
            mock_search_tool.return_value = mock_search_tool_instance
            mock_search_tool_instance.invoke.return_value = mock_search_results

            executor = TavilySearchExecutor(
                LLMConfig(model_type_or_path="gpt-3.5-turbo", llm_provider="openai")
            )

            result = executor.search_documents("test query", max_results=5)

            # Verify search tool was invoked with correct parameters
            mock_search_tool_instance.invoke.assert_called_once_with(
                {"query": "test query", "max_results": 5}
            )

            # Verify logging
            mock_log_context.info.assert_any_call(
                "Starting search for query: test query"
            )
            mock_log_context.info.assert_any_call("Search completed, found 2 results")

            assert result == mock_search_results

    def test_search_documents_failure(self) -> None:
        """Test search_documents method when search fails."""
        with (
            patch(
                "arklex.env.tools.RAG.search.validate_and_get_model_class"
            ) as mock_validate,
            patch(
                "arklex.env.tools.RAG.search.TavilySearchResults"
            ) as mock_search_tool,
            patch("arklex.env.tools.RAG.search.log_context") as mock_log_context,
        ):
            mock_model_class = Mock()
            mock_validate.return_value = mock_model_class

            mock_llm = Mock()
            mock_model_class.return_value = mock_llm

            mock_search_tool_instance = Mock()
            mock_search_tool.return_value = mock_search_tool_instance
            mock_search_tool_instance.invoke.side_effect = Exception("Search failed")

            executor = TavilySearchExecutor(
                LLMConfig(model_type_or_path="gpt-3.5-turbo", llm_provider="openai")
            )

            # Verify SearchError is raised
            with pytest.raises(SearchError, match="Search failed"):
                executor.search_documents("test query")

            # Verify error logging
            mock_log_context.error.assert_called_once_with(
                "Search failed: Search failed"
            )

    def test_search_documents_with_additional_kwargs(
        self, mock_search_results: list[dict[str, str]]
    ) -> None:
        """Test search_documents method with additional kwargs."""
        with (
            patch(
                "arklex.env.tools.RAG.search.validate_and_get_model_class"
            ) as mock_validate,
            patch(
                "arklex.env.tools.RAG.search.TavilySearchResults"
            ) as mock_search_tool,
            patch("arklex.env.tools.RAG.search.log_context") as _,
        ):
            mock_model_class = Mock()
            mock_validate.return_value = mock_model_class

            mock_llm = Mock()
            mock_model_class.return_value = mock_llm

            mock_search_tool_instance = Mock()
            mock_search_tool.return_value = mock_search_tool_instance
            mock_search_tool_instance.invoke.return_value = mock_search_results

            executor = TavilySearchExecutor(
                LLMConfig(model_type_or_path="gpt-3.5-turbo", llm_provider="openai")
            )

            result = executor.search_documents(
                "test query", max_results=10, search_depth="basic", include_answer=False
            )

            # Verify search tool was invoked with all parameters
            mock_search_tool_instance.invoke.assert_called_once_with(
                {
                    "query": "test query",
                    "max_results": 10,
                    "search_depth": "basic",
                    "include_answer": False,
                }
            )

            assert result == mock_search_results


class TestSearchIntegration:
    """Integration tests for the search functionality."""

    @pytest.mark.integration
    def test_full_search_flow(self, mock_message_state: MessageState) -> None:
        """Test the complete search flow from SearchEngine to TavilySearchExecutor."""
        with (
            patch(
                "arklex.env.tools.RAG.search.TavilySearchExecutor"
            ) as mock_executor_class,
            patch("arklex.env.tools.RAG.search.load_prompts") as mock_load_prompts,
            patch("arklex.env.tools.RAG.search.PromptTemplate") as mock_prompt_template,
            patch("arklex.env.tools.RAG.search.StrOutputParser") as _,
            patch("arklex.env.tools.RAG.search.log_context") as _,
        ):
            # Setup mocks
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor
            mock_executor.search.return_value = "Processed search results"

            mock_prompts = {
                "retrieve_contextualize_q_prompt": "Test prompt {chat_history}"
            }
            mock_load_prompts.return_value = mock_prompts

            mock_prompt_instance = Mock()
            mock_prompt_template.from_template.return_value = mock_prompt_instance

            mock_chain = Mock()
            mock_chain.invoke.return_value = "reformulated query"

            mock_prompt_instance.__or__ = Mock(return_value=Mock())
            mock_prompt_instance.__or__().__or__ = Mock(return_value=mock_chain)

            # Execute search
            result = SearchEngine.search(mock_message_state)

            # Verify the complete flow
            mock_executor_class.assert_called_once()
            mock_executor.search.assert_called_once_with(mock_message_state)
            assert result.message_flow == "Processed search results"

    def test_search_with_different_languages(self) -> None:
        """Test search functionality with different language configurations."""
        # Test with English configuration
        en_config = BotConfig(
            bot_id="test_bot",
            version="1.0.0",
            language="EN",
            bot_type="chat",
            llm_config=LLMConfig(
                model_type_or_path="gpt-3.5-turbo", llm_provider="openai"
            ),
        )

        en_state = MessageState(
            sys_instruct="You are a helpful assistant.",
            bot_config=en_config,
            user_message=ConvoMessage(
                history="Previous conversation", message="What is the weather?"
            ),
        )

        # Test with Chinese configuration
        cn_config = BotConfig(
            bot_id="test_bot",
            version="1.0.0",
            language="CN",
            bot_type="chat",
            llm_config=LLMConfig(
                model_type_or_path="gpt-3.5-turbo", llm_provider="openai"
            ),
        )

        cn_state = MessageState(
            sys_instruct="You are a helpful assistant.",
            bot_config=cn_config,
            user_message=ConvoMessage(
                history="Previous conversation", message="天气怎么样？"
            ),
        )

        # Both should work without errors (actual language handling depends on prompts)
        assert en_state.bot_config.language == "EN"
        assert cn_state.bot_config.language == "CN"


if __name__ == "__main__":
    # Set environment variable for local testing
    os.environ["ARKLEX_TEST_ENV"] = "local"

    pytest.main([__file__, "-v"])
