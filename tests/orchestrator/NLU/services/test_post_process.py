import pytest
from unittest.mock import Mock, patch, MagicMock
from arklex.orchestrator.post_process import (
    post_process_response,
    _build_context,
    _extract_links,
    _extract_links_from_nested_dict,
    _remove_invalid_links,
    _rephrase_answer,
    _include_resource,
    RAG_NODES_STEPS,
)
from arklex.utils.graph_state import MessageState, Params, ResourceRecord


@pytest.fixture
def mock_message_state():
    state = Mock(spec=MessageState)
    state.user_message = Mock()
    state.response = (
        "Test response with [link](https://example.com) and https://test.com"
    )
    state.sys_instruct = "System instruction with https://system.com"
    state.trajectory = []
    state.bot_config = Mock()
    state.bot_config.llm_config = Mock()
    return state


@pytest.fixture
def mock_params():
    return Mock(spec=Params)


@pytest.fixture
def mock_resource_record():
    resource = Mock(spec=ResourceRecord)
    resource.output = "Resource output with https://resource.com"
    resource.info = {"id": "test_resource"}
    resource.steps = []
    return resource


class TestPostProcessResponse:
    """Test the post_process_response function."""

    def test_post_process_response_no_missing_links(
        self, mock_message_state, mock_params
    ) -> None:
        """Test post_process_response when no links are missing."""
        mock_message_state.trajectory = []

        with (
            patch(
                "arklex.orchestrator.post_process._build_context",
                return_value={"https://example.com", "https://test.com"},
            ),
            patch(
                "arklex.orchestrator.post_process._extract_links",
                return_value={"https://example.com", "https://test.com"},
            ),
        ):
            result = post_process_response(mock_message_state, mock_params, True)

            assert result == mock_message_state
            assert result.response == mock_message_state.response  # No changes

    def test_post_process_response_with_missing_links(
        self, mock_message_state, mock_params
    ) -> None:
        """Test post_process_response when links are missing."""
        mock_message_state.trajectory = []

        with (
            patch(
                "arklex.orchestrator.post_process._build_context",
                return_value={"https://example.com"},
            ),
            patch(
                "arklex.orchestrator.post_process._extract_links",
                return_value={"https://example.com", "https://missing.com"},
            ),
            patch(
                "arklex.orchestrator.post_process._remove_invalid_links",
                return_value="Cleaned response",
            ),
            patch(
                "arklex.orchestrator.post_process._rephrase_answer",
                return_value="Rephrased response",
            ),
        ):
            result = post_process_response(mock_message_state, mock_params, True)

            assert result == mock_message_state
            assert result.response == "Rephrased response"

    def test_post_process_response_with_missing_links_no_hitl(
        self, mock_message_state, mock_params
    ) -> None:
        """Test post_process_response when links are missing and HITL is not available."""
        mock_message_state.trajectory = []
        # Mock the LLM config to return proper string values
        mock_message_state.bot_config.llm_config.llm_provider = "openai"
        mock_message_state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"

        with (
            patch(
                "arklex.orchestrator.post_process._build_context",
                return_value={"https://example.com"},
            ),
            patch(
                "arklex.orchestrator.post_process._extract_links",
                return_value={"https://example.com", "https://missing.com"},
            ),
            patch(
                "arklex.orchestrator.post_process._remove_invalid_links",
                return_value="Cleaned response",
            ),
            patch(
                "arklex.orchestrator.post_process._rephrase_answer",
                return_value="Rephrased response",
            ),
        ):
            result = post_process_response(mock_message_state, mock_params, False)

            assert result == mock_message_state
            assert result.response == "Rephrased response"


class TestBuildContext:
    """Test the _build_context function."""

    def test_build_context_basic(self, mock_message_state) -> None:
        """Test _build_context with basic trajectory."""
        resource = Mock(spec=ResourceRecord)
        resource.output = "Output with https://resource.com"
        resource.info = {"id": "test_resource"}
        resource.steps = []

        mock_message_state.trajectory = [[resource]]

        with (
            patch(
                "arklex.orchestrator.post_process._extract_links",
                return_value={"https://resource.com"},
            ),
            patch(
                "arklex.orchestrator.post_process._include_resource", return_value=True
            ),
        ):
            result = _build_context(mock_message_state)

            assert "https://resource.com" in result

    def test_build_context_with_rag_node(self, mock_message_state) -> None:
        """Test _build_context with RAG node."""
        resource = Mock(spec=ResourceRecord)
        resource.output = "Output with https://resource.com"
        resource.info = {"id": "FaissRAGWorker"}
        resource.steps = [
            {"faiss_retrieve": {"content": "RAG content with https://rag.com"}}
        ]

        mock_message_state.trajectory = [[resource]]

        with (
            patch(
                "arklex.orchestrator.post_process._extract_links",
                return_value={"https://resource.com"},
            ),
            patch(
                "arklex.orchestrator.post_process._include_resource", return_value=True
            ),
            patch(
                "arklex.orchestrator.post_process._extract_links_from_nested_dict",
                return_value={"https://rag.com"},
            ),
        ):
            result = _build_context(mock_message_state)

            assert "https://resource.com" in result
            assert "https://rag.com" in result

    def test_build_context_with_rag_node_exception(self, mock_message_state) -> None:
        """Test _build_context with RAG node that raises exception."""
        resource = Mock(spec=ResourceRecord)
        resource.output = "Output with https://resource.com"
        resource.info = {"id": "FaissRAGWorker"}
        resource.steps = [{"faiss_retrieve": "invalid"}]

        mock_message_state.trajectory = [[resource]]

        with (
            patch(
                "arklex.orchestrator.post_process._extract_links",
                return_value={"https://resource.com"},
            ),
            patch(
                "arklex.orchestrator.post_process._include_resource", return_value=True
            ),
            patch(
                "arklex.orchestrator.post_process._extract_links_from_nested_dict",
                side_effect=Exception("Test error"),
            ),
        ):
            result = _build_context(mock_message_state)

            assert "https://resource.com" in result

    def test_build_context_with_context_generate_flag(self, mock_message_state):
        """Test _build_context with resource that has context_generate flag."""
        resource = Mock(spec=ResourceRecord)
        resource.output = "Output with https://resource.com"
        resource.info = {"id": "test_resource"}
        resource.steps = [{"context_generate": True}]

        mock_message_state.trajectory = [[resource]]
        mock_message_state.sys_instruct = "System instructions without links"

        # Mock _include_resource to return False for resources with context_generate flag
        with (
            patch(
                "arklex.orchestrator.post_process._extract_links",
                side_effect=lambda text: {"https://resource.com"}
                if "resource.com" in text
                else set(),
            ),
            patch(
                "arklex.orchestrator.post_process._include_resource", return_value=False
            ),
        ):
            result = _build_context(mock_message_state)

            # Should not include the resource output due to context_generate flag
            # The _include_resource function returns False, so the resource output is not added
            assert "https://resource.com" not in result


class TestExtractLinks:
    """Test the _extract_links function."""

    def test_extract_links_markdown_links(self) -> None:
        """Test _extract_links with markdown links."""
        text = (
            "Check out [this link](https://example.com) and [another](https://test.com)"
        )

        result = _extract_links(text)

        assert "https://example.com" in result
        assert "https://test.com" in result

    def test_extract_links_raw_links(self) -> None:
        """Test _extract_links with raw links."""
        text = "Visit https://example.com and http://test.com for more info"

        result = _extract_links(text)

        assert "https://example.com" in result
        assert "http://test.com" in result

    def test_extract_links_mixed_links(self) -> None:
        """Test _extract_links with mixed markdown and raw links."""
        text = "Check [this](https://example.com) and visit https://test.com"

        result = _extract_links(text)

        assert "https://example.com" in result
        assert "https://test.com" in result

    def test_extract_links_with_punctuation(self) -> None:
        """Test _extract_links with punctuation at the end."""
        text = "Visit https://example.com. and https://test.com!"

        result = _extract_links(text)

        assert "https://example.com" in result
        assert "https://test.com" in result

    def test_extract_links_empty_text(self) -> None:
        """Test _extract_links with empty text."""
        result = _extract_links("")

        assert result == set()

    def test_extract_links_no_links(self) -> None:
        """Test _extract_links with text containing no links."""
        result = _extract_links("This is just plain text without any links.")

        assert result == set()


class TestExtractLinksFromNestedDict:
    """Test the _extract_links_from_nested_dict function."""

    def test_extract_links_from_nested_dict_string(self) -> None:
        """Test _extract_links_from_nested_dict with string value."""
        with patch(
            "arklex.orchestrator.post_process._extract_links",
            return_value={"https://example.com"},
        ):
            result = _extract_links_from_nested_dict("Check https://example.com")

            assert "https://example.com" in result

    def test_extract_links_from_nested_dict_dict(self) -> None:
        """Test _extract_links_from_nested_dict with dictionary value."""
        nested_dict = {
            "key1": "Check https://example.com",
            "key2": "Visit https://test.com",
        }

        with patch(
            "arklex.orchestrator.post_process._extract_links",
            return_value={"https://example.com", "https://test.com"},
        ):
            result = _extract_links_from_nested_dict(nested_dict)

            assert "https://example.com" in result
            assert "https://test.com" in result

    def test_extract_links_from_nested_dict_list(self) -> None:
        """Test _extract_links_from_nested_dict with list value."""
        nested_list = ["Check https://example.com", "Visit https://test.com"]

        with patch(
            "arklex.orchestrator.post_process._extract_links",
            return_value={"https://example.com", "https://test.com"},
        ):
            result = _extract_links_from_nested_dict(nested_list)

            assert "https://example.com" in result
            assert "https://test.com" in result

    def test_extract_links_from_nested_dict_complex(self) -> None:
        """Test _extract_links_from_nested_dict with complex nested structure."""
        complex_structure = {
            "level1": {
                "level2": ["Check https://example.com"],
                "level2b": "Visit https://test.com",
            },
            "level1b": ["Another https://another.com"],
        }

        with patch(
            "arklex.orchestrator.post_process._extract_links",
            return_value={
                "https://example.com",
                "https://test.com",
                "https://another.com",
            },
        ):
            result = _extract_links_from_nested_dict(complex_structure)

            assert "https://example.com" in result
            assert "https://test.com" in result
            assert "https://another.com" in result


class TestRemoveInvalidLinks:
    """Test the _remove_invalid_links function."""

    def test_remove_invalid_links(self) -> None:
        """Test _remove_invalid_links with multiple links."""
        response = "Check https://example.com and https://invalid.com for info"
        links = {"https://invalid.com"}

        result = _remove_invalid_links(response, links)

        assert "https://invalid.com" not in result
        assert "https://example.com" in result

    def test_remove_invalid_links_multiple_invalid(self) -> None:
        """Test _remove_invalid_links with multiple invalid links."""
        response = (
            "Check https://example.com, https://invalid1.com, and https://invalid2.com"
        )
        links = {"https://invalid1.com", "https://invalid2.com"}

        result = _remove_invalid_links(response, links)

        assert "https://invalid1.com" not in result
        assert "https://invalid2.com" not in result
        assert "https://example.com" in result

    def test_remove_invalid_links_no_invalid(self) -> None:
        """Test _remove_invalid_links with no invalid links."""
        response = "Check https://example.com and https://test.com"
        links = set()

        result = _remove_invalid_links(response, links)

        assert result == response

    def test_remove_invalid_links_whitespace_cleanup(self) -> None:
        """Test _remove_invalid_links cleans up whitespace."""
        response = "Check https://example.com    https://invalid.com   for info"
        links = {"https://invalid.com"}

        result = _remove_invalid_links(response, links)

        assert "https://invalid.com" not in result
        assert "  " not in result  # No double spaces


class TestRephraseAnswer:
    """Test the _rephrase_answer function."""

    def test_rephrase_answer(self, mock_message_state) -> None:
        """Test _rephrase_answer function."""
        mock_message_state.bot_config.llm_config.llm_provider = "openai"
        mock_message_state.bot_config.llm_config.model_type_or_path = "gpt-3.5-turbo"

        with (
            patch("arklex.orchestrator.post_process.PROVIDER_MAP") as mock_provider_map,
            patch("arklex.orchestrator.post_process.load_prompts") as mock_load_prompts,
            patch(
                "arklex.orchestrator.post_process.PromptTemplate"
            ) as mock_prompt_template,
            patch(
                "arklex.orchestrator.post_process.StrOutputParser"
            ) as mock_str_parser,
        ):
            # Create a mock LLM that supports the pipe operator
            mock_llm = Mock()
            mock_llm.__or__ = Mock(return_value=Mock())

            # Configure the mock to return the LLM
            mock_provider_map.get.return_value = Mock(return_value=mock_llm)

            mock_prompts = {"regenerate_response": "test prompt template"}
            mock_load_prompts.return_value = mock_prompts

            mock_prompt = Mock()
            mock_prompt.invoke.return_value = Mock()
            mock_prompt_template.from_template.return_value = mock_prompt

            # Mock the final chain
            mock_final_chain = Mock()
            mock_final_chain.invoke.return_value = "Rephrased answer"
            mock_llm.__or__.return_value = mock_final_chain

            result = _rephrase_answer(mock_message_state)

            assert result == "Rephrased answer"


class TestIncludeResource:
    """Test the _include_resource function."""

    def test_include_resource_no_context_generate(self, mock_resource_record) -> None:
        """Test _include_resource when no context_generate flag is present."""
        mock_resource_record.steps = [{"step": "data"}]

        result = _include_resource(mock_resource_record)

        assert result is True

    def test_include_resource_with_context_generate(self, mock_resource_record) -> None:
        """Test _include_resource when context_generate flag is present."""
        mock_resource_record.steps = [{"context_generate": True}]

        result = _include_resource(mock_resource_record)

        assert result is False

    def test_include_resource_mixed_steps(self, mock_resource_record) -> None:
        """Test _include_resource with mixed steps."""
        mock_resource_record.steps = [
            {"step": "data"},
            {"context_generate": True},
            {"another": "step"},
        ]

        result = _include_resource(mock_resource_record)

        assert result is False

    def test_include_resource_empty_steps(self, mock_resource_record) -> None:
        """Test _include_resource with empty steps."""
        mock_resource_record.steps = []

        result = _include_resource(mock_resource_record)

        assert result is True


class TestRAGNodesSteps:
    """Test the RAG_NODES_STEPS constant."""

    def test_rag_nodes_steps_contains_expected_keys(self) -> None:
        """Test that RAG_NODES_STEPS contains expected keys."""
        expected_keys = {"FaissRAGWorker", "milvus_rag_worker", "rag_message_worker"}

        assert set(RAG_NODES_STEPS.keys()) == expected_keys

    def test_rag_nodes_steps_values(self) -> None:
        """Test that RAG_NODES_STEPS has correct values."""
        assert RAG_NODES_STEPS["FaissRAGWorker"] == "faiss_retrieve"
        assert RAG_NODES_STEPS["milvus_rag_worker"] == "milvus_retrieve"
        assert RAG_NODES_STEPS["rag_message_worker"] == "milvus_retrieve"
