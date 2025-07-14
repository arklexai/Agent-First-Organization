"""Test configuration and fixtures for the Arklex framework.

This module provides test configuration, fixtures, and mocks for testing the Arklex framework.
"""

import os
import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from arklex.main import app
from arklex.utils.logging_config import setup_logging

# Mock the mysql module BEFORE any other imports to prevent connection issues
# This prevents actual database connections during testing
if os.getenv("ARKLEX_TEST_ENV") == "local":
    sys.modules["arklex.utils.mysql"] = MagicMock()

# Set up common environment variables for local testing
if os.getenv("ARKLEX_TEST_ENV") == "local":
    os.environ.setdefault("MYSQL_USERNAME", "test_user")
    os.environ.setdefault("MYSQL_PASSWORD", "test_password")
    os.environ.setdefault("MYSQL_HOSTNAME", "localhost")
    os.environ.setdefault("MYSQL_PORT", "3306")
    os.environ.setdefault("MYSQL_DB_NAME", "test_db")


# Mock OpenAI API key for testing
@pytest.fixture(autouse=True)
def mock_openai_api_key() -> Generator[None, None, None]:
    """Mock OpenAI API key for testing."""
    # Only mock if we're in test mode
    if os.getenv("ARKLEX_TEST_ENV") == "local":
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
            yield
    else:
        # Ensure real API key is set
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable is not set")
        yield


# Mock Shopify module for testing
@pytest.fixture(autouse=True)
def mock_shopify() -> Generator[MagicMock, None, None]:
    """Mock Shopify module for testing."""
    # Only mock if we're in test mode
    if os.getenv("ARKLEX_TEST_ENV") == "local":
        shopify_mock = MagicMock()
        sys.modules["shopify"] = shopify_mock
        yield shopify_mock
        del sys.modules["shopify"]
    else:
        yield


# Mock OpenAI LLM and Embeddings for testing
@pytest.fixture(autouse=True)
def mock_openai_llm_and_embeddings(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    """Mock LangChain OpenAI LLM and Embeddings for testing."""
    # Only mock if we're in test mode and not marked to skip LLM mocking
    if os.getenv("ARKLEX_TEST_ENV") == "local" and not request.node.get_closest_marker(
        "no_llm_mock"
    ):
        with (
            patch("langchain_openai.ChatOpenAI") as mock_llm,
            patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings,
        ):
            # Mock LLM: always returns a dummy response
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = "dummy response"
            mock_llm.return_value = mock_llm_instance

            # Mock Embeddings: always returns a list of dummy vectors
            mock_embeddings_instance = MagicMock()
            mock_embeddings_instance.embed_documents.return_value = [[0.0] * 1536]
            mock_embeddings_instance.embed_query.return_value = [0.0] * 1536
            mock_embeddings.return_value = mock_embeddings_instance

            yield
    else:
        yield


# Patch openai client to prevent real API calls
@pytest.fixture(autouse=True)
def mock_openai_client(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Mock openai.OpenAI and openai.resources.embeddings.Embeddings.create for all tests."""
    # Only mock if we're in test mode and not marked to skip LLM mocking
    if os.getenv("ARKLEX_TEST_ENV") == "local" and not request.node.get_closest_marker(
        "no_llm_mock"
    ):
        from unittest.mock import MagicMock, patch

        import openai

        dummy_embedding = [0.0] * 1536
        dummy_response = MagicMock()
        dummy_response.model_dump.return_value = {
            "data": [{"embedding": dummy_embedding}]
        }

        def fake_embed_documents(
            texts: list[str], *args: tuple[Any, ...], **kwargs: dict[str, Any]
        ) -> list[list[float]]:
            return [dummy_embedding for _ in texts]

        def fake_embed_query(
            text: str, *args: tuple[Any, ...], **kwargs: dict[str, Any]
        ) -> list[float]:
            return dummy_embedding

        # Create a mock chat completion response
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"result": "dummy response"}', role="assistant"
                )
            )
        ]
        mock_chat_completion.model_dump.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"result": "dummy response"}',
                        "role": "assistant",
                    }
                }
            ]
        }

        # Patch all OpenAI client methods
        with (
            patch(
                "langchain_openai.OpenAIEmbeddings.embed_documents",
                side_effect=fake_embed_documents,
            ),
            patch(
                "langchain_openai.OpenAIEmbeddings.embed_query",
                side_effect=fake_embed_query,
            ),
            patch.object(openai.OpenAI, "embeddings", create=True) as mock_embeddings,
            patch.object(openai.OpenAI, "chat", create=True) as mock_chat,
        ):
            mock_embeddings.create.return_value = dummy_response
            mock_chat.completions.create.return_value = mock_chat_completion
            yield
    else:
        yield


# Patch IntentDetector.predict_intent to return context-aware intents for tests
@pytest.fixture(autouse=True)
def mock_intent_detector_execute(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    """Mock IntentDetector.predict_intent to return context-aware intents for tests."""
    # Only mock if we're in test mode and not testing error handling
    if os.getenv("ARKLEX_TEST_ENV") == "local" and not request.node.get_closest_marker(
        "no_intent_mock"
    ):
        from unittest.mock import patch

        from arklex.orchestrator.NLU.core.intent import IntentDetector

        def mock_predict_intent(
            self: object,
            text: str,
            intents: dict[str, Any],
            chat_history_str: str,
            model_config: dict[str, Any],
            **kwargs: object,
        ) -> str:
            """Mock predict_intent method that returns context-aware intents."""
            # Check if the text matches any intent's sample utterances
            for intent_name, intent_list in intents.items():
                for intent_data in intent_list:
                    if (
                        "attribute" in intent_data
                        and "sample_utterances" in intent_data["attribute"]
                    ):
                        sample_utterances = intent_data["attribute"][
                            "sample_utterances"
                        ]
                        if text.lower() in [u.lower() for u in sample_utterances]:
                            return intent_name

            # If no match found, return "others" as fallback
            return "others"

        with patch.object(
            IntentDetector,
            "predict_intent",
            side_effect=mock_predict_intent,
            autospec=True,
        ):
            yield
    else:
        yield


@pytest.fixture(scope="session")
def test_client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging() -> None:
    """Set up test logging configuration."""
    # Create a test logs directory
    test_log_dir = os.path.join(os.getcwd(), "test_logs")
    os.makedirs(test_log_dir, exist_ok=True)

    # Set up logging for tests
    setup_logging(log_dir=test_log_dir, log_level="DEBUG", app_name="arklex_test")


@pytest.fixture
def mock_request_id() -> str:
    """Generate a mock request ID for testing."""
    return "test-request-id-123"


@pytest.fixture
def sample_error_details() -> dict[str, Any]:
    """Provide sample error details for testing."""
    return {"error": "Test error", "context": {"test": "context"}}
