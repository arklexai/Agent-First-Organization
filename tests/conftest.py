"""Test configuration and fixtures for the Arklex framework.

This module provides test configuration, fixtures, and mocks for testing the Arklex framework.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import sys


# Mock OpenAI API key for testing
@pytest.fixture(autouse=True)
def mock_openai_api_key():
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
def mock_shopify():
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
def mock_openai_llm_and_embeddings():
    """Mock LangChain OpenAI LLM and Embeddings for testing."""
    # Only mock if we're in test mode
    if os.getenv("ARKLEX_TEST_ENV") == "local":
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
def mock_openai_client():
    """Mock openai.OpenAI and openai.resources.embeddings.Embeddings.create for all tests."""
    # Only mock if we're in test mode
    if os.getenv("ARKLEX_TEST_ENV") == "local":
        import openai
        from unittest.mock import patch, MagicMock

        dummy_embedding = [0.0] * 1536
        dummy_response = MagicMock()
        dummy_response.model_dump.return_value = {
            "data": [{"embedding": dummy_embedding}]
        }

        def fake_embed_documents(texts, *args, **kwargs):
            return [dummy_embedding for _ in texts]

        def fake_embed_query(text, *args, **kwargs):
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


# Patch IntentDetector.execute to always return a dummy intent for tests
@pytest.fixture(autouse=True)
def mock_intent_detector_execute():
    """Mock IntentDetector.execute to always return a dummy intent for tests."""
    # Only mock if we're in test mode
    if os.getenv("ARKLEX_TEST_ENV") == "local":
        from arklex.orchestrator.NLU.core.intent import IntentDetector
        from unittest.mock import patch

        with patch.object(IntentDetector, "execute", return_value="others"):
            yield
    else:
        yield
