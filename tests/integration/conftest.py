"""
Common fixtures and configuration for integration tests.

This module provides shared fixtures and configuration for all integration tests,
including proper environment setup and common test utilities. It handles mocking
of external services, test data preparation, and environment variable management.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from arklex.orchestrator.entities.msg_state_entities import MessageState

# Add the project root to the Python path to ensure imports work correctly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Mock the mysql module BEFORE any other imports to prevent connection issues
# This prevents actual database connections during testing
sys.modules["arklex.utils.mysql"] = MagicMock()

# Set up common environment variables for integration tests
# These provide default values that prevent test failures due to missing env vars
os.environ.setdefault("OPENAI_API_KEY", "test_key")
os.environ.setdefault("MYSQL_USERNAME", "test_user")
os.environ.setdefault("MYSQL_PASSWORD", "test_password")
os.environ.setdefault("MYSQL_HOSTNAME", "localhost")
os.environ.setdefault("MYSQL_PORT", "3306")
os.environ.setdefault("MYSQL_DB_NAME", "test_db")
os.environ.setdefault("DATA_DIR", "./examples/hitl_server")
os.environ.setdefault("ARKLEX_TEST_ENV", "local")
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """
    Provide the test data directory path.

    Returns:
        Path: Path to the test data directory containing example configurations.

    This fixture provides a consistent path to test data files that are
    shared across multiple test sessions.
    """
    return project_root / "examples" / "hitl_server"


@pytest.fixture(scope="session")
def hitl_taskgraph_path(test_data_dir: Path) -> Path:
    """
    Provide the path to the HITL taskgraph configuration.

    Args:
        test_data_dir: Path to the test data directory.

    Returns:
        Path: Path to the HITL taskgraph.json configuration file.

    This fixture provides the path to the main taskgraph configuration
    used for HITL integration tests.
    """
    return test_data_dir / "taskgraph.json"


@pytest.fixture(scope="session")
def load_hitl_config(hitl_taskgraph_path: Path) -> dict[str, Any]:
    """
    Load the HITL taskgraph configuration.

    Args:
        hitl_taskgraph_path: Path to the taskgraph configuration file.

    Returns:
        dict: Loaded taskgraph configuration with test model settings.

    This fixture loads the taskgraph configuration and sets up test-specific
    model configuration to ensure tests run with predictable settings.
    """
    with open(hitl_taskgraph_path, encoding="utf-8") as f:
        config = json.load(f)

    # Set up model configuration for testing with predictable settings
    model = {
        "model_name": "gpt-3.5-turbo",
        "model_type_or_path": "gpt-3.5-turbo",
        "llm_provider": "openai",
        "api_key": "test_key",
        "endpoint": "https://api.openai.com/v1",
    }
    config["model"] = model
    config["input_dir"] = str(hitl_taskgraph_path.parent)

    return config


@pytest.fixture(scope="session")
def mock_environment_variables() -> None:
    """
    Set up mock environment variables for testing.

    This fixture temporarily sets test environment variables and restores
    the original environment after the test session completes. This ensures
    tests run in a controlled environment without affecting the system.
    """
    original_env = os.environ.copy()

    # Set test environment variables with safe defaults
    test_env_vars = {
        "OPENAI_API_KEY": "test_key",
        "DATA_DIR": "./examples/hitl_server",
        "MYSQL_USERNAME": "test_user",
        "MYSQL_PASSWORD": "test_password",
        "MYSQL_HOSTNAME": "localhost",
        "MYSQL_PORT": "3306",
        "MYSQL_DB_NAME": "test_db",
        "ARKLEX_TEST_ENV": "local",
        "TESTING": "true",
        "LOG_LEVEL": "WARNING",
    }

    for key, value in test_env_vars.items():
        os.environ[key] = value

    yield

    # Restore original environment to prevent test pollution
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def clean_test_state() -> None:
    """
    Ensure clean state for each test function.

    This fixture can be used to reset any state between tests.
    Currently serves as a placeholder for future cleanup logic.
    """
    # This fixture can be used to reset any state between tests
    yield
    # Cleanup can be added here if needed


# Shopify-specific fixtures
@pytest.fixture
def mock_shopify_session() -> Mock:
    """
    Create a mock Shopify session for testing.

    Returns:
        Mock: A mock Shopify session with proper context manager behavior.

    This fixture provides a mock Shopify session that can be used as a
    context manager, simulating the behavior of real Shopify sessions.
    """
    mock_session = Mock()
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=None)
    return mock_session


@pytest.fixture
def mock_shopify_graphql() -> Mock:
    """
    Create a mock Shopify GraphQL client for testing.

    Returns:
        Mock: A mock GraphQL client that returns test data.

    This fixture provides a mock GraphQL client that simulates Shopify's
    GraphQL API responses with test data.
    """
    mock_graphql = Mock()
    mock_graphql_instance = Mock()
    mock_graphql_instance.execute.return_value = json.dumps({"data": {"test": "value"}})
    mock_graphql.return_value = mock_graphql_instance
    return mock_graphql


@pytest.fixture
def sample_shopify_product_data() -> dict[str, Any]:
    """
    Provide sample Shopify product data for testing.

    Returns:
        dict: Sample product data with all required fields for testing.

    This fixture provides realistic sample data that matches the structure
    of actual Shopify product responses, including images, variants, and metadata.
    """
    return {
        "data": {
            "products": {
                "nodes": [
                    {
                        "id": "gid://shopify/Product/12345",
                        "title": "Test Product",
                        "description": "A test product description",
                        "handle": "test-product",
                        "onlineStoreUrl": "https://test-shop.myshopify.com/products/test-product",
                        "images": {
                            "edges": [
                                {
                                    "node": {
                                        "src": "https://cdn.shopify.com/test-image.jpg",
                                        "altText": "Test Product Image",
                                    }
                                }
                            ]
                        },
                        "variants": {
                            "nodes": [
                                {
                                    "displayName": "Small",
                                    "id": "gid://shopify/ProductVariant/67890",
                                    "price": "29.99",
                                    "inventoryQuantity": 10,
                                }
                            ]
                        },
                    }
                ],
                "pageInfo": {
                    "endCursor": "cursor123",
                    "hasNextPage": False,
                    "hasPreviousPage": False,
                    "startCursor": "cursor456",
                },
            }
        }
    }


# HubSpot-specific fixtures
@pytest.fixture
def mock_hubspot_client() -> MagicMock:
    """
    Create a mock HubSpot client with predefined responses.

    Returns:
        MagicMock: A mock HubSpot client with realistic API responses.

    This fixture provides a comprehensive mock of the HubSpot client that
    simulates successful contact searches, communication record creation,
    and contact-communication associations.
    """
    mock_client = MagicMock()

    # Mock successful contact search response with realistic data structure
    mock_search_response = MagicMock()
    mock_search_response.to_dict.return_value = {
        "total": 1,
        "results": [
            {
                "id": "12345",
                "properties": {
                    "firstname": "John",
                    "lastname": "Doe",
                    "email": "john.doe@example.com",
                },
            }
        ],
    }
    mock_client.crm.contacts.search_api.do_search.return_value = mock_search_response

    # Mock communication record creation with success response
    mock_comm_response = MagicMock()
    mock_comm_response.to_dict.return_value = {"id": "comm_123"}
    mock_client.crm.objects.communications.basic_api.create.return_value = (
        mock_comm_response
    )

    # Mock contact-communication association (no return value expected)
    mock_client.crm.associations.v4.basic_api.create.return_value = None

    return mock_client


@pytest.fixture
def mock_hubspot_ticket_client() -> MagicMock:
    """
    Create a mock HubSpot client specifically for ticket operations.

    Returns:
        MagicMock: A mock HubSpot client configured for ticket testing.

    This fixture provides a mock HubSpot client that simulates ticket
    creation and association operations with realistic responses.
    """
    mock_client = MagicMock()

    # Mock successful ticket creation
    mock_ticket_response = MagicMock()
    mock_ticket_response.to_dict.return_value = {"id": "ticket_123"}
    mock_client.crm.tickets.basic_api.create.return_value = mock_ticket_response

    # Mock ticket-contact association
    mock_client.crm.associations.v4.basic_api.create.return_value = None

    return mock_client


@pytest.fixture
def mock_hubspot_meeting_client() -> MagicMock:
    """
    Create a mock HubSpot client specifically for meeting operations.

    Returns:
        MagicMock: A mock HubSpot client configured for meeting testing.

    This fixture provides a mock HubSpot client that simulates meeting
    creation and availability checking with realistic responses.
    """
    mock_client = MagicMock()

    # Mock successful meeting creation
    mock_meeting_response = MagicMock()
    mock_meeting_response.to_dict.return_value = {"id": "meeting_123"}
    mock_client.crm.objects.meetings.basic_api.create.return_value = (
        mock_meeting_response
    )

    # Mock meeting-contact association
    mock_client.crm.associations.v4.basic_api.create.return_value = None

    return mock_client


@pytest.fixture
def mock_message_state() -> Mock:
    """
    Create a mock MessageState for testing.

    Returns:
        Mock: A mock MessageState with realistic test data.

    This fixture provides a mock MessageState object that can be used
    in tests that need to simulate message processing states.
    """
    mock_state = Mock()
    mock_state.response = "Mock response"
    mock_state.message_flow = "Mock message flow"
    mock_state.status = "COMPLETE"
    mock_state.slots = {}
    mock_state.metadata = Mock()
    mock_state.metadata.chat_id = "test-chat-id"
    mock_state.metadata.turn_id = 1
    mock_state.metadata.hitl = None
    return mock_state


@pytest.fixture
def mock_llm_response() -> Mock:
    """
    Create a mock LLM response for testing.

    Returns:
        Mock: A mock LLM response with test content.

    This fixture provides a mock LLM response that simulates the
    structure of responses from language model providers.
    """
    mock_response = Mock()
    mock_response.content = "Mock LLM response"
    return mock_response


@pytest.fixture
def mock_embeddings_response() -> list[list[float]]:
    """
    Create mock embeddings response for testing.

    Returns:
        list[list[float]]: Mock embedding vectors for testing.

    This fixture provides realistic embedding vectors that can be used
    to test RAG and similarity search functionality.
    """
    return [[0.1] * 1536] * 5  # 5 documents with 1536-dimensional embeddings


@pytest.fixture
def sample_conversation_history() -> list[dict[str, str]]:
    """
    Provide sample conversation history for testing.

    Returns:
        list[dict[str, str]]: Sample conversation history with user and assistant messages.

    This fixture provides realistic conversation history that can be used
    to test conversation flow and context handling.
    """
    return [
        {"role": "user", "content": "Hello, I need help with my order"},
        {
            "role": "assistant",
            "content": "I'd be happy to help you with your order. What's your order number?",
        },
        {"role": "user", "content": "My order number is 12345"},
    ]


@pytest.fixture
def sample_user_parameters() -> dict[str, Any]:
    """
    Provide sample user parameters for testing.

    Returns:
        dict[str, Any]: Sample user parameters with various data types.

    This fixture provides realistic user parameters that can be used
    to test parameter handling and user context management.
    """
    return {
        "order_id": "12345",
        "email": "test@example.com",
        "product_query": "shoes",
        "chat_id": "test-chat-123",
    }


# Milvus-specific fixtures
@pytest.fixture(scope="session")
def milvus_taskgraph_path() -> Path:
    """
    Provide the path to the Milvus taskgraph configuration.

    Returns:
        Path: Path to the Milvus taskgraph.json configuration file.

    This fixture provides the path to the taskgraph configuration
    used for Milvus integration tests.
    """
    return project_root / "examples" / "milvus_filter" / "taskgraph.json"


@pytest.fixture(scope="session")
def load_milvus_config(milvus_taskgraph_path: Path) -> dict[str, Any]:
    """
    Load the Milvus taskgraph configuration.

    Args:
        milvus_taskgraph_path: Path to the Milvus taskgraph configuration file.

    Returns:
        dict: Loaded taskgraph configuration with test model settings.

    This fixture loads the Milvus taskgraph configuration and sets up
    test-specific model configuration for Milvus integration tests.
    """
    with open(milvus_taskgraph_path, encoding="utf-8") as f:
        config = json.load(f)

    # Set up model configuration for testing with predictable settings
    model = {
        "model_name": "gpt-3.5-turbo",
        "model_type_or_path": "gpt-3.5-turbo",
        "llm_provider": "openai",
        "api_key": "test_key",
        "endpoint": "https://api.openai.com/v1",
    }
    config["model"] = model
    config["input_dir"] = str(milvus_taskgraph_path.parent)

    return config


@pytest.fixture(scope="session")
def config_and_env(load_milvus_config: dict) -> tuple[dict, Any, str]:
    """
    Load config and environment for Milvus tests.

    Args:
        load_milvus_config: Loaded Milvus taskgraph configuration.

    Returns:
        tuple: (config, environment, start_message) for Milvus testing.

    This fixture sets up the complete test environment for Milvus integration
    tests, including configuration loading, environment initialization, and
    start message extraction.
    """
    from arklex.env.env import Environment
    from arklex.orchestrator.NLU.services.model_service import ModelService

    config = load_milvus_config

    # Initialize model service with test configuration
    model_service = ModelService(config["model"])

    # Initialize environment with test settings
    env = Environment(
        tools=config.get("tools", []),
        workers=config.get("workers", []),
        agents=config.get("agents", []),
        slot_fill_api=config["slotfillapi"],
        planner_enabled=True,
        model_service=model_service,
    )

    # Find start message from taskgraph configuration
    start_message = (
        "Hello! I'm here to help you with information about our robotics products."
    )
    for node in config["nodes"]:
        if node[1].get("type", "") == "start":
            start_message = node[1]["attribute"]["value"]
            break

    return config, env, start_message


@pytest.fixture
def mock_milvus_retrieval() -> Mock:
    """
    Create a mock Milvus retrieval function for testing.

    Returns:
        Mock: A mock function that simulates Milvus retrieval behavior.

    This fixture provides a mock that simulates successful Milvus retrieval
    operations with proper message state updates.
    """

    def mock_milvus_retrieve_side_effect(
        message_state: MessageState, tags: dict[str, str] | None = None
    ) -> MessageState:
        # Verify that product tags are passed correctly
        assert tags is not None, "Tags should be passed to Milvus retrieval"
        assert "product" in tags, "Product tag should be present"
        assert tags["product"] == "robots", "Product tag should be 'robots'"

        # Add retrieved context to message state
        message_state.message_flow = "Retrieved information about ADAM robot bartender"
        return message_state

    mock_retrieval = Mock()
    mock_retrieval.side_effect = mock_milvus_retrieve_side_effect
    return mock_retrieval


@pytest.fixture
def mock_milvus_error_retrieval() -> Mock:
    """
    Create a mock Milvus retrieval function that simulates errors.

    Returns:
        Mock: A mock function that simulates Milvus retrieval errors.

    This fixture provides a mock that simulates failed Milvus retrieval
    operations for testing error handling scenarios.
    """

    def mock_milvus_retrieve_error_side_effect(
        message_state: MessageState, tags: dict[str, str] | None = None
    ) -> MessageState:
        # Simulate retrieval error
        message_state.message_flow = "Error: Failed to retrieve information"
        return message_state

    mock_retrieval = Mock()
    mock_retrieval.side_effect = mock_milvus_retrieve_error_side_effect
    return mock_retrieval


@pytest.fixture
def sample_milvus_product_queries() -> list[str]:
    """
    Provide sample product queries for Milvus testing.

    Returns:
        list[str]: Sample product queries for testing Milvus filtering.

    This fixture provides realistic product queries that can be used
    to test Milvus product filtering functionality.
    """
    return [
        "Tell me about your robot bartender",
        "What robotics products do you have?",
        "I'm interested in the ADAM robot",
        "Show me your automation solutions",
    ]


@pytest.fixture
def sample_milvus_conversation_history() -> list[dict[str, str]]:
    """
    Provide sample conversation history for Milvus testing.

    Returns:
        list[dict[str, str]]: Sample conversation history for Milvus tests.

    This fixture provides realistic conversation history that can be used
    to test Milvus integration with conversation context.
    """
    return [
        {"role": "user", "content": "Hello, I'm looking for robotics products"},
        {
            "role": "assistant",
            "content": "I'd be happy to help you find robotics products. What specific type are you interested in?",
        },
        {"role": "user", "content": "I'm interested in the ADAM robot bartender"},
    ]


# Pytest configuration functions
def pytest_configure(config: pytest.Config) -> None:
    """
    Configure pytest for integration tests.

    Args:
        config: Pytest configuration object.

    This function sets up pytest configuration specific to integration tests,
    including marker registration and test discovery settings.
    """
    # Register custom markers for integration tests
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "hitl: marks tests as HITL (Human-in-the-Loop) tests"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow running tests")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Modify test collection for integration tests.

    Args:
        config: Pytest configuration object.
        items: List of collected test items.

    This function applies default markers to tests based on their location
    and ensures proper categorization of integration tests.
    """
    # Add integration marker to all tests in this directory
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark HITL tests
        if "hitl" in str(item.fspath) or "hitl" in item.name:
            item.add_marker(pytest.mark.hitl)

        # Mark Shopify tests
        if "shopify" in str(item.fspath) or "shopify" in item.name:
            item.add_marker(pytest.mark.shopify)

        # Mark HubSpot tests
        if "hubspot" in str(item.fspath) or "hubspot" in item.name:
            item.add_marker(pytest.mark.hubspot)

        # Mark slow tests (can be customized based on test names)
        if any(
            slow_indicator in item.name.lower()
            for slow_indicator in ["slow", "integration", "e2e"]
        ):
            item.add_marker(pytest.mark.slow)
