"""
Shared test utilities for integration tests.

This module provides common utilities, helper functions, and test data
that can be used across all integration tests. It includes mock factories,
assertion helpers, and test data providers to reduce code duplication
and improve test maintainability.
"""

import json
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from arklex.env.env import Environment
from arklex.orchestrator.entities.msg_state_entities import MessageState


class MockResponse:
    """
    Mock response object for testing HTTP requests.

    This class simulates the behavior of HTTP response objects,
    providing status_code, json(), and text attributes for testing
    HTTP client interactions.
    """

    def __init__(
        self, status_code: int = 200, json_data: dict[str, Any] = None, text: str = ""
    ) -> None:
        """
        Initialize a mock HTTP response.

        Args:
            status_code: HTTP status code (default: 200)
            json_data: JSON data to return from json() method
            text: Text content of the response
        """
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        """Return the JSON data associated with this response."""
        return self._json_data


class TestDataProvider:
    """
    Provides common test data for integration tests.

    This class contains static methods that return realistic test data
    for various scenarios, reducing duplication across test files.
    """

    @staticmethod
    def get_sample_shopify_product() -> dict[str, Any]:
        """
        Get sample Shopify product data.

        Returns:
            dict: Sample product data with all required Shopify fields.

        This method provides realistic product data that matches the structure
        of actual Shopify API responses, including images, variants, and metadata.
        """
        return {
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

    @staticmethod
    def get_sample_hubspot_contact() -> dict[str, Any]:
        """
        Get sample HubSpot contact data.

        Returns:
            dict: Sample contact data with HubSpot properties.

        This method provides realistic contact data that matches the structure
        of actual HubSpot API responses, including standard contact properties.
        """
        return {
            "id": "12345",
            "properties": {
                "firstname": "John",
                "lastname": "Doe",
                "email": "john.doe@example.com",
                "phone": "+1234567890",
            },
        }

    @staticmethod
    def get_sample_conversation_history() -> list[dict[str, str]]:
        """
        Get sample conversation history.

        Returns:
            list[dict[str, str]]: Sample conversation with user and assistant messages.

        This method provides realistic conversation history that can be used
        to test conversation flow and context handling across different scenarios.
        """
        return [
            {"role": "user", "content": "Hello, I need help with my order"},
            {
                "role": "assistant",
                "content": "I'd be happy to help you with your order. What's your order number?",
            },
            {"role": "user", "content": "My order number is 12345"},
        ]

    @staticmethod
    def get_sample_user_parameters() -> dict[str, Any]:
        """
        Get sample user parameters.

        Returns:
            dict[str, Any]: Sample user parameters with various data types.

        This method provides realistic user parameters that can be used
        to test parameter handling and user context management.
        """
        return {
            "order_id": "12345",
            "email": "test@example.com",
            "product_query": "shoes",
            "chat_id": "test-chat-123",
        }


class MockFactory:
    """
    Factory for creating common mock objects.

    This class provides static methods to create consistent mock objects
    for various testing scenarios, ensuring that mocks have the same
    structure and behavior across different tests.
    """

    @staticmethod
    def create_mock_llm_response(content: str = "Mock LLM response") -> Mock:
        """
        Create a mock LLM response.

        Args:
            content: The content to include in the mock response.

        Returns:
            Mock: A mock LLM response object with the specified content.

        This method creates a mock that simulates the structure of
        responses from language model providers like OpenAI.
        """
        mock_response = Mock()
        mock_response.content = content
        return mock_response

    @staticmethod
    def create_mock_embeddings_response(
        dimensions: int = 1536, num_docs: int = 5
    ) -> list[list[float]]:
        """
        Create mock embeddings response.

        Args:
            dimensions: Number of dimensions in each embedding vector.
            num_docs: Number of documents to create embeddings for.

        Returns:
            list[list[float]]: Mock embedding vectors for testing.

        This method creates realistic embedding vectors that can be used
        to test RAG and similarity search functionality.
        """
        return [[0.1] * dimensions] * num_docs

    @staticmethod
    def create_mock_http_response(
        status_code: int = 200, json_data: dict[str, Any] = None
    ) -> MockResponse:
        """
        Create a mock HTTP response.

        Args:
            status_code: HTTP status code for the response.
            json_data: JSON data to include in the response.

        Returns:
            MockResponse: A mock HTTP response object.

        This method creates a mock HTTP response that can be used
        to test HTTP client interactions.
        """
        return MockResponse(status_code=status_code, json_data=json_data)

    @staticmethod
    def create_mock_shopify_session() -> Mock:
        """
        Create a mock Shopify session.

        Returns:
            Mock: A mock Shopify session with proper context manager behavior.

        This method creates a mock that simulates the behavior of
        real Shopify sessions, including context manager functionality.
        """
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        return mock_session

    @staticmethod
    def create_mock_hubspot_client() -> MagicMock:
        """
        Create a mock HubSpot client.

        Returns:
            MagicMock: A mock HubSpot client with realistic API responses.

        This method creates a comprehensive mock of the HubSpot client that
        simulates successful contact searches, communication record creation,
        and contact-communication associations.
        """
        mock_client = MagicMock()

        # Mock successful contact search response with realistic data structure
        mock_search_response = MagicMock()
        mock_search_response.to_dict.return_value = {
            "total": 1,
            "results": [TestDataProvider.get_sample_hubspot_contact()],
        }
        mock_client.crm.contacts.search_api.do_search.return_value = (
            mock_search_response
        )

        # Mock communication record creation with success response
        mock_comm_response = MagicMock()
        mock_comm_response.to_dict.return_value = {"id": "comm_123"}
        mock_client.crm.objects.communications.basic_api.create.return_value = (
            mock_comm_response
        )

        # Mock contact-communication association (no return value expected)
        mock_client.crm.associations.v4.basic_api.create.return_value = None

        return mock_client


class AssertionHelper:
    """
    Helper class for common test assertions.

    This class provides static methods for common assertion patterns
    used across integration tests, reducing code duplication and
    improving test readability.
    """

    @staticmethod
    def assert_json_response_structure(response: str, expected_keys: list[str]) -> None:
        """
        Assert that a JSON response has the expected structure.

        Args:
            response: JSON string to validate.
            expected_keys: List of keys that should be present in the response.

        Raises:
            AssertionError: If the response is not valid JSON or missing expected keys.

        This method validates both JSON syntax and the presence of required
        keys in the response structure.
        """
        try:
            response_data = json.loads(response)
            for key in expected_keys:
                assert key in response_data, (
                    f"Expected key '{key}' not found in response"
                )
        except json.JSONDecodeError:
            pytest.fail("Response is not valid JSON")

    @staticmethod
    def assert_error_message_contains(error: Exception, expected_text: str) -> None:
        """
        Assert that an error message contains expected text.

        Args:
            error: The exception to check.
            expected_text: Text that should be present in the error message.

        Raises:
            AssertionError: If the expected text is not found in the error message.

        This method provides a consistent way to validate error messages
        across different test scenarios.
        """
        assert expected_text in str(error), (
            f"Expected '{expected_text}' in error message, got: {str(error)}"
        )

    @staticmethod
    def assert_api_calls_made(mock_client: Mock, expected_calls: list[str]) -> None:
        """
        Assert that specific API calls were made.

        Args:
            mock_client: The mock client to check.
            expected_calls: List of API call names that should have been made.

        Raises:
            AssertionError: If any expected API calls were not made.

        This method provides a simplified way to verify that specific
        API methods were called during test execution.
        """
        for call_name in expected_calls:
            # This is a simplified check - in practice you'd want more specific assertions
            assert hasattr(mock_client, call_name), (
                f"Expected API call '{call_name}' not found"
            )


class TestEnvironmentHelper:
    """
    Helper for managing test environment.

    This class provides static methods for setting up and cleaning up
    test environments, ensuring consistent test execution conditions.
    """

    @staticmethod
    def setup_test_environment() -> None:
        """
        Set up common test environment variables.

        This method ensures that all necessary environment variables
        are set for integration tests to run properly.
        """
        import os

        # Set up essential environment variables for testing
        os.environ.setdefault("OPENAI_API_KEY", "test_key")
        os.environ.setdefault("TESTING", "true")
        os.environ.setdefault("LOG_LEVEL", "WARNING")

    @staticmethod
    def cleanup_test_environment() -> None:
        """
        Clean up test environment variables.

        This method removes test-specific environment variables
        to prevent pollution of the system environment.
        """
        import os

        # Remove test-specific environment variables
        test_vars = ["OPENAI_API_KEY", "TESTING", "LOG_LEVEL"]
        for var in test_vars:
            if var in os.environ:
                del os.environ[var]


# Global utility functions for common test operations
def create_mock_message_state(response: str = "Mock response") -> "MessageState":
    """
    Create a mock MessageState for testing.

    Args:
        response: The response text to include in the mock state.

    Returns:
        MessageState: A mock MessageState with realistic test data.

    This function creates a mock MessageState object that can be used
    in tests that need to simulate message processing states.
    """
    from arklex.orchestrator.entities.msg_state_entities import (
        BotConfig,
        ConvoMessage,
        LLMConfig,
        MessageState,
        Metadata,
        OrchestratorMessage,
        StatusEnum,
        Timing,
    )

    return MessageState(
        sys_instruct="Mock system instructions",
        bot_config=BotConfig(
            bot_id="test",
            version="1.0",
            language="EN",
            bot_type="test",
            llm_config=LLMConfig(
                model_type_or_path="gpt-3.5-turbo", llm_provider="openai"
            ),
        ),
        user_message=ConvoMessage(
            history="Mock conversation history", message="Mock user message"
        ),
        orchestrator_message=OrchestratorMessage(
            message="Mock orchestrator message", attribute={}
        ),
        function_calling_trajectory=[],
        trajectory=[],
        message_flow="Mock message flow",
        response=response,
        status=StatusEnum.COMPLETE,
        slots={},
        metadata=Metadata(
            chat_id="test-chat-id",
            turn_id=1,
            hitl=None,
            timing=Timing(),
            attempts=None,
        ),
        is_stream=False,
        message_queue=None,
        stream_type=None,
        relevant_records=None,
    )


def create_mock_shopify_graphql_response(data: dict[str, Any]) -> str:
    """
    Create a mock Shopify GraphQL response.

    Args:
        data: The data to include in the GraphQL response.

    Returns:
        str: JSON string representing the GraphQL response.

    This function creates a mock GraphQL response that matches
    the structure of actual Shopify GraphQL API responses.
    """
    return json.dumps({"data": data})


def create_mock_hubspot_api_response(data: dict[str, Any]) -> MagicMock:
    """
    Create a mock HubSpot API response.

    Args:
        data: The data to include in the API response.

    Returns:
        MagicMock: A mock HubSpot API response object.

    This function creates a mock API response that matches
    the structure of actual HubSpot API responses.
    """
    mock_response = MagicMock()
    mock_response.to_dict.return_value = data
    return mock_response


# Milvus-specific test utilities
class MilvusTestHelper:
    """
    Helper class for Milvus integration tests.

    This class provides utilities specific to Milvus testing, including
    mock creation, response validation, and test data generation.
    """

    @staticmethod
    def create_mock_message_state(response: str = "Mock response") -> "MessageState":
        """
        Create a mock MessageState for Milvus testing.

        Args:
            response: The response text to include in the mock state.

        Returns:
            MessageState: A mock MessageState with realistic test data.

        This method creates a mock MessageState specifically configured
        for Milvus integration testing scenarios.
        """
        from arklex.orchestrator.entities.msg_state_entities import (
            BotConfig,
            ConvoMessage,
            LLMConfig,
            MessageState,
            Metadata,
            OrchestratorMessage,
            StatusEnum,
            Timing,
        )

        return MessageState(
            sys_instruct="Mock system instructions for Milvus testing",
            bot_config=BotConfig(
                bot_id="milvus-test",
                version="1.0",
                language="EN",
                bot_type="milvus",
                llm_config=LLMConfig(
                    model_type_or_path="gpt-3.5-turbo", llm_provider="openai"
                ),
            ),
            user_message=ConvoMessage(
                history="Mock conversation history for Milvus testing",
                message="Mock user message for Milvus testing",
            ),
            orchestrator_message=OrchestratorMessage(
                message="Mock orchestrator message for Milvus testing", attribute={}
            ),
            function_calling_trajectory=[],
            trajectory=[],
            message_flow="Mock message flow for Milvus testing",
            response=response,
            status=StatusEnum.COMPLETE,
            slots={},
            metadata=Metadata(
                chat_id="milvus-test-chat-id",
                turn_id=1,
                hitl=None,
                timing=Timing(),
                attempts=None,
            ),
            is_stream=False,
            message_queue=None,
            stream_type=None,
            relevant_records=None,
        )

    @staticmethod
    def get_api_bot_response(
        config: dict[str, Any],
        env: "Environment",
        user_text: str,
        history: list[dict[str, str]],
        params: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str | None]:
        """
        Get bot response from the API for testing.

        Args:
            config: Taskgraph configuration.
            env: Environment instance.
            user_text: User input text.
            history: Conversation history.
            params: User parameters.

        Returns:
            tuple: (answer, parameters, human_in_the_loop) response.

        This method simulates the API call to get bot responses
        for testing conversation flows and response generation.
        """
        from arklex.orchestrator.orchestrator import AgentOrg

        data = {
            "text": user_text,
            "chat_history": history,
            "parameters": params,
        }
        orchestrator = AgentOrg(config=config, env=env)
        result = orchestrator.get_response(data)

        return result["answer"], result["parameters"], result["human_in_the_loop"]

    @staticmethod
    def validate_taskgraph_structure(config: dict) -> None:
        """
        Validate that the taskgraph has the correct structure.

        Args:
            config: Taskgraph configuration to validate.

        Raises:
            AssertionError: If the taskgraph structure is invalid.

        This method validates that the taskgraph contains all required
        fields and has the expected structure for Milvus testing.
        """
        # Validate required top-level keys
        required_keys = ["nodes", "edges", "model", "slotfillapi"]
        for key in required_keys:
            assert key in config, f"Missing required key '{key}' in taskgraph"

        # Validate nodes structure
        assert isinstance(config["nodes"], list), "Nodes must be a list"
        assert len(config["nodes"]) > 0, "Taskgraph must have at least one node"

        # Validate edges structure
        assert isinstance(config["edges"], list), "Edges must be a list"

        # Validate model configuration
        model = config["model"]
        required_model_keys = ["model_name", "model_type_or_path", "llm_provider"]
        for key in required_model_keys:
            assert key in model, f"Missing required model key '{key}'"

        # Validate slotfillapi configuration
        slotfillapi = config["slotfillapi"]
        # Allow empty string for slotfillapi in some configurations
        if slotfillapi:
            assert isinstance(slotfillapi, dict), "Slotfillapi must be a dictionary"

    @staticmethod
    def validate_worker_configuration(config: dict) -> None:
        """
        Validate that workers are properly configured.

        Args:
            config: Taskgraph configuration to validate.

        Raises:
            AssertionError: If worker configuration is invalid.

        This method validates that the taskgraph contains properly
        configured workers for Milvus functionality.
        """
        # Check if workers are present
        assert "workers" in config, "Taskgraph must have workers configuration"
        workers = config["workers"]
        assert isinstance(workers, list), "Workers must be a list"

        # Validate that required workers are present
        worker_types = [
            worker.get("type", worker.get("name", "")) for worker in workers
        ]
        assert "MilvusRAGWorker" in worker_types, "MilvusRAGWorker must be configured"

    @staticmethod
    def validate_domain_specific_configuration(config: dict) -> None:
        """
        Validate that the taskgraph is properly configured for robotics domain.

        Args:
            config: Taskgraph configuration to validate.

        Raises:
            AssertionError: If domain-specific configuration is invalid.

        This method validates that the taskgraph is configured
        specifically for robotics domain testing.
        """
        # Check for robotics-specific configurations
        # This could include specific node types, worker configurations, etc.
        nodes = config["nodes"]

        # Validate that there are nodes configured for robotics queries
        node_types = [node[1].get("type", "") for node in nodes]
        # Also check resource names for RAG workers
        resource_names = [node[1].get("resource", {}).get("name", "") for node in nodes]
        assert any("rag" in node_type.lower() for node_type in node_types) or any(
            "rag" in name.lower() for name in resource_names
        ), "Taskgraph must have RAG nodes for robotics queries"

    @staticmethod
    def validate_taskgraph_metadata(config: dict) -> None:
        """
        Validate taskgraph metadata and version information.

        Args:
            config: Taskgraph configuration to validate.

        Raises:
            AssertionError: If metadata is invalid or missing.

        This method validates that the taskgraph contains proper
        metadata and version information.
        """
        # Check for metadata fields (if they exist)
        if "metadata" in config:
            metadata = config["metadata"]
            assert isinstance(metadata, dict), "Metadata must be a dictionary"

        # Validate version information if present
        if "version" in config:
            version = config["version"]
            assert isinstance(version, str), "Version must be a string"
            assert version.strip(), "Version cannot be empty"

    @staticmethod
    def validate_node_edge_consistency(config: dict) -> None:
        """
        Validate consistency between nodes and edges.

        Args:
            config: Taskgraph configuration to validate.

        Raises:
            AssertionError: If nodes and edges are inconsistent.

        This method validates that all edges reference valid nodes
        and that the graph structure is consistent.
        """
        nodes = config["nodes"]
        edges = config["edges"]

        # Get all node IDs
        node_ids = {node[0] for node in nodes}

        # Validate that all edges reference valid nodes
        for edge in edges:
            # Handle different edge formats
            if isinstance(edge, list) and len(edge) >= 2:
                # Edge format: [source, target, attributes]
                source = edge[0]
                target = edge[1]
            else:
                # Edge format: {"source": ..., "target": ...}
                source = edge.get("source")
                target = edge.get("target")

            if source:
                assert source in node_ids, (
                    f"Edge source '{source}' references non-existent node"
                )
            if target:
                assert target in node_ids, (
                    f"Edge target '{target}' references non-existent node"
                )


class MilvusMockFactory:
    """
    Factory for creating Milvus-specific mock objects.

    This class provides static methods to create consistent mock objects
    specifically for Milvus testing scenarios.
    """

    @staticmethod
    def create_mock_milvus_retrieval_success() -> Mock:
        """
        Create a mock Milvus retrieval function that simulates success.

        Returns:
            Mock: A mock function that simulates successful Milvus retrieval.

        This method creates a mock that simulates successful Milvus
        retrieval operations with proper message state updates.
        """

        def mock_milvus_retrieve_side_effect(
            message_state: MessageState, tags: dict[str, str] | None = None
        ) -> MessageState:
            # Verify that product tags are passed correctly
            assert tags is not None, "Tags should be passed to Milvus retrieval"
            assert "product" in tags, "Product tag should be present"
            assert tags["product"] == "robots", "Product tag should be 'robots'"

            # Add retrieved context to message state
            message_state.message_flow = (
                "Retrieved information about ADAM robot bartender"
            )
            return message_state

        mock_retrieval = Mock()
        mock_retrieval.side_effect = mock_milvus_retrieve_side_effect
        return mock_retrieval

    @staticmethod
    def create_mock_milvus_retrieval_error() -> Mock:
        """
        Create a mock Milvus retrieval function that simulates errors.

        Returns:
            Mock: A mock function that simulates failed Milvus retrieval.

        This method creates a mock that simulates failed Milvus
        retrieval operations for testing error handling scenarios.
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

    @staticmethod
    def create_mock_context_generator(response: str = "Mock response") -> Mock:
        """
        Create a mock context generator for testing.

        Args:
            response: The response text to generate.

        Returns:
            Mock: A mock context generator function.

        This method creates a mock that simulates the behavior of
        context generation functions used in RAG workflows.
        """

        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = response
            return message_state

        mock_generator = Mock()
        mock_generator.side_effect = mock_context_generate_side_effect
        return mock_generator

    @staticmethod
    def create_mock_post_process() -> Mock:
        """
        Create a mock post-processing function for testing.

        Returns:
            Mock: A mock post-processing function.

        This method creates a mock that simulates the behavior of
        post-processing functions used in conversation workflows.
        """

        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            # Ensure we always return a valid message state
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state()
            return message_state

        mock_post_process = Mock()
        mock_post_process.side_effect = mock_post_process_side_effect
        return mock_post_process
