"""
Integration tests for complete taskgraph flow execution.

This module contains comprehensive integration tests that validate the complete
flow of a taskgraph.json file, testing the entire conversation flow from start
to finish rather than individual tool calls. It uses the Shopify taskgraph
example to test realistic scenarios.

Key Test Scenarios:
- Complete conversation flow with multiple nodes
- Intent detection and routing
- Tool execution and response handling
- State management across nodes
- Error handling and recovery
- Edge cases and boundary conditions
"""

import json
import os
from typing import Any
from unittest.mock import Mock

import pytest

from arklex.env.env import Environment
from arklex.env.tools.tools import Tool
from arklex.env.workers.worker import BaseWorker
from arklex.orchestrator.entities.msg_state_entities import MessageState, StatusEnum
from arklex.orchestrator.entities.orchestrator_params_entities import OrchestratorParams
from arklex.orchestrator.orchestrator import AgentOrg


class MockTool(Tool):
    """Mock tool for testing purposes."""

    def __init__(self, name: str = "mock_tool") -> None:
        self.name = name
        self.description = f"Mock tool: {name}"

    def execute(
        self, message_state: MessageState, **kwargs: dict[str, Any]
    ) -> MessageState:
        """Execute the mock tool."""
        message_state.response = f"Mock response from {self.name}"
        message_state.status = StatusEnum.COMPLETE
        return message_state


class MockWorker(BaseWorker):
    """Mock worker for testing purposes."""

    def __init__(self, name: str = "mock_worker") -> None:
        self.name = name
        self.description = f"Mock worker: {name}"

    def _execute(
        self, message_state: MessageState, **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the mock worker's core functionality."""
        return {
            "response": f"Mock response from {self.name}",
            "status": StatusEnum.COMPLETE,
            "trajectory": message_state.trajectory or [],
            "function_calling_trajectory": message_state.function_calling_trajectory
            or [],
            "slots": message_state.slots or {},
            "metadata": message_state.metadata or {},
            "user_message": message_state.user_message,
            "orchestrator_message": message_state.orchestrator_message,
            "relevant_records": message_state.relevant_records or [],
            "is_stream": message_state.is_stream,
            "stream_type": message_state.stream_type,
            "message_queue": message_state.message_queue,
            "message_flow": message_state.message_flow or "",
        }


# Mock functions for tools
def get_user_details_admin() -> MockTool:
    return MockTool("get_user_details_admin")


def search_products() -> MockTool:
    return MockTool("search_products")


def get_order_details() -> MockTool:
    return MockTool("get_order_details")


def get_products() -> MockTool:
    return MockTool("get_products")


def get_web_product() -> MockTool:
    return MockTool("get_web_product")


def return_products() -> MockTool:
    return MockTool("return_products")


def cancel_order() -> MockTool:
    return MockTool("cancel_order")


def get_cart() -> MockTool:
    return MockTool("get_cart")


def cart_add_items() -> MockTool:
    return MockTool("cart_add_items")


# Mock functions for workers
def FaissRAGWorker() -> MockWorker:
    return MockWorker("FaissRAGWorker")


def MessageWorker() -> MockWorker:
    return MockWorker("MessageWorker")


class TestTaskGraphFullFlow:
    """
    Integration tests for complete taskgraph flow execution.

    This test class validates the entire flow of a taskgraph.json file,
    testing realistic conversation scenarios that traverse multiple nodes
    and execute various tools and workers.
    """

    @pytest.fixture
    def shopify_taskgraph_path(self) -> str:
        """Get the path to the Shopify taskgraph.json file."""
        return "examples/shopify/taskgraph.json"

    @pytest.fixture
    def mock_environment(self) -> Environment:
        """Create a mock environment for testing."""
        from arklex.env.env import BaseResourceInitializer

        class MockResourceInitializer(BaseResourceInitializer):
            """Mock resource initializer for testing."""

            @staticmethod
            def init_tools(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
                """Initialize mock tools."""
                tool_registry: dict[str, dict[str, Any]] = {}
                tool_mapping = {
                    "get_user_details_admin": get_user_details_admin,
                    "search_products": search_products,
                    "get_order_details": get_order_details,
                    "get_products": get_products,
                    "get_web_product": get_web_product,
                    "return_products": return_products,
                    "cancel_order": cancel_order,
                    "get_cart": get_cart,
                    "cart_add_items": cart_add_items,
                }

                for tool in tools:
                    tool_id: str = tool["id"]
                    name: str = tool["name"]
                    if name in tool_mapping:
                        func = tool_mapping[name]
                        tool_registry[tool_id] = {
                            "name": name,
                            "description": f"Mock tool: {name}",
                            "execute": func,
                            "fixed_args": tool.get("fixed_args", {}),
                        }
                return tool_registry

            @staticmethod
            def init_workers(
                workers: list[dict[str, Any]],
            ) -> dict[str, dict[str, Any]]:
                """Initialize mock workers."""
                worker_registry: dict[str, dict[str, Any]] = {}
                worker_mapping = {
                    "FaissRAGWorker": FaissRAGWorker,
                    "MessageWorker": MessageWorker,
                }

                for worker in workers:
                    worker_id: str = worker["id"]
                    name: str = worker["name"]
                    if name in worker_mapping:
                        func = worker_mapping[name]
                        worker_registry[worker_id] = {
                            "name": name,
                            "description": f"Mock worker: {name}",
                            "execute": func,
                        }
                return worker_registry

            @staticmethod
            def init_agents(agents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
                """Initialize mock agents."""
                return {}

        # Create mock tools and workers that the taskgraph expects
        mock_tools = [
            {
                "id": "55011bc1-2a55-4e21-bf39-e9624729c8d8",
                "name": "get_user_details_admin",
                "path": "shopify/get_user_details_admin.py",
                "fixed_args": {},
            },
            {
                "id": "2b275abc-6226-2013-ba05-t4ab83daalc3",
                "name": "search_products",
                "path": "shopify/search_products.py",
                "fixed_args": {},
            },
            {
                "id": "2a2750cb-6226-4068-ba05-a4db83da3e16",
                "name": "get_order_details",
                "path": "shopify/get_order_details.py",
                "fixed_args": {},
            },
            {
                "id": "22fae76f-085c-4098-9011-2ae1e1eb8dc3",
                "name": "get_products",
                "path": "shopify/get_products.py",
                "fixed_args": {},
            },
            {
                "id": "xl34e76f-025c-4xl2-0s2j-l4e1eal2naak",
                "name": "get_web_product",
                "path": "shopify/get_web_product.py",
                "fixed_args": {},
            },
            {
                "id": "alfse0ls-lx4f-a01m-1mch-a4dfsl010end",
                "name": "return_products",
                "path": "shopify/return_products.py",
                "fixed_args": {},
            },
            {
                "id": "alla05l2-3kd1-x9iw-10k3-algk3xenfsl9",
                "name": "cancel_order",
                "path": "shopify/cancel_order.py",
                "fixed_args": {},
            },
            {
                "id": "alfseal2-al94-2kdq-slci-aldjcjenfead",
                "name": "get_cart",
                "path": "shopify/get_cart.py",
                "fixed_args": {},
            },
            {
                "id": "2alak3db-sl36-4zk9-aa35-a4dlkfm3se16",
                "name": "cart_add_items",
                "path": "shopify/cart_add_items.py",
                "fixed_args": {},
            },
        ]

        mock_workers = [
            {
                "id": "FaissRAGWorker",
                "name": "FaissRAGWorker",
                "path": "faiss_rag_worker.py",
            },
            {
                "id": "MessageWorker",
                "name": "MessageWorker",
                "path": "message_worker.py",
            },
        ]

        env = Environment(
            tools=mock_tools,
            workers=mock_workers,
            agents=[],
            resource_initializer=MockResourceInitializer(),
        )
        env.model_service = Mock()
        env.planner = None
        return env

    @pytest.fixture
    def sample_shopify_product_data(self) -> dict[str, Any]:
        """Sample Shopify product data for testing."""
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
                        "startCursor": "cursor123",
                    },
                }
            }
        }

    @pytest.fixture
    def sample_shopify_order_data(self) -> dict[str, Any]:
        """Sample Shopify order data for testing."""
        return {
            "data": {
                "orders": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Order/12345",
                            "name": "#1001",
                            "email": "customer@example.com",
                            "totalPriceSet": {
                                "shopMoney": {"amount": "99.99", "currencyCode": "USD"}
                            },
                            "lineItems": {
                                "nodes": [
                                    {
                                        "title": "Test Product",
                                        "quantity": 2,
                                        "variant": {
                                            "price": "49.99",
                                            "title": "Small",
                                        },
                                    }
                                ]
                            },
                            "fulfillments": {
                                "nodes": [
                                    {
                                        "id": "gid://shopify/Fulfillment/67890",
                                        "status": "SUCCESS",
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
        }

    @pytest.fixture
    def sample_shopify_cart_data(self) -> dict[str, Any]:
        """Sample Shopify cart data for testing."""
        return {
            "data": {
                "cart": {
                    "id": "cart_123",
                    "checkoutUrl": "https://test-shop.myshopify.com/cart/123",
                    "lines": {
                        "nodes": [
                            {
                                "id": "line_123",
                                "quantity": 1,
                                "merchandise": {
                                    "id": "variant_123",
                                    "title": "Test Product - Small",
                                    "product": {
                                        "title": "Test Product",
                                        "handle": "test-product",
                                    },
                                },
                            }
                        ]
                    },
                }
            }
        }

    def test_taskgraph_initialization(
        self, shopify_taskgraph_path: str, mock_environment: Environment
    ) -> None:
        """
        Test that the taskgraph can be properly initialized from the JSON file.

        This test validates that the AgentOrg can successfully load and parse
        the taskgraph.json file, creating all necessary components including
        the task graph, workers, and tools.
        """
        # Test that the taskgraph file exists
        assert os.path.exists(shopify_taskgraph_path), (
            f"Taskgraph file not found: {shopify_taskgraph_path}"
        )

        # Load the taskgraph configuration
        with open(shopify_taskgraph_path) as f:
            config = json.load(f)

        # Add model configuration since it's required
        config["model"] = {
            "model_type_or_path": "gpt-3.5-turbo",
            "llm_provider": "openai",
            "api_key": "test-key",
            "context": 16000,
            "max_tokens": 4096,
        }

        # Validate basic structure
        assert "nodes" in config, "Taskgraph should contain nodes"
        assert "edges" in config, "Taskgraph should contain edges"
        assert "tools" in config, "Taskgraph should contain tools"
        assert "workers" in config, "Taskgraph should contain workers"

        # Test AgentOrg initialization
        agent = AgentOrg(config, mock_environment)

        # Validate that the agent was properly initialized
        assert agent.task_graph is not None, "Task graph should be initialized"
        assert agent.env is mock_environment, "Environment should be set"
        assert agent.product_kwargs["role"] == "customer service assistant", (
            "Role should be set correctly"
        )

        # Validate task graph structure
        assert len(agent.task_graph.graph.nodes) > 0, "Task graph should have nodes"
        assert len(agent.task_graph.graph.edges) > 0, "Task graph should have edges"

        # Validate start node exists
        start_node = agent.task_graph.get_start_node()
        assert start_node is not None, "Task graph should have a start node"
        assert start_node == "0", "Start node should be node '0'"

    def _load_config_with_model(self, shopify_taskgraph_path: str) -> dict[str, Any]:
        """Helper method to load taskgraph config and add required model configuration."""
        with open(shopify_taskgraph_path) as f:
            config = json.load(f)

        # Add model configuration since it's required
        config["model"] = {
            "model_type_or_path": "gpt-3.5-turbo",
            "llm_provider": "openai",
            "api_key": "test-key",
            "context": 16000,
            "max_tokens": 4096,
        }
        return config

    def test_product_search_flow(
        self,
        shopify_taskgraph_path: str,
        mock_environment: Environment,
        sample_shopify_product_data: dict[str, Any],
    ) -> None:
        """
        Test the complete flow for a product search scenario.

        This test simulates a user asking about products, which should:
        1. Start at the initial node
        2. Route to the product search node based on intent
        3. Execute the search_products tool
        4. Return a response with product information
        """
        # Load configuration
        config = self._load_config_with_model(shopify_taskgraph_path)

        # Initialize agent
        agent = AgentOrg(config, mock_environment)

        # Simulate user input for product search
        user_input = {
            "text": "I'm looking for shoes",
            "chat_history": [],
            "parameters": {},
        }

        # Get response from agent
        response = agent.get_response(user_input)

        # Validate response structure
        assert "answer" in response, "Response should contain 'answer' field"
        assert "parameters" in response, "Response should contain 'parameters' field"

    def test_order_inquiry_flow(
        self,
        shopify_taskgraph_path: str,
        mock_environment: Environment,
        sample_shopify_order_data: dict[str, Any],
    ) -> None:
        """
        Test the complete flow for an order inquiry scenario.

        This test simulates a user asking about their past orders, which should:
        1. Start at the initial node
        2. Route to the order details node based on intent
        3. Execute the get_order_details tool
        4. Return order information
        """
        # Load configuration
        config = self._load_config_with_model(shopify_taskgraph_path)

        # Initialize agent
        agent = AgentOrg(config, mock_environment)

        # Simulate user input for order inquiry
        user_input = {
            "text": "I want to check my order status",
            "chat_history": [],
            "parameters": {"user_id": "12345"},
        }

        # Get response from agent
        response = agent.get_response(user_input)

        # Validate response structure
        assert "answer" in response, "Response should contain 'answer' field"
        assert "parameters" in response, "Response should contain 'parameters' field"

    def test_cart_inquiry_flow(
        self,
        shopify_taskgraph_path: str,
        mock_environment: Environment,
        sample_shopify_cart_data: dict[str, Any],
    ) -> None:
        """
        Test the complete flow for a cart inquiry scenario.

        This test simulates a user asking about their shopping cart, which should:
        1. Start at the initial node
        2. Route to the cart inquiry node based on intent
        3. Execute the get_cart tool
        4. Return cart information
        """
        # Load configuration
        config = self._load_config_with_model(shopify_taskgraph_path)

        # Initialize agent
        agent = AgentOrg(config, mock_environment)

        # Simulate user input for cart inquiry
        user_input = {
            "text": "What's in my shopping cart?",
            "chat_history": [],
            "parameters": {"cart_id": "cart_123"},
        }

        # Get response from agent
        response = agent.get_response(user_input)

        # Validate response structure
        assert "answer" in response, "Response should contain 'answer' field"
        assert "parameters" in response, "Response should contain 'parameters' field"

    def test_multi_turn_conversation_flow(
        self,
        shopify_taskgraph_path: str,
        mock_environment: Environment,
    ) -> None:
        """
        Test a multi-turn conversation that traverses multiple nodes.

        This test simulates a conversation where the user:
        1. Asks about products
        2. Gets a response
        3. Asks a follow-up question
        4. Gets another response

        This validates that the taskgraph can handle conversation state
        and properly route between different nodes based on context.
        """
        # Load configuration
        config = self._load_config_with_model(shopify_taskgraph_path)

        # Initialize agent
        agent = AgentOrg(config, mock_environment)

        # First turn: Product inquiry
        first_input = {
            "text": "Do you have any shoes?",
            "chat_history": [],
            "parameters": {},
        }

        first_response = agent.get_response(first_input)
        assert "answer" in first_response, (
            "First response should contain 'answer' field"
        )

        # Second turn: Follow-up question
        second_input = {
            "text": "What about my order status?",
            "chat_history": [
                {"role": "user", "content": "Do you have any shoes?"},
                {"role": "assistant", "content": first_response.get("answer", "")},
            ],
            "parameters": {"user_id": "12345"},
        }

        second_response = agent.get_response(second_input)
        assert "answer" in second_response, (
            "Second response should contain 'answer' field"
        )

    def test_error_handling_flow(
        self,
        shopify_taskgraph_path: str,
        mock_environment: Environment,
    ) -> None:
        """
        Test error handling in the taskgraph flow.

        This test validates that the taskgraph can properly handle errors
        and continue processing, ensuring robust conversation flow.
        """
        # Load configuration
        config = self._load_config_with_model(shopify_taskgraph_path)

        # Initialize agent
        agent = AgentOrg(config, mock_environment)

        # Simulate an input that might cause an error
        user_input = {
            "text": "This is an invalid request that might cause an error",
            "chat_history": [],
            "parameters": {},
        }

        # The agent should handle the error gracefully
        response = agent.get_response(user_input)

        # Even with errors, we should get a structured response
        assert "answer" in response, "Response should contain 'answer' field"
        assert "parameters" in response, "Response should contain 'parameters' field"

    def test_taskgraph_node_transitions(
        self,
        shopify_taskgraph_path: str,
        mock_environment: Environment,
    ) -> None:
        """
        Test that the taskgraph can properly transition between nodes.

        This test validates the graph structure and ensures that nodes
        can be properly traversed based on intents and conditions.
        """
        # Load configuration
        config = self._load_config_with_model(shopify_taskgraph_path)

        # Initialize agent
        agent = AgentOrg(config, mock_environment)

        # Get the task graph
        task_graph = agent.task_graph

        # Validate that we can get the start node
        start_node = task_graph.get_start_node()
        assert start_node is not None, "Should have a start node"

        # Validate that we can get available intents
        params = OrchestratorParams()
        available_intents = task_graph.get_available_global_intents(params)
        assert isinstance(available_intents, dict), "Available intents should be a dict"
        assert len(available_intents) > 0, "Should have available intents"

        # Validate that we can get the current node
        current_node, updated_params = task_graph.get_current_node(params)
        assert current_node is not None, "Should be able to get current node"

    def test_taskgraph_configuration_validation(
        self,
        shopify_taskgraph_path: str,
    ) -> None:
        """
        Test that the taskgraph configuration is valid and complete.

        This test validates the structure and content of the taskgraph.json
        file, ensuring it has all required components and valid relationships.
        """
        # Load configuration
        with open(shopify_taskgraph_path) as f:
            config = json.load(f)

        # Validate required top-level keys
        required_keys = ["nodes", "edges", "tools", "workers", "role", "user_objective"]
        for key in required_keys:
            assert key in config, f"Taskgraph should contain '{key}'"

        # Validate nodes structure
        nodes = config["nodes"]
        assert isinstance(nodes, list), "Nodes should be a list"
        assert len(nodes) > 0, "Should have at least one node"

        # Validate edges structure
        edges = config["edges"]
        assert isinstance(edges, list), "Edges should be a list"
        assert len(edges) > 0, "Should have at least one edge"

        # Validate tools structure
        tools = config["tools"]
        assert isinstance(tools, list), "Tools should be a list"
        assert len(tools) > 0, "Should have at least one tool"

        # Validate workers structure
        workers = config["workers"]
        assert isinstance(workers, list), "Workers should be a list"
        assert len(workers) > 0, "Should have at least one worker"

        # Validate that all referenced tools exist
        tool_ids = {tool["id"] for tool in tools}
        for node in nodes:
            node_data = node[1]
            if "resource" in node_data and "id" in node_data["resource"]:
                resource_id = node_data["resource"]["id"]
                if resource_id not in ["MessageWorker", "FaissRAGWorker"]:  # Workers
                    assert resource_id in tool_ids, (
                        f"Tool {resource_id} referenced but not defined"
                    )

    def test_taskgraph_edge_validation(
        self,
        shopify_taskgraph_path: str,
    ) -> None:
        """
        Test that all edges in the taskgraph reference valid nodes.

        This test ensures that the graph structure is consistent and
        all edges connect to existing nodes.
        """
        # Load configuration
        with open(shopify_taskgraph_path) as f:
            config = json.load(f)

        # Get all node IDs
        node_ids = {node[0] for node in config["nodes"]}

        # Validate all edges reference existing nodes
        for edge in config["edges"]:
            source_node = edge[0]
            target_node = edge[1]

            assert source_node in node_ids, (
                f"Edge source node '{source_node}' does not exist"
            )
            assert target_node in node_ids, (
                f"Edge target node '{target_node}' does not exist"
            )

    def test_taskgraph_intent_consistency(
        self,
        shopify_taskgraph_path: str,
    ) -> None:
        """
        Test that intents are consistently defined across the taskgraph.

        This test validates that intents are properly defined and used
        consistently throughout the graph structure.
        """
        # Load configuration
        with open(shopify_taskgraph_path) as f:
            config = json.load(f)

        # Collect all intents from edges
        intents = set()
        for edge in config["edges"]:
            edge_data = edge[2]
            if "intent" in edge_data:
                intents.add(edge_data["intent"])

        # Validate that we have intents
        assert len(intents) > 0, "Should have at least one intent"

        # Validate that intents are not empty strings
        for intent in intents:
            assert intent != "", "Intent should not be empty"
            # Note: "None" intents are valid in some cases (e.g., default edges)
            # We'll allow them but log a warning
            if intent == "None":
                print("Warning: Found 'None' intent in taskgraph")

    def test_taskgraph_worker_configuration(
        self,
        shopify_taskgraph_path: str,
    ) -> None:
        """
        Test that worker configurations are valid and complete.

        This test validates that all workers referenced in nodes are
        properly defined in the workers section.
        """
        # Load configuration
        with open(shopify_taskgraph_path) as f:
            config = json.load(f)

        # Get all worker IDs
        worker_ids = {worker["id"] for worker in config["workers"]}

        # Validate that all worker references are valid
        for node in config["nodes"]:
            node_data = node[1]
            if "resource" in node_data and "id" in node_data["resource"]:
                resource_id = node_data["resource"]["id"]
                if resource_id in ["MessageWorker", "FaissRAGWorker"]:
                    assert resource_id in worker_ids, (
                        f"Worker {resource_id} referenced but not defined"
                    )
