"""Core test utilities for the Arklex framework.

This module provides core test utilities for the Arklex framework, including
the MockOrchestrator base class for testing orchestrator behavior. It includes
methods for initializing tests, executing conversations, and validating results,
with support for custom validation through abstract methods.
"""

import contextlib
import functools
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Any
from unittest.mock import patch

from arklex.orchestrator.entities.msg_state_entities import StatusEnum


class MockMessageState:
    """Mock MessageState for testing purposes.

    This class provides a mock implementation of MessageState that can be used in tests.
    It simulates the behavior of a real MessageState object.
    """

    def __init__(self, **kwargs: dict[str, object]) -> None:
        """Initialize the mock message state.

        Args:
            **kwargs: Any attributes to set on the message state
        """
        self.sys_instruct = kwargs.get("sys_instruct", "")
        self.bot_config = kwargs.get("bot_config")
        self.user_message = kwargs.get("user_message")
        self.orchestrator_message = kwargs.get("orchestrator_message")
        self.function_calling_trajectory = kwargs.get("function_calling_trajectory", [])
        self.trajectory = kwargs.get("trajectory", [])
        self.message_flow = kwargs.get("message_flow", "Mock message flow")
        self.response = kwargs.get("response", "Mock response")
        self.status = kwargs.get("status", StatusEnum.COMPLETE)
        self.slots = kwargs.get("slots", {})
        self.metadata = kwargs.get("metadata")
        self.is_stream = kwargs.get("is_stream", False)
        self.stream_type = kwargs.get("stream_type")
        self.message_queue = kwargs.get("message_queue")
        self.relevant_records = kwargs.get("relevant_records")
        # Set any additional attributes
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class MockTool:
    """Mock tool for testing purposes.

    This class provides a mock implementation of a tool that can be used in tests.
    It simulates tool execution by returning predefined responses.
    """

    def __init__(self, name: str, description: str) -> None:
        """Initialize the mock tool.

        Args:
            name (str): Name of the tool
            description (str): Description of the tool's functionality
        """
        self.name = name
        self.description = description
        self.info = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        self.output = []
        self.fixed_args = {}
        # Add dummy attributes for compatibility with test expectations
        self.response = self  # allow .response access
        self.function_calling_trajectory = []

    def execute(
        self, message_state: object, **kwargs: dict[str, object]
    ) -> MockMessageState:
        """Return a mock MessageState to simulate tool execution.

        Args:
            message_state: The input message state (ignored in mock)
            **kwargs: Any keyword arguments (ignored in mock)

        Returns:
            MockMessageState: A mock message state with required attributes
        """
        return MockMessageState(
            slots={},
            status=StatusEnum.COMPLETE,
            function_calling_trajectory=[],
            response=f"Mock response from {self.name}",
            message_flow=f"Mock message flow from {self.name}",
        )

    # Add dict-style access for compatibility with production code
    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def __setitem__(self, key: str, value: object) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        return getattr(self, key, default)

    def init_slotfiller(self, slotfillapi: str) -> None:
        """Initialize slot filler for the mock tool.

        Args:
            slotfillapi: Slot filler API endpoint (ignored in mock)
        """
        # Mock implementation - do nothing
        pass

    def __repr__(self) -> str:
        return f"MockTool(name={self.name!r}, description={self.description!r})"


class MockResourceInitializer:
    """Mock resource initializer for testing purposes.

    This class provides a mock implementation of resource initialization that
    creates mock tools and workers for testing.
    """

    @staticmethod
    def init_tools(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize mock tools from configuration.

        Args:
            tools (List[Dict[str, Any]]): List of tool configurations

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping tool IDs to their configurations
        """
        print("\n=== Debug: MockResourceInitializer.init_tools ===")
        print(f"Input tools: {json.dumps(tools, indent=2)}")

        tools_map = {}
        if not tools:
            print("No tools provided, creating dummy tool")

            # Create a function that returns a MockTool instance
            def dummy_tool_func() -> MockTool:
                return MockTool("dummy_tool", "A dummy tool for testing.")

            tools_map["dummy_tool"] = {
                "name": "dummy_tool",
                "description": "A dummy tool for testing.",
                "execute": dummy_tool_func,
                "fixed_args": {},
            }
            print(f"Created dummy tool map: {list(tools_map.keys())}")
            return tools_map

        for tool in tools:
            name = tool.get("name", tool.get("id", "unnamed_tool"))
            description = tool.get("description", "No description provided.")
            print(f"\nProcessing tool: {name}")
            print(f"Tool config: {json.dumps(tool, indent=2)}")

            # Create a function that returns a MockTool instance
            def create_tool_func(
                tool_name: str, tool_description: str
            ) -> Callable[[], MockTool]:
                return functools.partial(MockTool, tool_name, tool_description)

            tool_func = create_tool_func(name, description)
            tool_id = tool.get("id", name)
            tools_map[tool_id] = {
                "name": name,
                "description": description,
                "execute": tool_func,
                "fixed_args": tool.get("fixed_args", {}),
            }
            print(f"Added tool to map with ID: {tool_id}")
            print(f"Tool entry: {tools_map[tool_id]}")

        print("\nFinal tools map:")
        print(list(tools_map.keys()))
        return tools_map

    @staticmethod
    def init_workers(workers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize mock workers from configuration.

        Args:
            workers (List[Dict[str, Any]]): List of worker configurations

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping worker IDs to their configurations
        """
        print("\n=== Debug: MockResourceInitializer.init_workers ===")
        print(f"Input workers: {json.dumps(workers, indent=2)}")

        workers_map = {}
        if not workers:
            print("No workers provided, creating dummy worker")

            # Create a function that returns a MockTool instance (workers are also mocked as tools)
            def dummy_worker_func() -> MockTool:
                return MockTool("dummy_worker", "A dummy worker for testing.")

            workers_map["dummy_worker"] = {
                "name": "dummy_worker",
                "description": "A dummy worker for testing.",
                "execute": dummy_worker_func,
                "fixed_args": {},
            }
            print(f"Created dummy worker map: {list(workers_map.keys())}")
            return workers_map

        for worker in workers:
            name = worker.get("name", worker.get("id", "unnamed_worker"))
            description = worker.get("description", "No description provided.")
            print(f"\nProcessing worker: {name}")
            print(f"Worker config: {json.dumps(worker, indent=2)}")

            # Create a function that returns a MockTool instance (workers are also mocked as tools)
            def create_worker_func(
                worker_name: str, worker_description: str
            ) -> Callable[[], MockTool]:
                return functools.partial(MockTool, worker_name, worker_description)

            worker_func = create_worker_func(name, description)
            worker_id = worker.get("id", name)
            workers_map[worker_id] = {
                "name": name,
                "description": description,
                "execute": worker_func,
                "fixed_args": worker.get("fixed_args", {}),
            }
            print(f"Added worker to map with ID: {worker_id}")
            print(f"Worker entry: {workers_map[worker_id]}")

        print("\nFinal workers map:")
        print(list(workers_map.keys()))
        return workers_map

    @staticmethod
    def init_agents(agents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize mock agents from configuration.
        Args:
            agents (List[Dict[str, Any]]): List of agent configurations
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping agent IDs to their configurations
        """
        print("\n=== Debug: MockResourceInitializer.init_agents ===")
        print(f"Input agents: {json.dumps(agents, indent=2)}")

        agents_map = {}
        if not agents:
            print("No agents provided, returning empty map")
            return agents_map

        for agent in agents:
            name = agent.get("name", agent.get("id", "unnamed_agent"))
            description = agent.get("description", "No description provided.")
            print(f"\nProcessing agent: {name}")
            print(f"Agent config: {json.dumps(agent, indent=2)}")

            mock_tool = MockTool(name, description)
            agent_id = agent.get("id", name)
            agents_map[agent_id] = mock_tool
            print(f"Added agent to map with ID: {agent_id}")
            print(f"Agent entry: {json.dumps(agents_map[agent_id].__dict__, indent=2)}")

        print("\nFinal agents map:")
        print(json.dumps({k: v.__dict__ for k, v in agents_map.items()}, indent=2))
        return agents_map


@contextlib.contextmanager
def mock_llm_invoke() -> Generator[None, None, None]:
    """Context manager to mock LLM invoke calls.

    This context manager patches the LLM invoke method to return dummy responses
    instead of making real API calls. This is useful for testing without
    incurring API costs or requiring network connectivity.

    Yields:
        None: The context manager yields nothing
    """

    # Create a dummy AI message class
    class DummyAIMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    # Helper function to extract the last user message from args/kwargs
    def get_last_user_message(args: tuple, kwargs: dict) -> str:
        """Extract the last user message from the arguments."""
        # Look for messages in args
        for arg in args:
            if isinstance(arg, list):
                for msg in arg:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        return msg.get("content", "")
        # Look for messages in kwargs
        messages = kwargs.get("messages", [])
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def dummy_invoke(*args: object, **kwargs: object) -> str:
        """Dummy invoke function that returns a mock response."""
        user_message = get_last_user_message(args, kwargs)
        if "planning steps" in str(user_message).lower():
            return "1) others"
        return "Mock response"

    async def dummy_ainvoke(*args: object, **kwargs: object) -> str:
        """Dummy async invoke function that returns a mock response."""
        return dummy_invoke(*args, **kwargs)

    def dummy_embed_documents(
        self: object, texts: list[str], *args: object, **kwargs: object
    ) -> list[list[float]]:
        # Return a list of fake embedding vectors (e.g., all zeros)
        return [[0.0] * 1536 for _ in texts]

    def dummy_embed_query(
        self: object, text: str, *args: object, **kwargs: object
    ) -> list[float]:
        # Return a fake embedding vector
        return [0.0] * 1536

    # Patch the LLM invoke method
    with (
        patch(
            "langchain_core.language_models.chat_models.BaseChatModel.invoke",
            dummy_invoke,
        ),
        patch(
            "langchain_core.language_models.chat_models.BaseChatModel.ainvoke",
            dummy_ainvoke,
        ),
        patch(
            "langchain_community.embeddings.base.Embeddings.embed_documents",
            dummy_embed_documents,
        ),
        patch(
            "langchain_community.embeddings.base.Embeddings.embed_query",
            dummy_embed_query,
        ),
    ):
        yield


class MockOrchestrator(ABC):
    """Abstract base class for mock orchestrators in tests.

    This class provides a base implementation for creating mock orchestrators
    that can be used in tests. It includes methods for initializing tests,
    executing conversations, and validating results.
    """

    def __init__(
        self, config_file_path: str, fixed_args: dict[str, Any] | None = None
    ) -> None:
        """Initialize the mock orchestrator.

        Args:
            config_file_path (str): Path to the configuration file
            fixed_args (Optional[Dict[str, Any]]): Fixed arguments to use in tests
        """
        self.config_file_path = config_file_path
        self.fixed_args = fixed_args or {}
        self.history: list[dict[str, str]] = []
        self.params: dict[str, Any] = {}

    def _get_test_response(
        self,
        user_text: str,
        history: list[dict[str, str]],
        params: dict[str, Any],
        test_case: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Get a test response for the given input.

        This method should be implemented by subclasses to provide
        appropriate test responses based on the input.

        Args:
            user_text (str): The user's input text
            history (List[Dict[str, str]]): The conversation history
            params (Dict[str, Any]): Additional parameters
            test_case (Optional[Dict[str, Any]]): The test case being executed

        Returns:
            Dict[str, Any]: The test response
        """
        # Default implementation returns a mock response
        return {
            "response": f"Mock response to: {user_text}",
            "status": "complete",
            "slots": {},
        }

    @contextlib.contextmanager
    def _patch_imports(self) -> Generator[None, None, None]:
        """Patch imports to use mock implementations.

        This method patches various imports to use mock implementations
        instead of real ones, which is useful for testing without
        making real API calls or requiring external dependencies.
        """

        # Patch TaskGraph to avoid real initialization
        def patched_taskgraph_init(
            self: object,
            name: str,
            product_kwargs: dict[str, Any],
            llm_config: dict[str, Any],
            slotfillapi: str = "",
            model_service: object | None = None,
        ) -> None:
            # Mock implementation - just store the config
            self.name = name
            self.product_kwargs = product_kwargs
            self.llm_config = llm_config
            self.slotfillapi = slotfillapi
            self.model_service = model_service

        # Patch ReactPlanner to avoid real initialization
        def patched_react_planner_init(
            self: object,
            tools_map: dict[str, Any],
            workers_map: dict[str, Any],
            name2id: dict[str, int],
        ) -> None:
            # Call the original init but with mocked LLM
            # Mock the LLM to avoid real API calls
            class MockLLM:
                def __init__(self, *args: object, **kwargs: object) -> None:
                    pass

                def invoke(self, *args: object, **kwargs: object) -> object:
                    # Return a mock response based on the input
                    if "planning steps" in str(args) + str(kwargs):
                        return "1) others"
                    else:
                        return "1) others"

                def ainvoke(self, *args: object, **kwargs: object) -> object:
                    return self.invoke(*args, **kwargs)

            # Store the mocked LLM
            self.llm = MockLLM()

        # Patch Environment to use mocked ReactPlanner
        from arklex.env.env import Environment

        orig_env_init = Environment.__init__

        def patched_env_init(
            self: object,
            tools: list[dict[str, Any]],
            workers: list[dict[str, Any]],
            agents: list[dict[str, Any]] | None = None,
            slotsfillapi: str = "",
            resource_initializer: object | None = None,
            planner_enabled: bool = False,
            model_service: object | None = None,
            **kwargs: str | int | float | bool | None,
        ) -> None:
            # Call the original init
            orig_env_init(
                self,
                tools,
                workers,
                agents,
                slotsfillapi,
                resource_initializer,
                planner_enabled,
                model_service,
                **kwargs,
            )

            # If planner is enabled, create a mock planner
            if planner_enabled:
                # Create a mock planner that doesn't make API calls
                class MockPlanner:
                    def __init__(self, *args: object, **kwargs: object) -> None:
                        pass

                    def set_llm_config_and_build_resource_library(
                        self, llm_config: object
                    ) -> None:
                        # Mock implementation - just store the config
                        self.llm_config = llm_config

                    def execute(self, msg_state: object, msg_history: object) -> object:
                        # Return a mock action
                        return (
                            "mock_action",
                            {"status": "complete", "response": "Mock planner response"},
                        )

                self.planner = MockPlanner()

        # Patch AgentOrg to mock the LLM initialization
        from arklex.orchestrator.orchestrator import AgentOrg

        orig_agentorg_init = AgentOrg.__init__

        def patched_agentorg_init(
            self: object,
            config: str | dict[str, Any],
            env: object | None,
            **kwargs: dict[str, Any],
        ) -> None:
            # Call the original init
            orig_agentorg_init(self, config, env, **kwargs)

            # Mock the LLM to avoid real API calls
            class MockLLM:
                def __init__(self, *args: object, **kwargs: object) -> None:
                    pass

                def invoke(self, *args: object, **kwargs: object) -> object:
                    # Return a mock response based on the input
                    prompt = str(args) + str(kwargs)
                    if "planning steps" in prompt.lower():
                        return type("MockResponse", (), {"content": "1) others"})()
                    elif "extract" in prompt.lower():
                        return type("MockResponse", (), {"content": "extracted_info"})()
                    else:
                        return type("MockResponse", (), {"content": "1) others"})()

                def ainvoke(self, *args: object, **kwargs: object) -> object:
                    return self.invoke(*args, **kwargs)

            # Replace the LLM with our mock
            if hasattr(self, "llm"):
                self.llm = MockLLM()

        # Apply the patches
        with (
            patch(
                "arklex.orchestrator.task_graph.task_graph.TaskGraph.__init__",
                patched_taskgraph_init,
            ),
            patch(
                "arklex.env.planner.react_planner.ReactPlanner.__init__",
                patched_react_planner_init,
            ),
            patch("arklex.env.env.Environment.__init__", patched_env_init),
            patch(
                "arklex.orchestrator.orchestrator.AgentOrg.__init__",
                patched_agentorg_init,
            ),
        ):
            yield

    def _initialize_test(self) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Initialize a test with the mock orchestrator.

        This method sets up the test environment and returns the initial
        history and parameters.

        Returns:
            tuple[List[Dict[str, str]], Dict[str, Any]]: Initial history and parameters
        """
        # Initialize history and parameters
        history: list[dict[str, str]] = []
        params: dict[str, Any] = {}

        # Apply patches for this test
        with self._patch_imports():
            # Initialize the orchestrator (this will use mocked components)
            # The actual initialization depends on the specific orchestrator type
            pass

        return history, params

    def _execute_conversation(
        self,
        test_case: dict[str, Any],
        history: list[dict[str, str]],
        params: dict[str, Any],
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Execute a conversation based on the test case.

        This method simulates a conversation by processing the test case
        and updating the history and parameters accordingly.

        Args:
            test_case (Dict[str, Any]): The test case to execute
            history (List[Dict[str, str]]): The current conversation history
            params (Dict[str, Any]): The current parameters

        Returns:
            tuple[List[Dict[str, str]], Dict[str, Any]]: Updated history and parameters
        """
        # Handle initial assistant message if present in expected_conversation
        if "expected_conversation" in test_case:
            expected_conversation = test_case["expected_conversation"]
            if (
                expected_conversation
                and expected_conversation[0]["role"] == "assistant"
            ):
                history.append(expected_conversation[0])

        # Extract user input from test case - handle both "input" and "user_utterance"
        user_input = test_case.get("input", "")
        if not user_input:
            # Try user_utterance field
            user_utterances = test_case.get("user_utterance", [])
            if user_utterances:
                user_input = (
                    user_utterances[0]
                    if isinstance(user_utterances, list)
                    else user_utterances
                )

        if not user_input:
            return history, params

        # Add user message to history
        history.append({"role": "user", "content": user_input})

        # Get response from mock orchestrator
        response = self._get_test_response(user_input, history, params, test_case)

        # Add assistant response to history
        assistant_response = response.get("response", "No response")
        history.append({"role": "assistant", "content": assistant_response})

        # Update parameters if provided in response
        if "slots" in response:
            params.update(response["slots"])

        return history, params

    @abstractmethod
    def _validate_result(
        self,
        test_case: dict[str, Any],
        history: list[dict[str, str]],
        params: dict[str, Any],
    ) -> None:
        """Validate the result of a test case.

        This method should be implemented by subclasses to validate
        the results of test cases according to their specific requirements.

        Args:
            test_case (Dict[str, Any]): The test case that was executed
            history (List[Dict[str, str]]): The conversation history
            params (Dict[str, Any]): The final parameters

        Raises:
            AssertionError: If the validation fails
        """
        pass

    def run_single_test(self, test_case: dict[str, Any]) -> None:
        """Run a single test case.

        This method executes a single test case by initializing the test,
        executing the conversation, and validating the results.

        Args:
            test_case (Dict[str, Any]): The test case to run

        Raises:
            AssertionError: If the test fails validation
        """
        # Initialize the test
        history, params = self._initialize_test()

        # Execute the conversation
        history, params = self._execute_conversation(test_case, history, params)

        # Validate the result
        self._validate_result(test_case, history, params)
