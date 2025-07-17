from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from arklex.env.env import Environment
from arklex.env.nested_graph.nested_graph import NESTED_GRAPH_ID
from arklex.orchestrator.entities.msg_state_entities import (
    BotConfig,
    MessageState,
    OrchestratorResp,
    StatusEnum,
)
from arklex.orchestrator.entities.orchestrator_params_entities import (
    OrchestratorParams as Params,
)
from arklex.orchestrator.entities.taskgraph_entities import NodeInfo, NodeTypeEnum
from arklex.orchestrator.orchestrator import AgentOrg
from arklex.types import StreamType


class DummyEnv(Environment):
    """Dummy environment for testing purposes."""

    def __init__(self) -> None:
        super().__init__(tools=[], workers=[], agents=[])
        self.model_service = Mock()
        self.planner = None

    def step(
        self,
        resource_id: str,
        message_state: MessageState,
        params: Params,
        node_info: NodeInfo,
    ) -> tuple[MessageState, Params]:
        """Mock step method for testing."""
        # Simulate a successful step
        message_state.response = "Mock response"
        return message_state, params


@pytest.fixture
def basic_config() -> dict[str, Any]:
    """Provide a basic configuration for testing."""
    return {
        "role": "test_role",
        "user_objective": "test objective",
        "builder_objective": "builder obj",
        "intro": "intro text",
        "model": {"llm_provider": "openai", "model_type_or_path": "gpt-3.5"},
        "workers": [
            {"id": "worker1", "name": "MessageWorker", "path": "message_worker"}
        ],
        "tools": [],
        "nodes": [("node1", {"type": "task", "name": "Test Node"})],
        "edges": [("node1", "node1", {"intent": "none", "weight": 1.0})],
    }


@pytest.fixture
def config_with_hitl() -> dict[str, Any]:
    """Provide a configuration with HITL worker for testing."""
    return {
        "role": "test_role",
        "user_objective": "test objective",
        "builder_objective": "builder obj",
        "intro": "intro text",
        "model": {"llm_provider": "openai", "model_type_or_path": "gpt-3.5"},
        "workers": [
            {"id": "worker1", "name": "MessageWorker", "path": "message_worker"},
            {"id": "hitl_worker", "name": "HITLWorkerChatFlag", "path": "hitl_worker"},
        ],
        "tools": [],
        "nodes": [("node1", {"type": "task", "name": "Test Node"})],
        "edges": [("node1", "node1", {"intent": "none", "weight": 1.0})],
    }


@pytest.fixture
def config_with_planner() -> dict[str, Any]:
    """Provide a configuration with planner for testing."""
    return {
        "role": "test_role",
        "user_objective": "test objective",
        "builder_objective": "builder obj",
        "intro": "intro text",
        "model": {"llm_provider": "openai", "model_type_or_path": "gpt-3.5"},
        "workers": [
            {"id": "worker1", "name": "MessageWorker", "path": "message_worker"},
            {"id": "planner", "name": "planner", "path": "planner"},
        ],
        "tools": [],
        "nodes": [("node1", {"type": "task", "name": "Test Node"})],
        "edges": [("node1", "node1", {"intent": "none", "weight": 1.0})],
    }


@pytest.fixture
def mock_node_info() -> Mock:
    """Provide a mock NodeInfo for testing."""
    node_info = Mock(spec=NodeInfo)
    node_info.node_id = "test_node"
    node_info.type = "task"
    node_info.resource_id = "test_resource"
    node_info.resource_name = "test_resource"
    node_info.can_skipped = True
    node_info.is_leaf = False
    node_info.add_flow_stack = False
    node_info.attributes = {"task": "test", "value": "test value", "direct": False}
    return node_info


@pytest.fixture
def mock_direct_node_info() -> Mock:
    """Provide a mock NodeInfo for direct response testing."""
    node_info = Mock(spec=NodeInfo)
    node_info.node_id = "direct_node"
    node_info.type = NodeTypeEnum.MULTIPLE_CHOICE.value
    node_info.resource_id = "direct_resource"
    node_info.resource_name = "direct_resource"
    node_info.can_skipped = False
    node_info.is_leaf = True
    node_info.add_flow_stack = False
    node_info.attributes = {
        "direct": True,
        "value": "Direct response",
        "choice_list": ["option1", "option2"],
    }
    return node_info


@pytest.fixture
def mock_nested_graph_node_info() -> Mock:
    """Provide a mock NodeInfo for nested graph testing."""
    node_info = Mock(spec=NodeInfo)
    node_info.node_id = "nested_node"
    node_info.type = "nested_graph"
    node_info.resource_id = NESTED_GRAPH_ID
    node_info.resource_name = "nested_graph"
    node_info.can_skipped = False
    node_info.is_leaf = False
    node_info.add_flow_stack = False
    node_info.attributes = {"value": "nested_graph_value"}
    return node_info


@pytest.fixture
def mock_llm() -> Mock:
    """Provide a mock LLM for testing."""
    llm = Mock()
    llm.invoke.return_value = Mock(content="yes")
    return llm


@pytest.fixture
def mock_orchestrator_response() -> Mock:
    """Provide a mock OrchestratorResp for testing."""
    resp = Mock(spec=OrchestratorResp)
    resp.model_dump.return_value = {"result": "ok"}
    return resp


class TestAgentOrgInitialization:
    """Test AgentOrg initialization with different configurations."""

    def test_init_with_dict_and_env(self, basic_config: dict[str, Any]) -> None:
        """Test initialization with dictionary config and environment."""
        env = DummyEnv()
        agent = AgentOrg(basic_config, env)
        assert agent.env is env
        assert agent.product_kwargs["role"] == "test_role"
        assert hasattr(agent, "llm")
        assert hasattr(agent, "task_graph")
        assert agent.hitl_proposal_enabled is False
        assert agent.hitl_worker_available is False

    def test_init_with_dict_no_env(self, basic_config: dict[str, Any]) -> None:
        """Test initialization with dictionary config and no environment."""
        agent = AgentOrg(basic_config, None)
        assert isinstance(agent.env, Environment)
        assert agent.product_kwargs["role"] == "test_role"

    def test_init_with_file(
        self, tmp_path: "Path", basic_config: dict[str, Any]
    ) -> None:
        """Test initialization with file path config."""
        import json

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(basic_config, f)
        agent = AgentOrg(str(config_path), None)
        assert agent.product_kwargs["role"] == "test_role"

    def test_init_with_hitl_worker(self, config_with_hitl: dict[str, Any]) -> None:
        """Test initialization with HITL worker."""
        agent = AgentOrg(config_with_hitl, None)
        assert agent.hitl_worker_available is True

    def test_init_with_planner(self, config_with_planner: dict[str, Any]) -> None:
        """Test initialization with planner."""
        env = DummyEnv()
        env.planner = Mock()
        env.planner.set_llm_config_and_build_resource_library = Mock()

        agent = AgentOrg(config_with_planner, env)
        assert hasattr(agent, "llm")
        env.planner.set_llm_config_and_build_resource_library.assert_called_once()


class TestAgentOrgInitParams:
    """Test AgentOrg parameter initialization."""

    def test_init_params_basic(self, basic_config: dict[str, Any]) -> None:
        """Test basic parameter initialization."""
        agent = AgentOrg(basic_config, None)
        inputs = {
            "text": "hello",
            "chat_history": [],
            "parameters": None,
        }
        text, chat_history_str, params, message_state = agent.init_params(inputs)
        assert text == "hello"
        assert "hello" in chat_history_str
        assert isinstance(params, Params)
        assert isinstance(message_state, MessageState)
        assert basic_config["role"] in message_state.sys_instruct

    def test_init_params_with_parameters(self, basic_config: dict[str, Any]) -> None:
        """Test parameter initialization with existing parameters."""
        agent = AgentOrg(basic_config, None)
        inputs = {
            "text": "hi",
            "chat_history": [],
            "parameters": {"metadata": {"turn_id": 5}},
        }
        text, chat_history_str, params, message_state = agent.init_params(inputs)
        assert params.metadata.turn_id == 6

    def test_init_params_edge_cases(self, basic_config: dict[str, Any]) -> None:
        """Test parameter initialization edge cases."""
        agent = AgentOrg(basic_config, None)

        # Empty chat history
        inputs = {"text": "hi", "chat_history": [], "parameters": None}
        text, chat_history_str, params, message_state = agent.init_params(inputs)
        assert text == "hi"

        # Chat history with previous messages
        inputs = {
            "text": "yo",
            "chat_history": [{"role": "user", "content": "prev"}],
            "parameters": None,
        }
        text, chat_history_str, params, message_state = agent.init_params(inputs)
        assert "prev" in chat_history_str and "yo" in chat_history_str

        # Parameters with memory
        inputs = {
            "text": "yo",
            "chat_history": [],
            "parameters": {
                "memory": {
                    "trajectory": [[{"resource": "test", "info": {}}]],
                    "function_calling_trajectory": [
                        {"role": "assistant", "content": "foo"}
                    ],
                }
            },
        }
        text, chat_history_str, params, message_state = agent.init_params(inputs)
        assert hasattr(params.memory, "trajectory")

    def test_init_params_with_existing_function_calling_trajectory(
        self, basic_config: dict[str, Any]
    ) -> None:
        """Test parameter initialization with existing function calling trajectory."""
        agent = AgentOrg(basic_config, None)
        inputs = {
            "text": "test",
            "chat_history": [{"role": "user", "content": "prev"}],
            "parameters": {
                "memory": {
                    "function_calling_trajectory": [
                        {"role": "assistant", "content": "existing"}
                    ]
                }
            },
        }
        text, chat_history_str, params, message_state = agent.init_params(inputs)
        assert len(params.memory.function_calling_trajectory) > 1

    def test_init_params_with_opt_instruct(self, basic_config: dict[str, Any]) -> None:
        """Test parameter initialization with optional instructions."""
        basic_config["opt_instruct"] = "Additional instructions"
        agent = AgentOrg(basic_config, None)
        inputs = {
            "text": "test",
            "chat_history": [],
            "parameters": None,
        }
        text, chat_history_str, params, message_state = agent.init_params(inputs)
        assert "Additional instructions" in message_state.sys_instruct


class TestAgentOrgSkipNode:
    """Test AgentOrg node skipping functionality."""

    def test_check_skip_node_true_false(
        self, basic_config: dict[str, Any], mock_node_info: Mock, mock_llm: Mock
    ) -> None:
        """Test node skipping with different conditions."""
        agent = AgentOrg(basic_config, None)
        agent.llm = mock_llm

        assert agent.check_skip_node(mock_node_info, "history") is True

        mock_node_info.can_skipped = False
        assert agent.check_skip_node(mock_node_info, "history") is False

    def test_check_skip_node_edge_cases(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test node skipping edge cases."""
        agent = AgentOrg(basic_config, None)

        # Node without task attribute
        mock_node_info.attributes = {}
        assert agent.check_skip_node(mock_node_info, "history") is False

        # Node with empty task
        mock_node_info.attributes = {"task": ""}
        assert agent.check_skip_node(mock_node_info, "history") is False

    def test_check_skip_node_llm_exception(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test node skipping when LLM raises exception."""
        agent = AgentOrg(basic_config, None)
        agent.llm = Mock()
        agent.llm.invoke.side_effect = Exception("LLM error")

        assert agent.check_skip_node(mock_node_info, "history") is False

    def test_check_skip_node_llm_response_no(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test node skipping with LLM response 'no'."""
        agent = AgentOrg(basic_config, None)
        agent.llm = Mock()
        agent.llm.invoke.return_value = Mock(content="no")

        assert agent.check_skip_node(mock_node_info, "history") is False

    def test_check_skip_node_llm_response_yes(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test node skipping with LLM response 'yes'."""
        agent = AgentOrg(basic_config, None)
        agent.llm = Mock()
        agent.llm.invoke.return_value = Mock(content="yes")

        assert agent.check_skip_node(mock_node_info, "history") is True


class TestAgentOrgPostProcessNode:
    """Test AgentOrg node post-processing functionality."""

    def test_post_process_node_basic(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        agent = AgentOrg(basic_config, None)
        params = Params()
        params.taskgraph.curr_node = "test_node"
        params.taskgraph.node_limit = {"test_node": 3}
        result = agent.post_process_node(mock_node_info, params)
        assert len(result.taskgraph.path) == 1
        assert result.taskgraph.node_limit["test_node"] == 2

    def test_post_process_node_with_skip_info(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        agent = AgentOrg(basic_config, None)
        params = Params()
        params.taskgraph.curr_node = "test_node"
        result = agent.post_process_node(mock_node_info, params, {"is_skipped": True})
        assert len(result.taskgraph.path) == 1
        path_node = result.taskgraph.path[0]
        assert path_node.is_skipped is True

    def test_post_process_node_without_node_limit(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        agent = AgentOrg(basic_config, None)
        params = Params()
        params.taskgraph.curr_node = "test_node"
        params.taskgraph.node_limit = {}
        result = agent.post_process_node(mock_node_info, params)
        assert len(result.taskgraph.path) == 1


class TestAgentOrgDirectNode:
    """Test AgentOrg direct node handling."""

    def test_handl_direct_node_basic(
        self, basic_config: dict[str, Any], mock_direct_node_info: Mock
    ) -> None:
        agent = AgentOrg(basic_config, None)
        params = Params()
        is_direct, response, params = agent.handl_direct_node(
            mock_direct_node_info, params
        )
        assert is_direct is True
        assert response.answer == "Direct response"
        assert response.choice_list == ["option1", "option2"]

    def test_handl_direct_node_not_direct(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        agent = AgentOrg(basic_config, None)
        params = Params()
        is_direct, response, params = agent.handl_direct_node(mock_node_info, params)
        assert is_direct is False
        assert response is None

    def test_handl_direct_node_empty_value(self, basic_config: dict[str, Any]) -> None:
        """Test direct node with empty value."""
        agent = AgentOrg(basic_config, None)
        params = Params()
        node_info = NodeInfo(attributes={"direct": True, "value": ""})
        is_direct, response, params = agent.handl_direct_node(node_info, params)
        # Empty values (after strip) return False for is_direct
        assert is_direct is False
        assert response is None

    def test_handl_direct_node_multiple_choice_without_choice_list(
        self, basic_config: dict[str, Any]
    ) -> None:
        """Test direct node with multiple choice but no choice list."""
        agent = AgentOrg(basic_config, None)
        params = Params()
        node_info = NodeInfo(
            type=NodeTypeEnum.MULTIPLE_CHOICE.value,
            attributes={"direct": True, "value": "test", "choice_list": []},
        )
        is_direct, response, params = agent.handl_direct_node(node_info, params)
        assert is_direct is True
        assert response.answer == "test"
        assert response.choice_list == []


class TestAgentOrgPerformNode:
    """Test AgentOrg node performance."""

    def test_perform_node_basic(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test basic node performance."""
        agent = AgentOrg(basic_config, None)
        params = Params()
        # Initialize trajectory with at least one empty list
        params.memory.trajectory = [[]]

        message_state = MessageState(
            sys_instruct="test",
            bot_config=BotConfig(
                bot_id="test",
                version="1.0",
                language="EN",
                bot_type="test",
                llm_config=agent.llm_config,
            ),
        )
        node_info, response_state, params = agent.perform_node(
            message_state,
            mock_node_info,
            params,
            "test text",
            "chat history",
            None,
            None,
        )
        # The environment step returns empty response, not "Mock response"
        assert response_state.response == ""
        assert response_state.is_stream is False
        assert response_state.stream_type is None

    def test_perform_node_with_streaming(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test node performance with streaming."""
        agent = AgentOrg(basic_config, None)
        params = Params()
        # Initialize trajectory with at least one empty list
        params.memory.trajectory = [[]]

        message_state = MessageState(
            sys_instruct="test",
            bot_config=BotConfig(
                bot_id="test",
                version="1.0",
                language="EN",
                bot_type="test",
                llm_config=agent.llm_config,
            ),
        )
        # Use a simple mock queue instead of janus.SyncQueue
        message_queue = Mock()
        node_info, response_state, params = agent.perform_node(
            message_state,
            mock_node_info,
            params,
            "test text",
            "chat history",
            StreamType.TEXT,
            message_queue,
        )
        assert response_state.is_stream is True
        assert response_state.stream_type == StreamType.TEXT
        assert response_state.message_queue == message_queue


class TestAgentOrgNestedGraphNode:
    """Test AgentOrg nested graph node handling."""

    @patch("arklex.orchestrator.orchestrator.NestedGraph")
    def test_handle_nested_graph_node_basic(
        self,
        mock_nested_graph_class: Mock,
        basic_config: dict[str, Any],
        mock_nested_graph_node_info: Mock,
    ) -> None:
        agent = AgentOrg(basic_config, None)
        params = Params()
        mock_nested_graph = Mock()
        mock_nested_graph.get_nested_graph_start_node_id.return_value = "start_node"
        mock_nested_graph_class.return_value = mock_nested_graph
        agent.task_graph._get_node = Mock(
            return_value=(mock_nested_graph_node_info, params)
        )
        node_info, params = agent.handle_nested_graph_node(
            mock_nested_graph_node_info, params
        )
        assert len(params.taskgraph.path) == 1
        assert params.taskgraph.curr_node == "start_node"
        assert (
            params.taskgraph.node_status[mock_nested_graph_node_info.node_id]
            == StatusEnum.INCOMPLETE
        )

    def test_handle_nested_graph_node_not_nested(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        agent = AgentOrg(basic_config, None)
        params = Params()
        node_info, params2 = agent.handle_nested_graph_node(mock_node_info, params)
        assert node_info == mock_node_info
        assert params2 == params


class TestAgentOrgGetResponse:
    """Test AgentOrg response generation."""

    def test_get_response_minimal(
        self,
        basic_config: dict[str, Any],
        mock_orchestrator_response: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test minimal response generation."""
        agent = AgentOrg(basic_config, None)
        monkeypatch.setattr(
            agent, "_get_response", lambda *a, **kw: mock_orchestrator_response
        )

        out = agent.get_response({"text": "hi", "chat_history": [], "parameters": None})
        assert out["result"] == "ok"

    @patch("arklex.orchestrator.orchestrator.RunnableLambda")
    def test_get_response_full_flow(
        self,
        mock_runnable_lambda: Mock,
        basic_config: dict[str, Any],
        mock_node_info: Mock,
    ) -> None:
        """Test full response generation flow."""
        agent = AgentOrg(basic_config, None)
        params = Params()

        # Mock the task graph chain
        mock_chain = Mock()
        mock_chain.invoke.return_value = (mock_node_info, params)
        mock_runnable_lambda.return_value.__or__ = Mock(return_value=mock_chain)

        # Mock check_skip_node to return False
        agent.check_skip_node = Mock(return_value=False)

        # Mock handl_direct_node to return False
        agent.handl_direct_node = Mock(return_value=(False, None, params))

        # Mock perform_node
        message_state = MessageState(
            sys_instruct="test",
            bot_config=BotConfig(
                bot_id="test",
                version="1.0",
                language="EN",
                bot_type="test",
                llm_config=agent.llm_config,
            ),
        )
        message_state.response = "Test response"
        agent.perform_node = Mock(return_value=(mock_node_info, message_state, params))

        # Mock post_process_node
        agent.post_process_node = Mock(return_value=params)

        # Mock ToolGenerator.context_generate
        with patch(
            "arklex.orchestrator.orchestrator.ToolGenerator"
        ) as mock_tool_generator:
            mock_tool_generator.context_generate.return_value = message_state

            # Mock post_process_response
            with patch(
                "arklex.orchestrator.orchestrator.post_process_response"
            ) as mock_post_process:
                mock_post_process.return_value = message_state

                response = agent._get_response(
                    {"text": "test", "chat_history": [], "parameters": None}
                )

                assert response.answer == "Test response"

    def test_get_response_with_skip_node(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test response generation with skipped node."""
        agent = AgentOrg(basic_config, None)
        params = Params()

        # Mock the task graph chain to return a leaf node after skipping
        with patch(
            "arklex.orchestrator.orchestrator.RunnableLambda"
        ) as mock_runnable_lambda:
            mock_chain = Mock()

            # Use a function that returns the same result each time
            def mock_invoke(inputs: dict[str, Any]) -> tuple[Mock, Params]:
                return (mock_node_info, params)

            mock_chain.invoke = mock_invoke
            mock_runnable_lambda.return_value.__or__ = Mock(return_value=mock_chain)

            # Mock check_skip_node to return True for first call, False for second
            check_skip_calls = 0

            def mock_check_skip(node_info: Mock, chat_history: str) -> bool:
                nonlocal check_skip_calls
                check_skip_calls += 1
                return (
                    check_skip_calls == 1
                )  # True for first call, False for subsequent

            agent.check_skip_node = mock_check_skip

            # Mock handl_direct_node to return False
            agent.handl_direct_node = Mock(return_value=(False, None, params))

            # Mock perform_node
            message_state = MessageState(
                sys_instruct="test",
                bot_config=BotConfig(
                    bot_id="test",
                    version="1.0",
                    language="EN",
                    bot_type="test",
                    llm_config=agent.llm_config,
                ),
            )
            message_state.response = "Test response"
            agent.perform_node = Mock(
                return_value=(mock_node_info, message_state, params)
            )

            # Mock post_process_node
            agent.post_process_node = Mock(return_value=params)

            # Mock ToolGenerator.context_generate
            with patch(
                "arklex.orchestrator.orchestrator.ToolGenerator"
            ) as mock_tool_generator:
                mock_tool_generator.context_generate.return_value = message_state

                # Mock post_process_response
                with patch(
                    "arklex.orchestrator.orchestrator.post_process_response"
                ) as mock_post_process:
                    mock_post_process.return_value = message_state

                    agent._get_response(
                        {"text": "test", "chat_history": [], "parameters": None}
                    )

                    # Should have called check_skip_node at least twice
                    assert check_skip_calls >= 2
                    # Should have called post_process_node for the skipped node
                    agent.post_process_node.assert_called()

    def test_get_response_with_direct_node(
        self, basic_config: dict[str, Any], mock_direct_node_info: Mock
    ) -> None:
        """Test response generation with direct node."""
        agent = AgentOrg(basic_config, None)
        params = Params()

        # Mock the task graph chain
        with patch(
            "arklex.orchestrator.orchestrator.RunnableLambda"
        ) as mock_runnable_lambda:
            mock_chain = Mock()
            mock_chain.invoke.return_value = (mock_direct_node_info, params)
            mock_runnable_lambda.return_value.__or__ = Mock(return_value=mock_chain)

            # Mock check_skip_node to return False
            agent.check_skip_node = Mock(return_value=False)

            # Mock post_process_node
            agent.post_process_node = Mock(return_value=params)

            response = agent._get_response(
                {"text": "test", "chat_history": [], "parameters": None}
            )

            assert response.answer == "Direct response"

    def test_get_response_with_incomplete_node(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test response generation with incomplete node."""
        agent = AgentOrg(basic_config, None)
        params = Params()
        params.taskgraph.node_status = {"test_node": StatusEnum.INCOMPLETE}
        params.taskgraph.curr_node = "test_node"

        # Mock the task graph chain
        with patch(
            "arklex.orchestrator.orchestrator.RunnableLambda"
        ) as mock_runnable_lambda:
            mock_chain = Mock()
            mock_chain.invoke.return_value = (mock_node_info, params)
            mock_runnable_lambda.return_value.__or__ = Mock(return_value=mock_chain)

            # Mock check_skip_node to return False
            agent.check_skip_node = Mock(return_value=False)

            # Mock handl_direct_node to return False
            agent.handl_direct_node = Mock(return_value=(False, None, params))

            # Mock perform_node
            message_state = MessageState(
                sys_instruct="test",
                bot_config=BotConfig(
                    bot_id="test",
                    version="1.0",
                    language="EN",
                    bot_type="test",
                    llm_config=agent.llm_config,
                ),
            )
            agent.perform_node = Mock(
                return_value=(mock_node_info, message_state, params)
            )

            # Mock post_process_node
            agent.post_process_node = Mock(return_value=params)

            # Mock ToolGenerator.context_generate
            with patch(
                "arklex.orchestrator.orchestrator.ToolGenerator"
            ) as mock_tool_generator:
                mock_tool_generator.context_generate.return_value = message_state

                # Mock post_process_response
                with patch(
                    "arklex.orchestrator.orchestrator.post_process_response"
                ) as mock_post_process:
                    mock_post_process.return_value = message_state

                    response = agent._get_response(
                        {"text": "test", "chat_history": [], "parameters": None}
                    )

                    # Should break loop due to incomplete status
                    assert response is not None

    def test_get_response_with_info_worker(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test response generation with info worker."""
        agent = AgentOrg(basic_config, None)
        params = Params()
        mock_node_info.resource_name = "MessageWorker"
        mock_node_info.is_leaf = False

        # Mock the task graph chain
        with patch(
            "arklex.orchestrator.orchestrator.RunnableLambda"
        ) as mock_runnable_lambda:
            mock_chain = Mock()
            mock_chain.invoke.return_value = (mock_node_info, params)
            mock_runnable_lambda.return_value.__or__ = Mock(return_value=mock_chain)

            # Mock check_skip_node to return False
            agent.check_skip_node = Mock(return_value=False)

            # Mock handl_direct_node to return False
            agent.handl_direct_node = Mock(return_value=(False, None, params))

            # Mock perform_node
            message_state = MessageState(
                sys_instruct="test",
                bot_config=BotConfig(
                    bot_id="test",
                    version="1.0",
                    language="EN",
                    bot_type="test",
                    llm_config=agent.llm_config,
                ),
            )
            agent.perform_node = Mock(
                return_value=(mock_node_info, message_state, params)
            )

            # Mock post_process_node
            agent.post_process_node = Mock(return_value=params)

            # Mock ToolGenerator.context_generate
            with patch(
                "arklex.orchestrator.orchestrator.ToolGenerator"
            ) as mock_tool_generator:
                mock_tool_generator.context_generate.return_value = message_state

                # Mock post_process_response
                with patch(
                    "arklex.orchestrator.orchestrator.post_process_response"
                ) as mock_post_process:
                    mock_post_process.return_value = message_state

                    response = agent._get_response(
                        {"text": "test", "chat_history": [], "parameters": None}
                    )

                    # Should break loop after info worker
                    assert response is not None

    def test_get_response_with_leaf_node(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test response generation with leaf node."""
        agent = AgentOrg(basic_config, None)
        params = Params()
        mock_node_info.is_leaf = True

        # Mock the task graph chain
        with patch(
            "arklex.orchestrator.orchestrator.RunnableLambda"
        ) as mock_runnable_lambda:
            mock_chain = Mock()
            mock_chain.invoke.return_value = (mock_node_info, params)
            mock_runnable_lambda.return_value.__or__ = Mock(return_value=mock_chain)

            # Mock check_skip_node to return False
            agent.check_skip_node = Mock(return_value=False)

            # Mock handl_direct_node to return False
            agent.handl_direct_node = Mock(return_value=(False, None, params))

            # Mock perform_node
            message_state = MessageState(
                sys_instruct="test",
                bot_config=BotConfig(
                    bot_id="test",
                    version="1.0",
                    language="EN",
                    bot_type="test",
                    llm_config=agent.llm_config,
                ),
            )
            agent.perform_node = Mock(
                return_value=(mock_node_info, message_state, params)
            )

            # Mock post_process_node
            agent.post_process_node = Mock(return_value=params)

            # Mock ToolGenerator.context_generate
            with patch(
                "arklex.orchestrator.orchestrator.ToolGenerator"
            ) as mock_tool_generator:
                mock_tool_generator.context_generate.return_value = message_state

                # Mock post_process_response
                with patch(
                    "arklex.orchestrator.orchestrator.post_process_response"
                ) as mock_post_process:
                    mock_post_process.return_value = message_state

                    response = agent._get_response(
                        {"text": "test", "chat_history": [], "parameters": None}
                    )

                    # Should break loop due to leaf node
                    assert response is not None

    def test_get_response_with_streaming(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test response generation with streaming."""
        agent = AgentOrg(basic_config, None)
        params = Params()

        # Mock the task graph chain
        with patch(
            "arklex.orchestrator.orchestrator.RunnableLambda"
        ) as mock_runnable_lambda:
            mock_chain = Mock()
            mock_chain.invoke.return_value = (mock_node_info, params)
            mock_runnable_lambda.return_value.__or__ = Mock(return_value=mock_chain)

            # Mock check_skip_node to return False
            agent.check_skip_node = Mock(return_value=False)

            # Mock handl_direct_node to return False
            agent.handl_direct_node = Mock(return_value=(False, None, params))

            # Mock perform_node
            message_state = MessageState(
                sys_instruct="test",
                bot_config=BotConfig(
                    bot_id="test",
                    version="1.0",
                    language="EN",
                    bot_type="test",
                    llm_config=agent.llm_config,
                ),
            )
            agent.perform_node = Mock(
                return_value=(mock_node_info, message_state, params)
            )

            # Mock post_process_node
            agent.post_process_node = Mock(return_value=params)

            # Mock ToolGenerator.stream_context_generate
            with patch(
                "arklex.orchestrator.orchestrator.ToolGenerator"
            ) as mock_tool_generator:
                mock_tool_generator.stream_context_generate.return_value = message_state

                # Mock post_process_response
                with patch(
                    "arklex.orchestrator.orchestrator.post_process_response"
                ) as mock_post_process:
                    mock_post_process.return_value = message_state

                    agent._get_response(
                        {"text": "test", "chat_history": [], "parameters": None},
                        stream_type=StreamType.TEXT,
                    )

                    # Should use stream_context_generate
                    mock_tool_generator.stream_context_generate.assert_called_once()

    def test_get_response_max_nodes_reached(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test response generation when max nodes reached."""
        agent = AgentOrg(basic_config, None)
        params = Params()

        # Mock the task graph chain to return the same node multiple times
        with patch(
            "arklex.orchestrator.orchestrator.RunnableLambda"
        ) as mock_runnable_lambda:
            mock_chain = Mock()
            mock_chain.invoke.return_value = (mock_node_info, params)
            mock_runnable_lambda.return_value.__or__ = Mock(return_value=mock_chain)

            # Mock check_skip_node to return False
            agent.check_skip_node = Mock(return_value=False)

            # Mock handl_direct_node to return False
            agent.handl_direct_node = Mock(return_value=(False, None, params))

            # Mock perform_node
            message_state = MessageState(
                sys_instruct="test",
                bot_config=BotConfig(
                    bot_id="test",
                    version="1.0",
                    language="EN",
                    bot_type="test",
                    llm_config=agent.llm_config,
                ),
            )
            agent.perform_node = Mock(
                return_value=(mock_node_info, message_state, params)
            )

            # Mock post_process_node
            agent.post_process_node = Mock(return_value=params)

            # Mock ToolGenerator.context_generate
            with patch(
                "arklex.orchestrator.orchestrator.ToolGenerator"
            ) as mock_tool_generator:
                mock_tool_generator.context_generate.return_value = message_state

                # Mock post_process_response
                with patch(
                    "arklex.orchestrator.orchestrator.post_process_response"
                ) as mock_post_process:
                    mock_post_process.return_value = message_state

                    response = agent._get_response(
                        {"text": "test", "chat_history": [], "parameters": None}
                    )

                    # Should break after max_n_node_performed iterations
                    assert response is not None
                    assert agent.perform_node.call_count == 5  # max_n_node_performed


class TestAgentOrgEdgeCases:
    """Test AgentOrg edge cases and error handling."""

    def test_init_with_invalid_config_file(self, tmp_path: "Path") -> None:
        """Test initialization with invalid config file."""
        config_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            AgentOrg(str(config_path), None)

    def test_init_params_with_empty_text(self, basic_config: dict[str, Any]) -> None:
        """Test parameter initialization with empty text."""
        agent = AgentOrg(basic_config, None)
        inputs = {
            "text": "",
            "chat_history": [],
            "parameters": None,
        }
        text, chat_history_str, params, message_state = agent.init_params(inputs)
        assert text == ""

    def test_post_process_node_with_node_limit_decrement(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test node post-processing with node limit decrement."""
        agent = AgentOrg(basic_config, None)
        params = Params()
        params.taskgraph.curr_node = "test_node"
        params.taskgraph.node_limit = {"test_node": 1}

        result = agent.post_process_node(mock_node_info, params)

        assert result.taskgraph.node_limit["test_node"] == 0

    def test_handl_direct_node_with_whitespace_value(
        self, basic_config: dict[str, Any]
    ) -> None:
        """Test direct node with whitespace value."""
        agent = AgentOrg(basic_config, None)
        params = Params()
        node_info = NodeInfo(attributes={"direct": True, "value": "   "})

        is_direct, response, params = agent.handl_direct_node(node_info, params)

        # Whitespace values (after strip) return False for is_direct
        assert is_direct is False
        assert response is None

    def test_perform_node_with_none_stream_type(
        self, basic_config: dict[str, Any], mock_node_info: Mock
    ) -> None:
        """Test node performance with None stream type."""
        agent = AgentOrg(basic_config, None)
        params = Params()
        # Initialize trajectory with at least one empty list
        params.memory.trajectory = [[]]

        message_state = MessageState(
            sys_instruct="test",
            bot_config=BotConfig(
                bot_id="test",
                version="1.0",
                language="EN",
                bot_type="test",
                llm_config=agent.llm_config,
            ),
        )

        node_info, response_state, params = agent.perform_node(
            message_state,
            mock_node_info,
            params,
            "test text",
            "chat history",
            None,
            None,
        )

        assert response_state.is_stream is False
        assert response_state.stream_type is None


class TestAgentOrgImportFallback:
    """Test import fallback scenarios."""

    def test_import_fallback_unpack_type(self, basic_config: dict[str, Any]) -> None:
        """Test import fallback when typing.Unpack is not available."""
        # This test verifies that the import fallback mechanism works
        # The actual import happens at module level, so we just verify the class works
        agent = AgentOrg(basic_config, None)
        assert agent.product_kwargs["role"] == "test_role"

    def test_import_fallback_with_environment(
        self, basic_config: dict[str, Any]
    ) -> None:
        """Test import fallback with environment parameter."""
        env = DummyEnv()
        agent = AgentOrg(basic_config, env)
        assert agent.env is env

    def test_import_fallback_with_kwargs(self, basic_config: dict[str, Any]) -> None:
        """Test import fallback with custom kwargs."""
        agent = AgentOrg(
            basic_config, None, user_prefix="custom_user", worker_prefix="custom_worker"
        )
        assert agent.user_prefix == "custom_user"
        assert agent.worker_prefix == "custom_worker"


class TestAgentOrgAdditionalCoverage:
    """Additional test cases for better coverage."""

    def test_init_with_dict_config_and_planner(
        self, basic_config: dict[str, Any]
    ) -> None:
        """Test initialization with planner enabled."""
        # Add planner to config
        basic_config["workers"].append(
            {"id": "planner", "name": "planner", "path": "planner"}
        )
        env = DummyEnv()
        env.planner = Mock()
        env.planner.set_llm_config_and_build_resource_library = Mock()

        agent = AgentOrg(basic_config, env)
        assert agent.env.planner is not None
        env.planner.set_llm_config_and_build_resource_library.assert_called_once()

    def test_init_with_dict_config_and_no_planner(
        self, basic_config: dict[str, Any]
    ) -> None:
        """Test initialization without planner."""
        env = DummyEnv()
        env.planner = None

        agent = AgentOrg(basic_config, env)
        assert agent.env.planner is None

    def test_init_with_hitl_proposal_enabled(
        self, config_with_hitl: dict[str, Any]
    ) -> None:
        """Test initialization with HITL proposal enabled."""
        config_with_hitl["settings"] = {"hitl_proposal": True}
        agent = AgentOrg(config_with_hitl, None)
        assert agent.hitl_proposal_enabled is True

    def test_init_with_hitl_proposal_disabled(
        self, config_with_hitl: dict[str, Any]
    ) -> None:
        """Test initialization with HITL proposal disabled."""
        config_with_hitl["settings"] = {"hitl_proposal": False}
        agent = AgentOrg(config_with_hitl, None)
        assert agent.hitl_proposal_enabled is False

    def test_init_with_no_hitl_worker(self, basic_config: dict[str, Any]) -> None:
        """Test initialization without HITL worker."""
        agent = AgentOrg(basic_config, None)
        assert agent.hitl_worker_available is False

    def test_init_with_empty_workers_list(self) -> None:
        """Test initialization with empty workers list."""
        config = {
            "role": "test_role",
            "user_objective": "test objective",
            "builder_objective": "builder obj",
            "intro": "intro text",
            "model": {"llm_provider": "openai", "model_type_or_path": "gpt-3.5"},
            "workers": [],
            "tools": [],
            "nodes": [("node1", {"type": "task", "name": "Test Node"})],
            "edges": [("node1", "node1", {"intent": "none", "weight": 1.0})],
        }
        agent = AgentOrg(config, None)
        assert agent.hitl_worker_available is False

    def test_init_with_missing_workers_key(self) -> None:
        """Test initialization with missing workers key."""
        config = {
            "role": "test_role",
            "user_objective": "test objective",
            "builder_objective": "builder obj",
            "intro": "intro text",
            "model": {"llm_provider": "openai", "model_type_or_path": "gpt-3.5"},
            "tools": [],
            "nodes": [("node1", {"type": "task", "name": "Test Node"})],
            "edges": [("node1", "node1", {"intent": "none", "weight": 1.0})],
        }
        agent = AgentOrg(config, None)
        assert agent.hitl_worker_available is False

    def test_init_with_missing_settings_key(
        self, config_with_hitl: dict[str, Any]
    ) -> None:
        """Test initialization with missing settings key."""
        agent = AgentOrg(config_with_hitl, None)
        assert agent.hitl_proposal_enabled is False

    def test_init_with_none_settings(self, config_with_hitl: dict[str, Any]) -> None:
        """Test initialization with None settings."""
        config_with_hitl["settings"] = None
        agent = AgentOrg(config_with_hitl, None)
        assert agent.hitl_proposal_enabled is False


def test_agentorg_env_none_valid_config() -> None:
    """Test AgentOrg with None environment and valid config."""
    config = {
        "role": "test_role",
        "user_objective": "test objective",
        "builder_objective": "builder obj",
        "intro": "intro text",
        "model": {"llm_provider": "openai", "model_type_or_path": "gpt-3.5"},
        "workers": [],
        "tools": [],
        "nodes": [("node1", {"type": "task", "name": "Test Node"})],
        "edges": [("node1", "node1", {"intent": "none", "weight": 1.0})],
    }
    agent = AgentOrg(config, None)
    assert isinstance(agent.env, Environment)
    assert agent.product_kwargs["role"] == "test_role"


def test_agentorg_hitl_proposal_enabled_valid_config() -> None:
    """Test AgentOrg with HITL proposal enabled in valid config."""
    config = {
        "role": "test_role",
        "user_objective": "test objective",
        "builder_objective": "builder obj",
        "intro": "intro text",
        "model": {"llm_provider": "openai", "model_type_or_path": "gpt-3.5"},
        "workers": [
            {"id": "hitl_worker", "name": "HITLWorkerChatFlag", "path": "hitl_worker"}
        ],
        "settings": {"hitl_proposal": True},
        "tools": [],
        "nodes": [("node1", {"type": "task", "name": "Test Node"})],
        "edges": [("node1", "node1", {"intent": "none", "weight": 1.0})],
    }
    agent = AgentOrg(config, None)
    assert agent.hitl_worker_available is True
    assert agent.hitl_proposal_enabled is True
