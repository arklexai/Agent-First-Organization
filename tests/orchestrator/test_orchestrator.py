import pytest
from unittest.mock import Mock, patch
from arklex.orchestrator.orchestrator import AgentOrg
from arklex.env.env import Environment
from arklex.utils.graph_state import NodeInfo, Params, MessageState, BotConfig


class DummyEnv(Environment):
    """Dummy environment for testing purposes."""

    def __init__(self):
        super().__init__(tools=[], workers=[])
        self.model_service = Mock()
        self.planner = None


@pytest.fixture
def basic_config():
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
def mock_node_info():
    """Provide a mock NodeInfo for testing."""
    node_info = Mock(spec=NodeInfo)
    node_info.skippable = True
    node_info.can_skipped = True
    node_info.execution_limit = 1
    node_info.execution_count = 1
    node_info.attributes = {"task": "test"}
    return node_info


@pytest.fixture
def mock_llm():
    """Provide a mock LLM for testing."""
    llm = Mock()
    llm.invoke.side_effect = ["yes", "no"]
    return llm


@pytest.fixture
def mock_orchestrator_response():
    """Provide a mock OrchestratorResp for testing."""
    resp = Mock()
    resp.model_dump.return_value = {"result": "ok"}
    return resp


class TestAgentOrgInitialization:
    """Test AgentOrg initialization with different configurations."""

    def test_init_with_dict_and_env(self, basic_config) -> None:
        """Test initialization with dictionary config and environment."""
        env = DummyEnv()
        agent = AgentOrg(basic_config, env)
        assert agent.env is env
        assert agent.product_kwargs["role"] == "test_role"
        assert hasattr(agent, "llm")
        assert hasattr(agent, "task_graph")
        assert agent.hitl_worker_available is False

    def test_init_with_dict_no_env(self, basic_config) -> None:
        """Test initialization with dictionary config and no environment."""
        agent = AgentOrg(basic_config, None)
        assert isinstance(agent.env, Environment)
        assert agent.product_kwargs["role"] == "test_role"

    def test_init_with_file(self, tmp_path, basic_config) -> None:
        """Test initialization with file path config."""
        import json

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(basic_config, f)
        agent = AgentOrg(str(config_path), None)
        assert agent.product_kwargs["role"] == "test_role"


class TestAgentOrgInitParams:
    """Test AgentOrg parameter initialization."""

    def test_init_params_basic(self, basic_config) -> None:
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

    def test_init_params_with_parameters(self, basic_config) -> None:
        """Test parameter initialization with existing parameters."""
        agent = AgentOrg(basic_config, None)
        inputs = {
            "text": "hi",
            "chat_history": [],
            "parameters": {"metadata": {"turn_id": 5}},
        }
        text, chat_history_str, params, message_state = agent.init_params(inputs)
        assert params.metadata.turn_id == 6

    def test_init_params_edge_cases(self, basic_config) -> None:
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


class TestAgentOrgSkipNode:
    """Test AgentOrg node skipping functionality."""

    def test_check_skip_node_true_false(
        self, basic_config, mock_node_info, mock_llm
    ) -> None:
        """Test node skipping with different conditions."""
        agent = AgentOrg(basic_config, None)
        agent.llm = mock_llm

        assert agent.check_skip_node(mock_node_info, "history") is True

        mock_node_info.execution_count = 0
        assert agent.check_skip_node(mock_node_info, "history") is False

        mock_node_info.skippable = False
        mock_node_info.can_skipped = False
        assert agent.check_skip_node(mock_node_info, "history") is False

    def test_check_skip_node_edge_cases(self, basic_config, mock_node_info) -> None:
        """Test node skipping edge cases."""
        agent = AgentOrg(basic_config, None)

        mock_node_info.execution_limit = 0
        mock_node_info.execution_count = 0
        assert agent.check_skip_node(mock_node_info, "history") is False

        mock_node_info.skippable = None
        mock_node_info.can_skipped = False
        assert agent.check_skip_node(mock_node_info, "history") is False


class TestAgentOrgResponse:
    """Test AgentOrg response handling."""

    def test_get_response_minimal(
        self, basic_config, mock_orchestrator_response, monkeypatch
    ) -> None:
        """Test minimal response generation."""
        agent = AgentOrg(basic_config, None)
        monkeypatch.setattr(
            agent, "_get_response", lambda *a, **kw: mock_orchestrator_response
        )

        out = agent.get_response({"text": "hi", "chat_history": [], "parameters": None})
        assert out["result"] == "ok"
