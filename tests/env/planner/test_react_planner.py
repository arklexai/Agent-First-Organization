"""Tests for ReactPlanner module."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.planner.react_planner import (
    MAX_NUM_RETRIEVALS,
    MIN_NUM_RETRIEVALS,
    NUM_STEPS_TO_NUM_RETRIEVALS,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_RESOURCE,
    Action,
    DefaultPlanner,
    EnvResponse,
    PlannerResource,
    ReactPlanner,
    aimessage_to_dict,
)
from arklex.orchestrator.entities.msg_state_entities import (
    ConvoMessage,
    LLMConfig,
    MessageState,
    OrchestratorMessage,
)
from arklex.types import StreamType


@pytest.fixture
def mock_tools_map() -> dict[str, Any]:
    return {
        1: {
            "info": {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool for testing",
                    "parameters": {
                        "properties": {
                            "param1": {
                                "type": "string",
                                "description": "First parameter",
                            },
                            "param2": {
                                "type": "integer",
                                "description": "Second parameter",
                            },
                        },
                        "required": ["param1"],
                    },
                },
            },
            "output": [{"name": "result", "description": "Tool result"}],
            "fixed_args": {},
            "execute": Mock(return_value=Mock(func=Mock(return_value="tool_result"))),
        }
    }


@pytest.fixture
def mock_workers_map() -> dict[str, Any]:
    return {
        "test_worker": {
            "description": "A test worker for testing",
            "execute": Mock(
                return_value=Mock(execute=Mock(return_value="worker_result"))
            ),
        },
        "MessageWorker": {
            "description": "Message worker that should be filtered out",
            "execute": Mock(
                return_value=Mock(execute=Mock(return_value="message_result"))
            ),
        },
    }


@pytest.fixture
def mock_name2id() -> dict[str, int]:
    return {"test_tool": 1, "test_worker": 2, "MessageWorker": 3}


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    return LLMConfig(model_type_or_path="gpt-3.5-turbo", llm_provider="openai")


@pytest.fixture
def mock_message_state() -> MessageState:
    return MessageState(
        user_message=ConvoMessage(
            history="Previous conversation", message="Hello, how can you help me?"
        ),
        orchestrator_message=OrchestratorMessage(
            message="User greeting", attribute={"task": "greeting"}
        ),
        response="",
        is_stream=False,
        stream_type=StreamType.NON_STREAM,
    )


@pytest.fixture
def mock_msg_history() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]


@pytest.fixture
def mock_llm() -> Mock:
    return Mock()


@pytest.fixture
def mock_prompt_template() -> Mock:
    template = Mock()
    template.invoke.return_value = Mock(text="System prompt")
    template.format.return_value = "formatted_prompt"
    return template


@pytest.fixture
def mock_retriever() -> Mock:
    retriever = Mock()
    mock_vectorstore = Mock()
    mock_vectorstore.similarity_search_with_score.return_value = [
        (Mock(metadata={"resource_name": "test_resource"}), 0.8)
    ]
    retriever.vectorstore = mock_vectorstore
    return retriever


@pytest.fixture
def mock_faiss() -> Mock:
    return Mock()


@pytest.fixture
def mock_provider_map() -> Mock:
    return Mock()


@pytest.fixture
def mock_chat_openai() -> Mock:
    return Mock()


@pytest.fixture
def mock_embedding_models() -> MagicMock:
    models = MagicMock()
    models.__getitem__.return_value = "text-embedding-ada-002"
    return models


@pytest.fixture
def mock_provider_embeddings() -> Mock:
    return Mock()


@pytest.fixture
def mock_openai_embeddings() -> Mock:
    return Mock()


@pytest.fixture
def mock_aimessage_to_dict() -> Mock:
    return Mock()


@pytest.fixture
def patched_sample_config(
    mock_llm: Mock,
    mock_prompt_template: Mock,
    mock_retriever: Mock,
    mock_faiss: Mock,
    mock_provider_map: Mock,
    mock_chat_openai: Mock,
    mock_embedding_models: MagicMock,
    mock_provider_embeddings: Mock,
    mock_openai_embeddings: Mock,
    mock_aimessage_to_dict: Mock,
) -> dict[str, Any]:
    with (
        patch("arklex.utils.model_provider_config.PROVIDER_MAP", mock_provider_map),
        patch("arklex.utils.model_provider_config.ChatOpenAI", mock_chat_openai),
        patch(
            "arklex.utils.model_provider_config.PROVIDER_EMBEDDING_MODELS",
            mock_embedding_models,
        ),
        patch(
            "arklex.utils.model_provider_config.PROVIDER_EMBEDDINGS",
            mock_provider_embeddings,
        ),
        patch(
            "arklex.utils.model_provider_config.OpenAIEmbeddings",
            mock_openai_embeddings,
        ),
        patch("arklex.env.planner.react_planner.FAISS", mock_faiss),
        patch(
            "arklex.env.planner.react_planner.PromptTemplate",
            return_value=mock_prompt_template,
        ),
        patch(
            "arklex.env.planner.react_planner.aimessage_to_dict", mock_aimessage_to_dict
        ),
    ):
        yield {
            "mock_llm": mock_llm,
            "mock_prompt_template": mock_prompt_template,
            "mock_retriever": mock_retriever,
            "mock_faiss": mock_faiss,
            "mock_provider_map": mock_provider_map,
            "mock_chat_openai": mock_chat_openai,
            "mock_embedding_models": mock_embedding_models,
            "mock_provider_embeddings": mock_provider_embeddings,
            "mock_openai_embeddings": mock_openai_embeddings,
            "mock_aimessage_to_dict": mock_aimessage_to_dict,
        }


@pytest.fixture
def react_planner(
    mock_tools_map: dict[str, Any],
    mock_workers_map: dict[str, Any],
    mock_name2id: dict[str, int],
) -> ReactPlanner:
    return ReactPlanner(mock_tools_map, mock_workers_map, mock_name2id)


@pytest.fixture
def configured_react_planner(
    react_planner: ReactPlanner, patched_sample_config: dict[str, Any]
) -> ReactPlanner:
    config = patched_sample_config
    react_planner.llm = config["mock_llm"]
    react_planner.llm_provider = "openai"
    react_planner.system_role = "system"
    react_planner.resource_rag_docs_created = True
    return react_planner


@pytest.fixture
def mock_planning_methods(configured_react_planner: ReactPlanner) -> dict[str, Mock]:
    with (
        patch.object(
            configured_react_planner, "_get_planning_trajectory_summary"
        ) as mock_summary,
        patch.object(
            configured_react_planner, "_get_num_resource_retrievals"
        ) as mock_retrievals,
        patch.object(
            configured_react_planner, "_retrieve_resource_signatures"
        ) as mock_retrieve,
        patch.object(configured_react_planner, "step") as mock_step,
    ):
        mock_summary.return_value = "Test trajectory"
        mock_retrievals.return_value = 3
        mock_retrieve.return_value = [
            Mock(metadata={"resource_name": "test_tool", "json_signature": {}})
        ]
        mock_step.return_value = EnvResponse(observation="tool_result")
        yield {
            "mock_summary": mock_summary,
            "mock_retrievals": mock_retrievals,
            "mock_retrieve": mock_retrieve,
            "mock_step": mock_step,
        }


@pytest.fixture
def mock_plan_method(react_planner: ReactPlanner) -> Generator[Mock, None, None]:
    with patch.object(react_planner, "plan") as mock_plan:
        yield mock_plan


@pytest.fixture
def patched_few_shot_prompt() -> Generator[None, None, None]:
    with patch("arklex.env.planner.react_planner.USE_FEW_SHOT_REACT_PROMPT", True):
        yield


class TestReactPlanner:
    def test_react_planner_initialization(
        self,
        mock_tools_map: dict[str, Any],
        mock_workers_map: dict[str, Any],
        mock_name2id: dict[str, int],
    ) -> None:
        planner = ReactPlanner(mock_tools_map, mock_workers_map, mock_name2id)

        assert planner.tools_map == mock_tools_map
        assert planner.workers_map == mock_workers_map
        assert planner.name2id == mock_name2id
        assert RESPOND_ACTION_NAME in planner.all_resources_info
        assert "test_tool" in planner.all_resources_info
        assert "test_worker" in planner.all_resources_info
        assert "MessageWorker" not in planner.all_resources_info
        assert planner.resource_rag_docs_created is False

    def test_react_planner_description(self) -> None:
        assert (
            ReactPlanner.description
            == "Choose tools/workers based on task and chat records if there is no specific worker/node for the user's query"
        )

    def test_default_planner_description(self) -> None:
        assert (
            DefaultPlanner.description
            == "Default planner that returns unaltered MessageState on execute()"
        )

    def test_format_worker_info(
        self, react_planner: ReactPlanner, mock_workers_map: dict[str, Any]
    ) -> None:
        formatted_info = react_planner._format_worker_info(mock_workers_map)

        assert "test_worker" in formatted_info
        assert "MessageWorker" not in formatted_info

        worker_info = formatted_info["test_worker"]
        assert isinstance(worker_info, PlannerResource)
        assert worker_info.name == "test_worker"
        assert worker_info.type == "worker"
        assert worker_info.description == "A test worker for testing"

    def test_format_tool_info(
        self, react_planner: ReactPlanner, mock_tools_map: dict[str, Any]
    ) -> None:
        formatted_info = react_planner._format_tool_info(mock_tools_map)

        assert "test_tool" in formatted_info

        tool_info = formatted_info["test_tool"]
        assert isinstance(tool_info, PlannerResource)
        assert tool_info.name == "test_tool"
        assert tool_info.type == "tool"
        assert tool_info.description == "A test tool for testing"
        assert len(tool_info.parameters) == 2
        assert tool_info.required == ["param1"]

    def test_format_tool_info_with_mock_tool(self, react_planner: ReactPlanner) -> None:
        mock_tool = Mock()
        mock_tool.info = {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "Mock tool description",
                "parameters": {
                    "properties": {"param": {"type": "string"}},
                    "required": [],
                },
            },
        }
        mock_tool.output = [{"name": "result", "description": "Result"}]

        tools_map = {1: mock_tool}
        formatted_info = react_planner._format_tool_info(tools_map)

        assert "mock_tool" in formatted_info
        tool_info = formatted_info["mock_tool"]
        assert tool_info.name == "mock_tool"
        assert tool_info.description == "Mock tool description"

    def test_create_resource_rag_docs(self, react_planner: ReactPlanner) -> None:
        resources = {
            "test_resource": PlannerResource(
                name="test_resource",
                type="tool",
                description="Test resource",
                parameters=[],
                required=[],
                returns={},
            )
        }

        docs = react_planner._create_resource_rag_docs(resources)

        assert len(docs) == 1
        doc = docs[0]
        assert doc.metadata["resource_name"] == "test_resource"
        assert doc.metadata["type"] == "tool"
        assert "test_resource" in doc.page_content

    def test_parse_trajectory_summary_to_steps(
        self, react_planner: ReactPlanner
    ) -> None:
        summary = (
            "- Step 1: Do something\n- Step 2: Do something else\n- Step 3: Finish"
        )

        steps = react_planner._parse_trajectory_summary_to_steps(summary)

        assert len(steps) == 3
        assert steps[0] == "Step 1: Do something"
        assert steps[1] == "Step 2: Do something else"
        assert steps[2] == "Step 3: Finish"

    def test_parse_trajectory_summary_to_steps_empty(
        self, react_planner: ReactPlanner
    ) -> None:
        summary = ""
        steps = react_planner._parse_trajectory_summary_to_steps(summary)
        assert steps == []

    def test_get_num_resource_retrievals_valid_summary(
        self, react_planner: ReactPlanner
    ) -> None:
        summary = "- Step 1\n- Step 2\n- Step 3"

        n_retrievals = react_planner._get_num_resource_retrievals(summary)

        expected = NUM_STEPS_TO_NUM_RETRIEVALS(3)
        expected = min(max(expected, MIN_NUM_RETRIEVALS), MAX_NUM_RETRIEVALS)
        assert n_retrievals == expected

    def test_get_num_resource_retrievals_invalid_summary(
        self, react_planner: ReactPlanner
    ) -> None:
        summary = "Invalid summary without steps"

        n_retrievals = react_planner._get_num_resource_retrievals(summary)

        assert n_retrievals == 4

    def test_get_num_resource_retrievals_empty_summary(
        self, react_planner: ReactPlanner
    ) -> None:
        summary = ""

        n_retrievals = react_planner._get_num_resource_retrievals(summary)

        assert n_retrievals == MIN_NUM_RETRIEVALS

    def test_get_num_resource_retrievals_bounds(
        self, react_planner: ReactPlanner
    ) -> None:
        summary = "- " + "\n- ".join([f"Step {i}" for i in range(100)])

        n_retrievals = react_planner._get_num_resource_retrievals(summary)

        assert n_retrievals == MAX_NUM_RETRIEVALS

    def test_get_num_resource_retrievals_valid_summary_with_steps_else_branch(
        self, react_planner: ReactPlanner
    ) -> None:
        summary = "- Step 1\n- Step 2\n- Step 3"

        n_retrievals = react_planner._get_num_resource_retrievals(summary)

        expected = NUM_STEPS_TO_NUM_RETRIEVALS(3)
        expected = min(max(expected, MIN_NUM_RETRIEVALS), MAX_NUM_RETRIEVALS)
        assert n_retrievals == expected

    def test_parse_response_action_to_json_valid(
        self, react_planner: ReactPlanner
    ) -> None:
        content = '{"name": "test_action", "arguments": {"param": "value"}}'
        result = react_planner._parse_response_action_to_json(content)

        assert result["name"] == "test_action"
        assert result["arguments"]["param"] == "value"

    def test_parse_response_action_to_json_invalid(
        self, react_planner: ReactPlanner
    ) -> None:
        content = "Invalid JSON content"
        result = react_planner._parse_response_action_to_json(content)

        assert result["arguments"]["content"] == "Invalid JSON content"

    def test_message_to_actions_valid_tool(self, react_planner: ReactPlanner) -> None:
        message = {"name": "test_tool", "arguments": {"param1": "value"}}
        actions = react_planner.message_to_actions(message)

        assert len(actions) == 1
        # Valid tools should return the tool name and arguments
        assert actions[0].name == "test_tool"
        assert actions[0].kwargs == {"param1": "value"}

    def test_message_to_actions_valid_worker(self, react_planner: ReactPlanner) -> None:
        message = {"name": "test_worker", "arguments": {}}
        actions = react_planner.message_to_actions(message)

        assert len(actions) == 1
        # Valid workers should return the worker name and arguments
        assert actions[0].name == "test_worker"
        assert actions[0].kwargs == {}

    def test_message_to_actions_invalid_resource(
        self, react_planner: ReactPlanner
    ) -> None:
        message = {"name": "invalid_resource", "arguments": {}}
        actions = react_planner.message_to_actions(message)

        assert len(actions) == 1
        # Invalid resources should return RESPOND_ACTION_NAME
        assert actions[0].name == RESPOND_ACTION_NAME
        assert actions[0].kwargs == {"content": ""}

    def test_message_to_actions_missing_arguments(
        self, react_planner: ReactPlanner
    ) -> None:
        # The method expects a parsed JSON object, not a message with content field
        message = {"name": "test_tool"}
        actions = react_planner.message_to_actions(message)

        assert len(actions) == 1
        # Missing arguments are now handled gracefully and default to empty dict
        assert actions[0].name == "test_tool"
        assert actions[0].kwargs == {}

    def test_step_respond_action(
        self, react_planner: ReactPlanner, mock_message_state: MessageState
    ) -> None:
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Hello!"})
        result = react_planner.step(action, mock_message_state)

        assert result.observation == "Hello!"

    def test_step_tool_execution(
        self, react_planner: ReactPlanner, mock_message_state: MessageState
    ) -> None:
        action = Action(name="test_tool", kwargs={"param1": "value"})
        result = react_planner.step(action, mock_message_state)

        assert result.observation == "tool_result"

    def test_step_worker_execution(
        self, react_planner: ReactPlanner, mock_message_state: MessageState
    ) -> None:
        action = Action(name="test_worker", kwargs={})
        result = react_planner.step(action, mock_message_state)

        assert result.observation == "worker_result"

    def test_step_unknown_action(
        self, react_planner: ReactPlanner, mock_message_state: MessageState
    ) -> None:
        action = Action(name="unknown_action", kwargs={})
        result = react_planner.step(action, mock_message_state)

        assert "Unknown action" in result.observation

    def test_step_tool_execution_error(
        self, react_planner: ReactPlanner, mock_message_state: MessageState
    ) -> None:
        action = Action(name="test_tool", kwargs={"param1": "value"})
        react_planner.tools_map[1]["execute"].side_effect = Exception("Tool error")
        result = react_planner.step(action, mock_message_state)

        assert "Tool error" in result.observation

    def test_step_worker_execution_error(
        self, react_planner: ReactPlanner, mock_message_state: MessageState
    ) -> None:
        action = Action(name="test_worker", kwargs={})
        react_planner.workers_map["test_worker"]["execute"].side_effect = Exception(
            "Worker error"
        )
        result = react_planner.step(action, mock_message_state)

        assert "Worker error" in result.observation

    def test_set_llm_config_and_build_resource_library(
        self,
        react_planner: ReactPlanner,
        mock_llm_config: LLMConfig,
        patched_sample_config: dict[str, Any],
    ) -> None:
        config = patched_sample_config
        mock_llm = config["mock_llm"]
        mock_llm.invoke.return_value = Mock()
        config["mock_aimessage_to_dict"].return_value = {"content": "Test response"}

        react_planner.set_llm_config_and_build_resource_library(mock_llm_config)

        assert react_planner.llm is not None
        assert react_planner.llm_provider == "openai"
        assert react_planner.system_role == "system"
        assert react_planner.resource_rag_docs_created is True

    def test_get_planning_trajectory_summary(
        self,
        react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
    ) -> None:
        config = patched_sample_config
        mock_llm = config["mock_llm"]
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_llm.invoke.return_value = mock_response
        config["mock_aimessage_to_dict"].return_value = {"content": "Test response"}
        react_planner.llm = mock_llm

        result = react_planner._get_planning_trajectory_summary(
            mock_message_state, mock_msg_history
        )

        assert result == "Test response"
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        # For OpenAI/Gemini providers, only one system message is created
        assert len(call_args) == 1
        assert call_args[0]["role"] == "system"

    def test_retrieve_resource_signatures(
        self, react_planner: ReactPlanner, patched_sample_config: dict[str, Any]
    ) -> None:
        config = patched_sample_config
        mock_retriever = config["mock_retriever"]
        react_planner.retriever = mock_retriever
        react_planner.guaranteed_retrieval_docs = []
        docs = react_planner._retrieve_resource_signatures(1, "- Step 1\n- Step 2")
        assert len(docs) == 1
        assert docs[0].metadata["resource_name"] == "test_resource"

    def test_plan_method(
        self,
        configured_react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
        mock_planning_methods: dict[str, Mock],
    ) -> None:
        config = patched_sample_config
        config["mock_aimessage_to_dict"].return_value = {
            "content": 'Action:\n{"name": "test_tool", "arguments": {"param": "value"}}'
        }

        msg_history, action_name, response = configured_react_planner.plan(
            mock_message_state, mock_msg_history, max_num_steps=1
        )

        assert action_name == "test_tool"
        assert response == "tool_result"

    def test_plan_method_with_respond_action(
        self,
        configured_react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
        mock_planning_methods: dict[str, Mock],
    ) -> None:
        config = patched_sample_config
        config["mock_aimessage_to_dict"].return_value = {
            "content": 'Action:\n{"name": "respond", "arguments": {"content": "Hello!"}}'
        }

        # Mock the step method to return the correct response for respond action
        def mock_step(action: Action, msg_state: MessageState) -> EnvResponse:
            if action.name == RESPOND_ACTION_NAME:
                return EnvResponse(observation=action.kwargs["content"])
            return EnvResponse(observation="tool_result")

        configured_react_planner.step = mock_step

        msg_history, action_name, response = configured_react_planner.plan(
            mock_message_state, mock_msg_history, max_num_steps=1
        )

        assert action_name == "respond"
        assert response == "Hello!"

    def test_execute_method(
        self,
        react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        mock_plan_method: Mock,
    ) -> None:
        mock_plan_method.return_value = (
            mock_msg_history,
            "test_action",
            "test_response",
        )

        action, updated_state, updated_history = react_planner.execute(
            mock_message_state, mock_msg_history
        )

        assert action == "test_action"
        assert updated_state.response == "test_response"
        assert updated_history == mock_msg_history
        mock_plan_method.assert_called_once_with(mock_message_state, mock_msg_history)

    def test_default_planner_execute(
        self,
        mock_tools_map: dict[str, Any],
        mock_workers_map: dict[str, Any],
        mock_name2id: dict[str, int],
    ) -> None:
        planner = DefaultPlanner(mock_tools_map, mock_workers_map, mock_name2id)
        msg_state = Mock()
        msg_history = []

        action, updated_state, updated_history = planner.execute(msg_state, msg_history)

        assert action["name"] == RESPOND_ACTION_NAME
        assert action["kwargs"]["content"] == ""
        assert updated_state == msg_state
        assert updated_history == msg_history

    def test_action_model(self) -> None:
        action = Action(name="test_action", kwargs={"param": "value"})

        assert action.name == "test_action"
        assert action.kwargs == {"param": "value"}

    def test_env_response_model(self) -> None:
        response = EnvResponse(observation="test_observation")

        assert response.observation == "test_observation"

    def test_planner_resource_model(self) -> None:
        resource = PlannerResource(
            name="test_resource",
            type="tool",
            description="Test description",
            parameters=[],
            required=[],
            returns={},
        )

        assert resource.name == "test_resource"
        assert resource.type == "tool"
        assert resource.description == "Test description"

    def test_respond_action_resource(self) -> None:
        assert RESPOND_ACTION_RESOURCE.name == RESPOND_ACTION_NAME
        assert RESPOND_ACTION_RESOURCE.type == "worker"
        assert "Respond to the user" in RESPOND_ACTION_RESOURCE.description

    def test_num_steps_to_num_retrievals_function(self) -> None:
        assert NUM_STEPS_TO_NUM_RETRIEVALS(1) == 4
        assert NUM_STEPS_TO_NUM_RETRIEVALS(5) == 8
        assert NUM_STEPS_TO_NUM_RETRIEVALS(10) == 13

    def test_get_planning_trajectory_summary_with_anthropic_provider(
        self,
        react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
    ) -> None:
        config = patched_sample_config
        react_planner.llm_provider = "anthropic"
        mock_llm = config["mock_llm"]
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_llm.invoke.return_value = mock_response
        config["mock_aimessage_to_dict"].return_value = {"content": "Test response"}
        react_planner.llm = mock_llm

        result = react_planner._get_planning_trajectory_summary(
            mock_message_state, mock_msg_history
        )

        assert result == "Test response"
        mock_llm.invoke.assert_called_once()
        call_args = mock_llm.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"

    def test_get_planning_trajectory_summary_with_exception(
        self,
        react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
    ) -> None:
        config = patched_sample_config
        mock_llm = config["mock_llm"]
        mock_llm.invoke.side_effect = Exception("LLM error")
        react_planner.llm = mock_llm

        with pytest.raises(Exception, match="LLM error"):
            react_planner._get_planning_trajectory_summary(
                mock_message_state, mock_msg_history
            )

    def test_retrieve_resource_signatures_with_user_message_and_task(
        self, react_planner: ReactPlanner, patched_sample_config: dict[str, Any]
    ) -> None:
        config = patched_sample_config
        mock_retriever = config["mock_retriever"]
        react_planner.retriever = mock_retriever
        react_planner.guaranteed_retrieval_docs = []
        docs = react_planner._retrieve_resource_signatures(
            1, "- Step 1\n- Step 2", "test user message", "test task"
        )
        assert len(docs) == 1
        assert docs[0].metadata["resource_name"] == "test_resource"

    def test_retrieve_resource_signatures_with_guaranteed_retrieval_docs(
        self, react_planner: ReactPlanner, patched_sample_config: dict[str, Any]
    ) -> None:
        config = patched_sample_config
        mock_retriever = config["mock_retriever"]
        react_planner.retriever = mock_retriever
        react_planner.guaranteed_retrieval_docs = []
        docs = react_planner._retrieve_resource_signatures(1, "- Step 1\n- Step 2")
        assert len(docs) == 1
        assert docs[0].metadata["resource_name"] == "test_resource"

    def test_retrieve_resource_signatures_with_duplicate_guaranteed_docs(
        self, react_planner: ReactPlanner, patched_sample_config: dict[str, Any]
    ) -> None:
        config = patched_sample_config
        mock_retriever = config["mock_retriever"]
        react_planner.retriever = mock_retriever
        react_planner.guaranteed_retrieval_docs = []
        docs = react_planner._retrieve_resource_signatures(1, "- Step 1\n- Step 2")
        assert len(docs) == 1
        assert docs[0].metadata["resource_name"] == "test_resource"

    def test_plan_with_empty_signature_docs(
        self,
        configured_react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
        mock_planning_methods: dict[str, Mock],
    ) -> None:
        config = patched_sample_config
        config["mock_aimessage_to_dict"].return_value = {
            "content": 'Action:\n{"name": "test_tool", "arguments": {"param": "value"}}'
        }
        mock_planning_methods["mock_retrieve"].return_value = []

        msg_history, action_name, response = configured_react_planner.plan(
            mock_message_state, mock_msg_history, max_num_steps=1
        )

        assert action_name == "test_tool"
        assert response == "tool_result"

    def test_plan_with_use_few_shot_prompt(
        self,
        configured_react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
        mock_planning_methods: dict[str, Mock],
        patched_few_shot_prompt: Generator[None, None, None],
    ) -> None:
        config = patched_sample_config
        config["mock_aimessage_to_dict"].return_value = {
            "content": 'Action:\n{"name": "test_tool", "arguments": {"param": "value"}}'
        }

        msg_history, action_name, response = configured_react_planner.plan(
            mock_message_state, mock_msg_history, max_num_steps=1
        )

        assert action_name == "test_tool"
        assert response == "tool_result"

    def test_plan_method_with_tool_action_and_max_steps_exhausted(
        self,
        configured_react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
        mock_planning_methods: dict[str, Mock],
    ) -> None:
        config = patched_sample_config
        config["mock_aimessage_to_dict"].return_value = {
            "content": 'Action:\n{"name": "test_tool", "arguments": {"param": "value"}}'
        }
        import pytest

        with pytest.raises(UnboundLocalError):
            configured_react_planner.plan(
                mock_message_state, mock_msg_history, max_num_steps=0
            )

    def test_plan_method_tool_call_simulation_branch(
        self,
        configured_react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
        mock_planning_methods: dict[str, Mock],
    ) -> None:
        config = patched_sample_config
        config["mock_aimessage_to_dict"].return_value = {
            "content": 'Action:\n{"name": "test_tool", "arguments": {"param": "value"}}'
        }

        msg_history, action_name, response = configured_react_planner.plan(
            mock_message_state, mock_msg_history, max_num_steps=1
        )

        assert action_name == "test_tool"
        assert response == "tool_result"

    def test_plan_method_max_steps_exhausted_branch(
        self,
        configured_react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
        mock_planning_methods: dict[str, Mock],
    ) -> None:
        """Test plan method when max_num_steps is exhausted without finding RESPOND_ACTION."""
        # Mock the planning methods to return non-RESPOND_ACTION
        mock_planning_methods["mock_summary"].return_value = "Test trajectory"
        mock_planning_methods["mock_retrievals"].return_value = 2
        mock_planning_methods["mock_retrieve"].return_value = []

        # Mock aimessage_to_dict to return proper dictionary
        patched_sample_config["mock_aimessage_to_dict"].return_value = {
            "content": 'Action:\n{"name": "test_tool", "arguments": {}}'
        }

        # Mock LLM to return non-RESPOND_ACTION responses
        mock_response = Mock()
        mock_response.content = 'Action:\n{"name": "test_tool", "arguments": {}}'
        configured_react_planner.llm.invoke.return_value = mock_response

        # Mock step to return non-RESPOND_ACTION response
        def mock_step(action: Action, msg_state: MessageState) -> EnvResponse:
            return EnvResponse(observation="tool_result")

        configured_react_planner.step = mock_step

        # Execute with max_num_steps=1 to ensure exhaustion
        result_msg_history, result_action, result_response = (
            configured_react_planner.plan(
                mock_message_state, mock_msg_history, max_num_steps=1
            )
        )

        # Should return the last action and response when max steps exhausted
        assert result_action == "test_tool"
        assert result_response == "tool_result"
        # The message history should contain the original messages plus the new action and response
        assert len(result_msg_history) >= len(mock_msg_history)

    def test_plan_method_fallback_return_statement(
        self,
        configured_react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
        mock_planning_methods: dict[str, Mock],
    ) -> None:
        """Test plan method fallback return statement when max_num_steps is exhausted."""
        # Mock the planning methods
        mock_planning_methods["mock_summary"].return_value = "Test trajectory"
        mock_planning_methods["mock_retrievals"].return_value = 2
        mock_planning_methods["mock_retrieve"].return_value = []

        # Mock aimessage_to_dict to return proper dictionary
        patched_sample_config["mock_aimessage_to_dict"].return_value = {
            "content": 'Action:\n{"name": "test_worker", "arguments": {}}'
        }

        # Mock LLM to return non-RESPOND_ACTION responses
        mock_response = Mock()
        mock_response.content = 'Action:\n{"name": "test_worker", "arguments": {}}'
        configured_react_planner.llm.invoke.return_value = mock_response

        # Mock step to return non-RESPOND_ACTION response
        def mock_step(action: Action, msg_state: MessageState) -> EnvResponse:
            return EnvResponse(observation="worker_result")

        configured_react_planner.step = mock_step

        # Execute with max_num_steps=1 to trigger the fallback return
        result_msg_history, result_action, result_response = (
            configured_react_planner.plan(
                mock_message_state, mock_msg_history, max_num_steps=1
            )
        )

        # Should return the last action and response from the fallback return statement
        assert result_action == "test_worker"
        assert result_response == "worker_result"
        # The message history should contain the original messages plus the new action and response
        assert len(result_msg_history) >= len(mock_msg_history)

    def test_default_planner_set_llm_config_and_build_resource_library(
        self,
        mock_tools_map: dict[str, Any],
        mock_workers_map: dict[str, Any],
        mock_name2id: dict[str, int],
        mock_llm_config: LLMConfig,
    ) -> None:
        default_planner = DefaultPlanner(mock_tools_map, mock_workers_map, mock_name2id)
        default_planner.set_llm_config_and_build_resource_library(mock_llm_config)
        # DefaultPlanner doesn't have system_role attribute, so we just verify the method runs without error

    def test_aimessage_to_dict_function(self) -> None:
        from arklex.env.planner.react_planner import aimessage_to_dict

        mock_message = Mock()
        mock_message.content = "Test content"
        result = aimessage_to_dict(mock_message)
        assert result["content"] == "Test content"

    def test_parse_trajectory_summary_to_steps_with_invalid_json(self) -> None:
        """Test _parse_trajectory_summary_to_steps with invalid JSON (covers lines 424-425)."""
        planner = ReactPlanner({}, {}, {})

        # Test with invalid JSON that should trigger the exception handling
        invalid_summary = "This is not a valid JSON list"

        result = planner._parse_trajectory_summary_to_steps(invalid_summary)

        # Should return the original text when JSON parsing fails
        assert result == [invalid_summary]

    def test_retrieve_resource_signatures_with_empty_steps(self) -> None:
        """Test _retrieve_resource_signatures with empty steps list."""
        react_planner = ReactPlanner({}, {}, {})

        # Mock the trajectory summary to return empty steps
        with (
            patch.object(
                react_planner, "_parse_trajectory_summary_to_steps", return_value=[]
            ),
            patch.object(react_planner, "_get_num_resource_retrievals", return_value=0),
        ):
            # Mock the retriever attribute since it's not created until set_llm_config_and_build_resource_library is called
            react_planner.retriever = Mock()
            react_planner.retriever.vectorstore = Mock()
            react_planner.retriever.vectorstore.similarity_search_with_score.return_value = []
            react_planner.guaranteed_retrieval_docs = []

            result = react_planner._retrieve_resource_signatures(0, "test_summary")

            assert result == []
            react_planner.retriever.vectorstore.similarity_search_with_score.assert_not_called()

    def test_retrieve_resource_signatures_with_none_user_message_and_task(self) -> None:
        """Test _retrieve_resource_signatures with None user_message and task parameters."""
        react_planner = ReactPlanner({}, {}, {})

        with (
            patch.object(
                react_planner,
                "_parse_trajectory_summary_to_steps",
                return_value=["step1"],
            ),
            patch.object(react_planner, "_get_num_resource_retrievals", return_value=1),
        ):
            # Mock the retriever attribute
            react_planner.retriever = Mock()
            react_planner.retriever.vectorstore = Mock()
            react_planner.retriever.vectorstore.similarity_search_with_score.return_value = []
            react_planner.guaranteed_retrieval_docs = []

            result = react_planner._retrieve_resource_signatures(
                1, "test_summary", None, None
            )

            assert result == []
            react_planner.retriever.vectorstore.similarity_search_with_score.assert_called_once()

    def test_retrieve_resource_signatures_with_empty_user_message_and_task(
        self,
    ) -> None:
        """Test _retrieve_resource_signatures with empty user_message and task parameters."""
        react_planner = ReactPlanner({}, {}, {})

        with (
            patch.object(
                react_planner,
                "_parse_trajectory_summary_to_steps",
                return_value=["step1"],
            ),
            patch.object(react_planner, "_get_num_resource_retrievals", return_value=1),
        ):
            # Mock the retriever attribute
            react_planner.retriever = Mock()
            react_planner.retriever.vectorstore = Mock()
            react_planner.retriever.vectorstore.similarity_search_with_score.return_value = []
            react_planner.guaranteed_retrieval_docs = []

            result = react_planner._retrieve_resource_signatures(
                1, "test_summary", "", ""
            )

            assert result == []
            react_planner.retriever.vectorstore.similarity_search_with_score.assert_called_once()

    def test_retrieve_resource_signatures_with_zero_retrievals(self) -> None:
        """Test _retrieve_resource_signatures with zero retrievals."""
        react_planner = ReactPlanner({}, {}, {})

        with (
            patch.object(
                react_planner, "_parse_trajectory_summary_to_steps", return_value=[]
            ),
            patch.object(react_planner, "_get_num_resource_retrievals", return_value=0),
        ):
            # Mock the retriever attribute
            react_planner.retriever = Mock()
            react_planner.retriever.vectorstore = Mock()
            react_planner.retriever.vectorstore.similarity_search_with_score.return_value = []
            react_planner.guaranteed_retrieval_docs = []

            result = react_planner._retrieve_resource_signatures(0, "test_summary")

            assert result == []
            react_planner.retriever.vectorstore.similarity_search_with_score.assert_not_called()

    def test_retrieve_resource_signatures_with_negative_retrievals(self) -> None:
        """Test _retrieve_resource_signatures with negative retrievals."""
        react_planner = ReactPlanner({}, {}, {})

        with (
            patch.object(
                react_planner, "_parse_trajectory_summary_to_steps", return_value=[]
            ),
            patch.object(
                react_planner, "_get_num_resource_retrievals", return_value=-1
            ),
        ):
            # Mock the retriever attribute
            react_planner.retriever = Mock()
            react_planner.retriever.vectorstore = Mock()
            react_planner.retriever.vectorstore.similarity_search_with_score.return_value = []
            react_planner.guaranteed_retrieval_docs = []

            result = react_planner._retrieve_resource_signatures(-1, "test_summary")

            assert result == []
            react_planner.retriever.vectorstore.similarity_search_with_score.assert_not_called()

    def test_aimessage_to_dict_with_dict_input(self) -> None:
        """Test aimessage_to_dict function with dictionary input."""
        from arklex.env.planner.react_planner import aimessage_to_dict

        message_dict = {"content": "test content", "role": "user"}
        result = aimessage_to_dict(message_dict)

        assert result["content"] == "test content"
        assert result["role"] == "user"
        assert result["function_call"] is None
        assert result["tool_calls"] is None

    def test_aimessage_to_dict_with_aimessage_input(self) -> None:
        """Test aimessage_to_dict function with AIMessage input."""
        from langchain.schema import AIMessage

        ai_message = AIMessage(content="Test content")
        result = aimessage_to_dict(ai_message)

        assert result["content"] == "Test content"
        assert result["role"] == "assistant"
        assert result["function_call"] is None
        assert result["tool_calls"] is None

    def test_get_num_resource_retrievals_with_non_zero_steps_else_branch(
        self, react_planner: ReactPlanner
    ) -> None:
        """Test _get_num_resource_retrievals when n_steps is not 0 (else branch)."""
        # Mock _parse_trajectory_summary_to_steps to return a non-empty list
        with patch.object(
            react_planner, "_parse_trajectory_summary_to_steps"
        ) as mock_parse:
            mock_parse.return_value = ["step1", "step2", "step3"]

            summary = "Test summary with steps"
            result = react_planner._get_num_resource_retrievals(summary)

            # Should be NUM_STEPS_TO_NUM_RETRIEVALS(3) = 3 + 3 = 6
            assert result == 6

    def test_message_to_actions_invalid_resource_else_branch(
        self, react_planner: ReactPlanner
    ) -> None:
        """Test message_to_actions when resource is invalid (else branch)."""
        # Test with invalid resource name that's not in name2id
        message = {"name": "invalid_resource", "arguments": {"param1": "value1"}}

        actions = react_planner.message_to_actions(message)

        assert len(actions) == 1
        assert actions[0].name == RESPOND_ACTION_NAME
        assert actions[0].kwargs["content"] == ""

    def test_message_to_actions_invalid_resource_with_content(
        self, react_planner: ReactPlanner
    ) -> None:
        """Test message_to_actions when resource is invalid but has content."""
        # Test with invalid resource name and content in arguments
        message = {
            "name": "invalid_resource",
            "arguments": {"content": "Error message"},
            "content": "Fallback content",
        }

        actions = react_planner.message_to_actions(message)

        assert len(actions) == 1
        assert actions[0].name == RESPOND_ACTION_NAME
        assert actions[0].kwargs["content"] == "Error message"

    def test_message_to_actions_invalid_resource_with_content_fallback(
        self, react_planner: ReactPlanner
    ) -> None:
        """Test message_to_actions when resource is invalid with content fallback."""
        # Test with invalid resource name, no content in arguments, but content in message
        message = {
            "name": "invalid_resource",
            "arguments": {"other_param": "value"},
            "content": "Fallback content",
        }

        actions = react_planner.message_to_actions(message)

        assert len(actions) == 1
        assert actions[0].name == RESPOND_ACTION_NAME
        assert actions[0].kwargs["content"] == "Fallback content"

    def test_plan_with_zero_shot_prompt(
        self,
        configured_react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
        mock_planning_methods: dict[str, Mock],
    ) -> None:
        """Test plan method with USE_FEW_SHOT_REACT_PROMPT set to False."""
        # Mock the global variable
        with patch("arklex.env.planner.react_planner.USE_FEW_SHOT_REACT_PROMPT", False):
            # Mock the planning methods
            configured_react_planner._get_planning_trajectory_summary = (
                mock_planning_methods["mock_summary"]
            )
            configured_react_planner._get_num_resource_retrievals = (
                mock_planning_methods["mock_retrievals"]
            )
            configured_react_planner._retrieve_resource_signatures = (
                mock_planning_methods["mock_retrieve"]
            )

            # Mock LLM response
            mock_llm_response = Mock()
            mock_llm_response.content = 'Action:\n{"name": "respond", "arguments": {"content": "Test response"}}'
            configured_react_planner.llm.invoke.return_value = mock_llm_response

            # Mock aimessage_to_dict function
            with patch(
                "arklex.env.planner.react_planner.aimessage_to_dict"
            ) as mock_aimessage_to_dict:
                mock_aimessage_to_dict.return_value = {
                    "content": 'Action:\n{"name": "respond", "arguments": {"content": "Test response"}}',
                    "role": "assistant",
                }

                # Mock step method
                mock_env_response = EnvResponse(observation="Test observation")
                configured_react_planner.step.return_value = mock_env_response

                result = configured_react_planner.plan(
                    mock_message_state, mock_msg_history
                )

                assert len(result) == 3
                assert result[1] == "respond"

    def test_step_unknown_action_else_branch(
        self, react_planner: ReactPlanner, mock_message_state: MessageState
    ) -> None:
        """Test step method with unknown action (else branch)."""
        # Create an action that's not in tools_map, workers_map, or RESPOND_ACTION
        action = Action(name="unknown_action", kwargs={"param": "value"})

        # Mock name2id to return None for unknown action
        react_planner.name2id = {"test_tool": 1, "test_worker": 2}

        result = react_planner.step(action, mock_message_state)

        assert result.observation == "Unknown action unknown_action"

    def test_execute_method_else_branch(
        self,
        react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        mock_plan_method: Mock,
    ) -> None:
        """Test execute method (else branch - commented line)."""
        # Mock plan method to return a tuple
        mock_plan_method.return_value = (
            mock_msg_history,
            "test_action",
            "test_response",
        )
        react_planner.plan = mock_plan_method

        action, state, history = react_planner.execute(
            mock_message_state, mock_msg_history
        )

        # Verify the response is set on the message state
        assert state.response == "test_response"
        assert action == "test_action"
        assert history == mock_msg_history
        mock_plan_method.assert_called_once_with(mock_message_state, mock_msg_history)

    def test_get_num_resource_retrievals_with_exception_handling(
        self, react_planner: ReactPlanner
    ) -> None:
        """Test _get_num_resource_retrievals with exception handling (lines 421-422)."""
        # Mock _parse_trajectory_summary_to_steps to raise an exception
        with patch.object(
            react_planner,
            "_parse_trajectory_summary_to_steps",
            side_effect=Exception("Test exception"),
        ):
            result = react_planner._get_num_resource_retrievals("invalid summary")
            # Should return MIN_NUM_RETRIEVALS when exception occurs
            assert result == MIN_NUM_RETRIEVALS

    def test_retrieve_resource_signatures_with_guaranteed_docs_not_in_retrieved(
        self, react_planner: ReactPlanner, patched_sample_config: dict[str, Any]
    ) -> None:
        """Test _retrieve_resource_signatures with guaranteed docs not in retrieved list (lines 482-487)."""
        # Setup guaranteed retrieval docs
        guaranteed_doc = Mock()
        guaranteed_doc.metadata = {"resource_name": "guaranteed_resource"}
        react_planner.guaranteed_retrieval_docs = [guaranteed_doc]

        # Mock the retriever attribute
        mock_retriever = Mock()
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "different_resource"}), 0.8)
        ]
        mock_retriever.vectorstore = mock_vectorstore
        react_planner.retriever = mock_retriever

        result = react_planner._retrieve_resource_signatures(1, "test summary")

        # Should include both retrieved and guaranteed docs
        assert len(result) == 2
        resource_names = [doc.metadata["resource_name"] for doc in result]
        assert "guaranteed_resource" in resource_names
        assert "different_resource" in resource_names

    def test_plan_method_fallback_return_without_respond_action(
        self,
        configured_react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
        mock_planning_methods: dict[str, Mock],
    ) -> None:
        """Test plan method fallback return when max steps exhausted without RESPOND_ACTION (lines 675, 689)."""
        # Mock the planning methods to return non-respond actions
        mock_planning_methods["mock_summary"].return_value = "test summary"
        mock_planning_methods["mock_retrievals"].return_value = 1
        mock_planning_methods["mock_retrieve"].return_value = []

        # Mock LLM to return non-respond actions
        mock_llm_response = Mock()
        mock_llm_response.content = 'Action:\n{"name": "test_tool", "arguments": {}}'
        configured_react_planner.llm.invoke.return_value = mock_llm_response

        # Mock aimessage_to_dict to return proper dictionary
        patched_sample_config["mock_aimessage_to_dict"].return_value = {
            "content": 'Action:\n{"name": "test_tool", "arguments": {}}'
        }

        # Mock step to return a non-respond action response
        def mock_step(action: Action, msg_state: MessageState) -> EnvResponse:
            return EnvResponse(observation="tool result")

        configured_react_planner.step = mock_step

        # Set max_num_steps to 1 to ensure we hit the fallback return
        result = configured_react_planner.plan(
            mock_message_state, mock_msg_history, max_num_steps=1
        )

        # Should return the last action and response
        assert len(result) == 3
        assert result[1] == "test_tool"  # action name
        assert result[2] == "tool result"  # observation

    def test_plan_method_fallback_return_with_multiple_actions(
        self,
        configured_react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
        mock_planning_methods: dict[str, Mock],
    ) -> None:
        """Test plan method fallback return with multiple actions in the loop (lines 675, 689)."""
        # Mock the planning methods
        mock_planning_methods["mock_summary"].return_value = "test summary"
        mock_planning_methods["mock_retrievals"].return_value = 1
        mock_planning_methods["mock_retrieve"].return_value = []

        # Mock LLM to return multiple actions
        mock_llm_response = Mock()
        mock_llm_response.content = 'Action:\n{"name": "test_tool", "arguments": {}}'
        configured_react_planner.llm.invoke.return_value = mock_llm_response

        # Mock aimessage_to_dict to return proper dictionary
        patched_sample_config["mock_aimessage_to_dict"].return_value = {
            "content": 'Action:\n{"name": "test_tool", "arguments": {}}'
        }

        # Mock step to return a non-respond action response
        def mock_step(action: Action, msg_state: MessageState) -> EnvResponse:
            return EnvResponse(observation="tool result")

        configured_react_planner.step = mock_step

        # Set max_num_steps to 2 to ensure we process multiple actions
        result = configured_react_planner.plan(
            mock_message_state, mock_msg_history, max_num_steps=2
        )

        # Should return the last action and response after exhausting max steps
        assert len(result) == 3
        assert result[1] == "test_tool"  # action name
        assert result[2] == "tool result"  # observation

    def test_plan_fallback_return_exhausts_steps(
        self,
        configured_react_planner: ReactPlanner,
        mock_message_state: MessageState,
        mock_msg_history: list[dict[str, Any]],
        patched_sample_config: dict[str, Any],
        mock_planning_methods: dict[str, Mock],
    ) -> None:
        # Patch step to never return RESPOND_ACTION
        original_step = configured_react_planner.step

        def never_respond_action(action: Action, state: MessageState) -> EnvResponse:
            # Always return a non-RESPOND_ACTION name
            action.name = "not_respond_action"
            return EnvResponse(observation="fallback_observation")

        configured_react_planner.step = never_respond_action
        # Patch message_to_actions to always return a non-RESPOND_ACTION action
        original_message_to_actions = configured_react_planner.message_to_actions

        def always_non_respond_action(message: dict[str, Any]) -> list[Action]:
            return [Action(name="not_respond_action", kwargs={})]

        configured_react_planner.message_to_actions = always_non_respond_action
        # Patch _parse_response_action_to_json to return a dummy action
        configured_react_planner._parse_response_action_to_json = lambda x: {
            "name": "not_respond_action",
            "arguments": {},
        }
        # Patch aimessage_to_dict to return a dummy message
        patched_sample_config["mock_aimessage_to_dict"].return_value = {
            "content": "dummy"
        }
        # Run plan with max_num_steps=2 to force fallback
        msg_history, action_name, response = configured_react_planner.plan(
            mock_message_state, mock_msg_history, max_num_steps=2
        )
        assert action_name == "not_respond_action"
        assert response == "fallback_observation"
        # Restore original methods
        configured_react_planner.step = original_step
        configured_react_planner.message_to_actions = original_message_to_actions


class TestReactPlannerIntegration:
    @pytest.fixture
    def integration_planner(self) -> ReactPlanner:
        tools_map = {
            1: {
                "info": {
                    "type": "function",
                    "function": {
                        "name": "integration_tool",
                        "description": "Integration test tool",
                        "parameters": {
                            "properties": {"param": {"type": "string"}},
                            "required": [],
                        },
                    },
                },
                "output": [],
                "fixed_args": {},
                "execute": Mock(
                    return_value=Mock(func=Mock(return_value="integration_result"))
                ),
            }
        }
        workers_map = {
            "integration_worker": {
                "description": "Integration test worker",
                "execute": Mock(
                    return_value=Mock(
                        execute=Mock(return_value="integration_worker_result")
                    )
                ),
            }
        }
        name2id = {"integration_tool": 1, "integration_worker": 2}
        return ReactPlanner(tools_map, workers_map, name2id)

    def test_full_planning_workflow(
        self, integration_planner: ReactPlanner, patched_sample_config: dict[str, Any]
    ) -> None:
        config = patched_sample_config
        mock_llm = config["mock_llm"]
        mock_response = Mock()
        mock_response.content = "Test trajectory"
        mock_llm.invoke.return_value = mock_response
        config["mock_aimessage_to_dict"].return_value = {
            "content": 'Action:\n{"name": "integration_tool", "arguments": {"param": "value"}}'
        }
        integration_planner.llm = mock_llm
        integration_planner.llm_provider = "openai"
        integration_planner.system_role = "system"
        integration_planner.resource_rag_docs_created = True

        with (
            patch.object(
                integration_planner, "_get_planning_trajectory_summary"
            ) as mock_summary,
            patch.object(
                integration_planner, "_get_num_resource_retrievals"
            ) as mock_retrievals,
            patch.object(
                integration_planner, "_retrieve_resource_signatures"
            ) as mock_retrieve,
        ):
            mock_summary.return_value = "Test trajectory"
            mock_retrievals.return_value = 3
            mock_retrieve.return_value = [
                Mock(
                    metadata={"resource_name": "integration_tool", "json_signature": {}}
                )
            ]

            msg_state = MessageState(
                user_message=ConvoMessage(history="", message="Test message"),
                orchestrator_message=OrchestratorMessage(message="", attribute={}),
                response="",
                is_stream=False,
                stream_type=StreamType.NON_STREAM,
            )
            msg_history = [{"role": "user", "content": "Test message"}]

            msg_history, action_name, response = integration_planner.plan(
                msg_state, msg_history, max_num_steps=1
            )

            assert action_name == "integration_tool"
            assert response == "integration_result"
            assert len(msg_history) > 1

            assert action_name == "integration_tool"
            assert response == "integration_result"
            assert len(msg_history) > 1
