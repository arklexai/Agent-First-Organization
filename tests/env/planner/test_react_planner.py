"""Comprehensive tests for the ReactPlanner module."""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Tuple

from arklex.env.planner.react_planner import (
    ReactPlanner,
    DefaultPlanner,
    Action,
    EnvResponse,
    PlannerResource,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_RESOURCE,
    NUM_STEPS_TO_NUM_RETRIEVALS,
    MIN_NUM_RETRIEVALS,
    MAX_NUM_RETRIEVALS,
)
from arklex.utils.graph_state import (
    MessageState,
    LLMConfig,
    ConvoMessage,
    OrchestratorMessage,
)
from arklex.types import StreamType


class TestReactPlanner:
    """Test cases for ReactPlanner class."""

    @pytest.fixture
    def mock_tools_map(self) -> Dict[str, Any]:
        """Create a mock tools map for testing."""
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
                "execute": Mock(
                    return_value=Mock(func=Mock(return_value="tool_result"))
                ),
            }
        }

    @pytest.fixture
    def mock_workers_map(self) -> Dict[str, Any]:
        """Create a mock workers map for testing."""
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
    def mock_name2id(self) -> Dict[str, int]:
        """Create a mock name2id mapping for testing."""
        return {"test_tool": 1, "test_worker": 2, "MessageWorker": 3}

    @pytest.fixture
    def mock_llm_config(self) -> LLMConfig:
        """Create a mock LLM config for testing."""
        return LLMConfig(model_type_or_path="gpt-3.5-turbo", llm_provider="openai")

    @pytest.fixture
    def mock_message_state(self) -> MessageState:
        """Create a mock message state for testing."""
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
    def mock_msg_history(self) -> List[Dict[str, Any]]:
        """Create a mock message history for testing."""
        return [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

    @pytest.fixture
    def react_planner(
        self, mock_tools_map, mock_workers_map, mock_name2id
    ) -> ReactPlanner:
        """Create a ReactPlanner instance for testing."""
        return ReactPlanner(mock_tools_map, mock_workers_map, mock_name2id)

    def test_react_planner_initialization(
        self, mock_tools_map, mock_workers_map, mock_name2id
    ) -> None:
        """Test ReactPlanner initialization."""
        planner = ReactPlanner(mock_tools_map, mock_workers_map, mock_name2id)

        assert planner.tools_map == mock_tools_map
        assert planner.workers_map == mock_workers_map
        assert planner.name2id == mock_name2id
        assert RESPOND_ACTION_NAME in planner.all_resources_info
        assert "test_tool" in planner.all_resources_info
        assert "test_worker" in planner.all_resources_info
        # MessageWorker should be filtered out
        assert "MessageWorker" not in planner.all_resources_info
        assert planner.resource_rag_docs_created is False

    def test_react_planner_description(self) -> None:
        """Test ReactPlanner description."""
        assert (
            ReactPlanner.description
            == "Choose tools/workers based on task and chat records if there is no specific worker/node for the user's query"
        )

    def test_default_planner_description(self) -> None:
        """Test DefaultPlanner description."""
        assert (
            DefaultPlanner.description
            == "Default planner that returns unaltered MessageState on execute()"
        )

    def test_format_worker_info(self, react_planner, mock_workers_map) -> None:
        """Test _format_worker_info method."""
        formatted_info = react_planner._format_worker_info(mock_workers_map)

        assert "test_worker" in formatted_info
        assert "MessageWorker" not in formatted_info  # Should be filtered out

        worker_info = formatted_info["test_worker"]
        assert isinstance(worker_info, PlannerResource)
        assert worker_info.name == "test_worker"
        assert worker_info.type == "worker"
        assert worker_info.description == "A test worker for testing"

    def test_format_tool_info(self, react_planner, mock_tools_map) -> None:
        """Test _format_tool_info method."""
        formatted_info = react_planner._format_tool_info(mock_tools_map)

        assert "test_tool" in formatted_info

        tool_info = formatted_info["test_tool"]
        assert isinstance(tool_info, PlannerResource)
        assert tool_info.name == "test_tool"
        assert tool_info.type == "tool"
        assert tool_info.description == "A test tool for testing"
        assert len(tool_info.parameters) == 2
        assert tool_info.required == ["param1"]

    def test_format_tool_info_with_mock_tool(self, react_planner) -> None:
        """Test _format_tool_info with MockTool instance."""
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

    def test_create_resource_rag_docs(self, react_planner) -> None:
        """Test _create_resource_rag_docs method."""
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

    def test_parse_trajectory_summary_to_steps(self, react_planner) -> None:
        """Test _parse_trajectory_summary_to_steps method."""
        summary = (
            "- Step 1: Do something\n- Step 2: Do something else\n- Step 3: Finish"
        )

        steps = react_planner._parse_trajectory_summary_to_steps(summary)

        assert len(steps) == 3
        assert steps[0] == "Step 1: Do something"
        assert steps[1] == "Step 2: Do something else"
        assert steps[2] == "Step 3: Finish"

    def test_parse_trajectory_summary_to_steps_empty(self, react_planner) -> None:
        """Test _parse_trajectory_summary_to_steps with empty summary."""
        summary = ""
        steps = react_planner._parse_trajectory_summary_to_steps(summary)
        assert steps == []

    def test_get_num_resource_retrievals_valid_summary(self, react_planner) -> None:
        """Test _get_num_resource_retrievals with valid summary."""
        summary = "- Step 1\n- Step 2\n- Step 3"

        n_retrievals = react_planner._get_num_resource_retrievals(summary)

        expected = NUM_STEPS_TO_NUM_RETRIEVALS(3)
        expected = min(max(expected, MIN_NUM_RETRIEVALS), MAX_NUM_RETRIEVALS)
        assert n_retrievals == expected

    def test_get_num_resource_retrievals_invalid_summary(self, react_planner) -> None:
        """Test _get_num_resource_retrievals with invalid summary."""
        summary = "Invalid summary without steps"

        n_retrievals = react_planner._get_num_resource_retrievals(summary)

        assert n_retrievals == 4

    def test_get_num_resource_retrievals_empty_summary(self, react_planner) -> None:
        """Test _get_num_resource_retrievals with empty summary."""
        summary = ""

        n_retrievals = react_planner._get_num_resource_retrievals(summary)

        assert n_retrievals == MIN_NUM_RETRIEVALS

    def test_get_num_resource_retrievals_bounds(self, react_planner) -> None:
        """Test _get_num_resource_retrievals respects bounds."""
        # Test with very large number of steps
        summary = "- " + "\n- ".join([f"Step {i}" for i in range(100)])

        n_retrievals = react_planner._get_num_resource_retrievals(summary)

        assert n_retrievals == MAX_NUM_RETRIEVALS

    def test_parse_response_action_to_json_valid(self, react_planner) -> None:
        """Test _parse_response_action_to_json with valid JSON."""
        response = (
            'Some text\nAction:\n{"name": "test_tool", "arguments": {"param": "value"}}'
        )

        result = react_planner._parse_response_action_to_json(response)

        assert result["name"] == "test_tool"
        assert result["arguments"]["param"] == "value"

    def test_parse_response_action_to_json_invalid(self, react_planner) -> None:
        """Test _parse_response_action_to_json with invalid JSON."""
        response = "Some text\nAction:\nInvalid JSON"

        result = react_planner._parse_response_action_to_json(response)

        assert result["name"] == RESPOND_ACTION_NAME
        assert result["arguments"]["content"] == "Invalid JSON"

    def test_message_to_actions_valid_tool(self, react_planner) -> None:
        """Test message_to_actions with valid tool."""
        message = {"name": "test_tool", "arguments": {"param1": "value1"}}

        actions = react_planner.message_to_actions(message)

        assert len(actions) == 1
        assert actions[0].name == "test_tool"
        assert actions[0].kwargs == {"param1": "value1"}

    def test_message_to_actions_valid_worker(self, react_planner) -> None:
        """Test message_to_actions with valid worker."""
        message = {"name": "test_worker", "arguments": {}}

        actions = react_planner.message_to_actions(message)

        assert len(actions) == 1
        assert actions[0].name == "test_worker"
        assert actions[0].kwargs == {}

    def test_message_to_actions_invalid_resource(self, react_planner) -> None:
        """Test message_to_actions with invalid resource."""
        message = {
            "name": "invalid_resource",
            "arguments": {"content": "Error message"},
        }

        actions = react_planner.message_to_actions(message)

        assert len(actions) == 1
        assert actions[0].name == RESPOND_ACTION_NAME
        assert actions[0].kwargs["content"] == "Error message"

    def test_message_to_actions_missing_arguments(self, react_planner) -> None:
        """Test message_to_actions with missing arguments."""
        message = {"name": "invalid_resource", "content": "Direct content"}

        actions = react_planner.message_to_actions(message)

        assert len(actions) == 1
        assert actions[0].name == RESPOND_ACTION_NAME
        assert actions[0].kwargs["content"] == "Direct content"

    def test_step_respond_action(self, react_planner, mock_message_state) -> None:
        """Test step method with RESPOND_ACTION."""
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": "Hello user!"})

        response = react_planner.step(action, mock_message_state)

        assert isinstance(response, EnvResponse)
        assert response.observation == "Hello user!"

    def test_step_tool_execution(self, react_planner, mock_message_state) -> None:
        """Test step method with tool execution."""
        action = Action(name="test_tool", kwargs={"param1": "value1"})

        response = react_planner.step(action, mock_message_state)

        assert isinstance(response, EnvResponse)
        assert response.observation == "tool_result"

    def test_step_worker_execution(self, react_planner, mock_message_state) -> None:
        """Test step method with worker execution."""
        action = Action(name="test_worker", kwargs={})

        response = react_planner.step(action, mock_message_state)

        assert isinstance(response, EnvResponse)
        assert response.observation == "worker_result"

    def test_step_unknown_action(self, react_planner, mock_message_state) -> None:
        """Test step method with unknown action."""
        action = Action(name="unknown_action", kwargs={})

        response = react_planner.step(action, mock_message_state)

        assert isinstance(response, EnvResponse)
        assert "Unknown action unknown_action" in response.observation

    def test_step_tool_execution_error(self, react_planner, mock_message_state) -> None:
        """Test step method with tool execution error."""
        # Mock tool to raise an exception
        react_planner.tools_map[1]["execute"] = Mock(
            return_value=Mock(func=Mock(side_effect=Exception("Tool error")))
        )

        action = Action(name="test_tool", kwargs={"param1": "value1"})

        response = react_planner.step(action, mock_message_state)

        assert isinstance(response, EnvResponse)
        assert "Error: Tool error" in response.observation

    def test_step_worker_execution_error(
        self, react_planner, mock_message_state
    ) -> None:
        """Test step method with worker execution error."""
        # Mock worker to raise an exception
        react_planner.workers_map["test_worker"]["execute"] = Mock(
            return_value=Mock(execute=Mock(side_effect=Exception("Worker error")))
        )

        action = Action(name="test_worker", kwargs={})

        response = react_planner.step(action, mock_message_state)

        assert isinstance(response, EnvResponse)
        assert "Error: Worker error" in response.observation

    @patch("arklex.env.planner.react_planner.PROVIDER_MAP")
    @patch("arklex.env.planner.react_planner.ChatOpenAI")
    def test_set_llm_config_and_build_resource_library(
        self, mock_chat_openai, mock_provider_map, react_planner, mock_llm_config
    ) -> None:
        """Test set_llm_config_and_build_resource_library method."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        mock_provider_map.get.return_value = Mock(return_value=mock_llm)

        # Mock embedding model and FAISS
        with (
            patch(
                "arklex.env.planner.react_planner.PROVIDER_EMBEDDING_MODELS"
            ) as mock_embedding_models,
            patch(
                "arklex.env.planner.react_planner.PROVIDER_EMBEDDINGS"
            ) as mock_provider_embeddings,
            patch(
                "arklex.env.planner.react_planner.OpenAIEmbeddings"
            ) as mock_openai_embeddings,
            patch("arklex.env.planner.react_planner.FAISS") as mock_faiss,
        ):
            mock_embedding_models.__getitem__.return_value = "text-embedding-ada-002"
            mock_provider_embeddings.get.return_value = Mock(return_value=Mock())
            mock_openai_embeddings.return_value = Mock()
            mock_faiss.from_documents.return_value = Mock(
                as_retriever=Mock(return_value=Mock())
            )

            react_planner.set_llm_config_and_build_resource_library(mock_llm_config)

            assert react_planner.llm_config == mock_llm_config
            assert react_planner.resource_rag_docs_created is True

    @patch("arklex.env.planner.react_planner.PLANNER_SUMMARIZE_TRAJECTORY_PROMPT")
    @patch("arklex.env.planner.react_planner.PromptTemplate")
    def test_get_planning_trajectory_summary(
        self,
        mock_prompt_template,
        mock_prompt,
        react_planner,
        mock_message_state,
        mock_msg_history,
    ) -> None:
        """Test _get_planning_trajectory_summary method."""
        mock_prompt_template.from_template.return_value = Mock(
            invoke=Mock(return_value=Mock(text="System prompt"))
        )

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(content="Trajectory summary")
        react_planner.llm = mock_llm
        react_planner.llm_provider = "openai"
        react_planner.system_role = "system"

        summary = react_planner._get_planning_trajectory_summary(
            mock_message_state, mock_msg_history
        )

        assert summary == "Trajectory summary"
        mock_llm.invoke.assert_called_once()

    @patch("arklex.env.planner.react_planner.FAISS")
    def test_retrieve_resource_signatures(self, mock_faiss, react_planner) -> None:
        """Test _retrieve_resource_signatures method."""
        # Mock the retriever
        mock_retriever = Mock()
        mock_retriever.vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        react_planner.retriever = mock_retriever
        react_planner.guaranteed_retrieval_docs = [
            Mock(metadata={"resource_name": RESPOND_ACTION_NAME})
        ]

        docs = react_planner._retrieve_resource_signatures(
            n_retrievals=3,
            trajectory_summary="Test summary",
            user_message="Hello",
            task="Test task",
        )

        assert len(docs) == 2  # 1 retrieved + 1 guaranteed
        mock_retriever.vectorstore.similarity_search_with_score.assert_called_once()

    @patch("arklex.env.planner.react_planner.PLANNER_REACT_INSTRUCTION_FEW_SHOT")
    @patch("arklex.env.planner.react_planner.PromptTemplate")
    def test_plan_method(
        self,
        mock_prompt_template,
        mock_prompt,
        react_planner,
        mock_message_state,
        mock_msg_history,
    ) -> None:
        """Test plan method."""
        # Mock all the dependencies
        mock_prompt_template.from_template.return_value = Mock(
            invoke=Mock(return_value=Mock(text="System prompt"))
        )

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content='Action:\n{"name": "test_tool", "arguments": {"param": "value"}}'
        )
        react_planner.llm = mock_llm
        react_planner.llm_provider = "openai"
        react_planner.system_role = "system"
        react_planner.resource_rag_docs_created = True

        # Mock the trajectory summary and retrieval methods
        with (
            patch.object(
                react_planner, "_get_planning_trajectory_summary"
            ) as mock_summary,
            patch.object(
                react_planner, "_get_num_resource_retrievals"
            ) as mock_retrievals,
            patch.object(
                react_planner, "_retrieve_resource_signatures"
            ) as mock_retrieve,
            patch.object(react_planner, "step") as mock_step,
        ):
            mock_summary.return_value = "Test trajectory"
            mock_retrievals.return_value = 3
            mock_retrieve.return_value = [
                Mock(metadata={"resource_name": "test_tool", "json_signature": {}})
            ]
            mock_step.return_value = EnvResponse(observation="tool_result")

            msg_history, action_name, response = react_planner.plan(
                mock_message_state, mock_msg_history, max_num_steps=1
            )

            assert action_name == "test_tool"
            assert response == "tool_result"
            assert len(msg_history) > 2  # Should have added new messages

    def test_plan_method_with_respond_action(
        self, react_planner, mock_message_state, mock_msg_history
    ) -> None:
        """Test plan method that returns RESPOND_ACTION."""
        # Mock LLM to return RESPOND_ACTION
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            content=f'Action:\n{{"name": "{RESPOND_ACTION_NAME}", "arguments": {{"content": "Hello!"}}}}'
        )
        react_planner.llm = mock_llm
        react_planner.llm_provider = "openai"
        react_planner.system_role = "system"
        react_planner.resource_rag_docs_created = True

        # Mock the trajectory summary and retrieval methods
        with (
            patch.object(
                react_planner, "_get_planning_trajectory_summary"
            ) as mock_summary,
            patch.object(
                react_planner, "_get_num_resource_retrievals"
            ) as mock_retrievals,
            patch.object(
                react_planner, "_retrieve_resource_signatures"
            ) as mock_retrieve,
            patch.object(react_planner, "step") as mock_step,
            patch(
                "arklex.env.planner.react_planner.PromptTemplate"
            ) as mock_prompt_template,
        ):
            mock_summary.return_value = "Test trajectory"
            mock_retrievals.return_value = 3
            mock_retrieve.return_value = [
                Mock(
                    metadata={
                        "resource_name": RESPOND_ACTION_NAME,
                        "json_signature": {},
                    }
                )
            ]
            mock_step.return_value = EnvResponse(observation="Hello!")
            mock_prompt_template.from_template.return_value = Mock(
                invoke=Mock(return_value=Mock(text="System prompt"))
            )

            msg_history, action_name, response = react_planner.plan(
                mock_message_state, mock_msg_history, max_num_steps=1
            )

            assert action_name == RESPOND_ACTION_NAME
            assert response == "Hello!"

    def test_execute_method(
        self, react_planner, mock_message_state, mock_msg_history
    ) -> None:
        """Test execute method."""
        # Mock the plan method
        with patch.object(react_planner, "plan") as mock_plan:
            mock_plan.return_value = (mock_msg_history, "test_action", "test_response")

            action, updated_state, updated_history = react_planner.execute(
                mock_message_state, mock_msg_history
            )

            assert action == "test_action"
            assert updated_state.response == "test_response"
            assert updated_history == mock_msg_history
            mock_plan.assert_called_once_with(mock_message_state, mock_msg_history)

    def test_default_planner_execute(
        self, mock_tools_map, mock_workers_map, mock_name2id
    ) -> None:
        """Test DefaultPlanner execute method."""
        planner = DefaultPlanner(mock_tools_map, mock_workers_map, mock_name2id)
        msg_state = Mock()
        msg_history = []

        action, updated_state, updated_history = planner.execute(msg_state, msg_history)

        assert action["name"] == RESPOND_ACTION_NAME
        assert action["kwargs"]["content"] == ""
        assert updated_state == msg_state
        assert updated_history == msg_history

    def test_action_model(self) -> None:
        """Test Action model."""
        action = Action(name="test_action", kwargs={"param": "value"})

        assert action.name == "test_action"
        assert action.kwargs == {"param": "value"}

    def test_env_response_model(self) -> None:
        """Test EnvResponse model."""
        response = EnvResponse(observation="test_observation")

        assert response.observation == "test_observation"

    def test_planner_resource_model(self) -> None:
        """Test PlannerResource model."""
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
        """Test RESPOND_ACTION_RESOURCE constant."""
        assert RESPOND_ACTION_RESOURCE.name == RESPOND_ACTION_NAME
        assert RESPOND_ACTION_RESOURCE.type == "worker"
        assert "Respond to the user" in RESPOND_ACTION_RESOURCE.description

    def test_num_steps_to_num_retrievals_function(self) -> None:
        """Test NUM_STEPS_TO_NUM_RETRIEVALS function."""
        assert NUM_STEPS_TO_NUM_RETRIEVALS(1) == 4
        assert NUM_STEPS_TO_NUM_RETRIEVALS(5) == 8
        assert NUM_STEPS_TO_NUM_RETRIEVALS(10) == 13


class TestReactPlannerIntegration:
    """Integration tests for ReactPlanner."""

    @pytest.fixture
    def integration_planner(self) -> ReactPlanner:
        """Create a ReactPlanner instance for integration testing."""
        tools_map = {
            1: {
                "info": {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Perform mathematical calculations",
                        "parameters": {
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Mathematical expression",
                                }
                            },
                            "required": ["expression"],
                        },
                    },
                },
                "output": [{"name": "result", "description": "Calculation result"}],
                "fixed_args": {},
                "execute": Mock(return_value=Mock(func=Mock(return_value="42"))),
            }
        }

        workers_map = {
            "greeter": {
                "description": "Greet the user",
                "execute": Mock(return_value=Mock(execute=Mock(return_value="Hello!"))),
            }
        }

        name2id = {"calculator": 1, "greeter": 2}

        return ReactPlanner(tools_map, workers_map, name2id)

    def test_full_planning_workflow(self, integration_planner) -> None:
        """Test a complete planning workflow."""
        msg_state = MessageState(
            user_message=ConvoMessage(history="", message="Calculate 2 + 2"),
            orchestrator_message=OrchestratorMessage(
                message="Calculation request", attribute={"task": "calculation"}
            ),
            response="",
            is_stream=False,
            stream_type=StreamType.NON_STREAM,
        )

        msg_history = []

        # Mock the LLM responses
        with (
            patch.object(integration_planner, "llm") as mock_llm,
            patch.object(
                integration_planner, "_get_planning_trajectory_summary"
            ) as mock_summary,
            patch.object(
                integration_planner, "_retrieve_resource_signatures"
            ) as mock_retrieve,
            patch(
                "arklex.env.planner.react_planner.PromptTemplate"
            ) as mock_prompt_template,
        ):
            mock_summary.return_value = "- Use calculator to compute the result"
            mock_retrieve.return_value = [
                Mock(metadata={"resource_name": "calculator", "json_signature": {}})
            ]
            mock_prompt_template.from_template.return_value = Mock(
                invoke=Mock(return_value=Mock(text="System prompt"))
            )

            # Mock LLM to return calculator action
            mock_llm.invoke.return_value = Mock(
                content='Action:\n{"name": "calculator", "arguments": {"expression": "2 + 2"}}'
            )

            msg_history, action_name, response = integration_planner.plan(
                msg_state, msg_history, max_num_steps=1
            )

            assert action_name == "calculator"
            assert response == "42"
            assert len(msg_history) > 0
