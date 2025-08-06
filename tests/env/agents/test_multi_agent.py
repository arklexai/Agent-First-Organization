from collections.abc import Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from arklex.env.agents.multi_agent import MultiAgent
from arklex.orchestrator.entities.msg_state_entities import MessageState, StatusEnum


@pytest.fixture
def mock_state() -> MessageState:
    state = Mock(spec=MessageState)
    state.status = StatusEnum.INCOMPLETE
    state.function_calling_trajectory = []
    state.message_flow = ""
    state.response = ""
    state.sys_instruct = "System instructions"

    # Bot config
    state.bot_config = Mock()
    state.bot_config.llm_config = Mock()
    state.bot_config.llm_config.llm_provider = "openai"
    state.bot_config.llm_config.model_type_or_path = "gpt-4.0-mini"

    return state


@pytest.fixture
def dummy_config() -> dict:
    return {
        "node_specific_data": {"pattern": "dummy_pattern"},
        "sub_agents": [],
    }


@pytest.fixture
def mock_graph() -> Generator[Mock, None, None]:
    """Mock graph with compile and invoke methods."""
    compiled = Mock()
    compiled.invoke.return_value = {"result": "sync_result"}
    compiled.model_dump.return_value = {"result": "sync_result"}
    graph = Mock()
    graph.compile.return_value = compiled
    return graph


@pytest.fixture
def mock_graph_async() -> Generator[Mock, None, None]:
    """Mock graph that supports ainvoke."""
    compiled = AsyncMock()
    compiled.ainvoke.return_value = {"result": "async_result"}
    compiled.model_dump.return_value = {"result": "async_result"}
    graph = Mock()
    graph.compile.return_value = compiled
    return graph


@patch("arklex.env.agents.multi_agent.dispatch_pattern")
def test_multi_agent_sync_execution(
    mock_dispatch: Mock,
    mock_state: Mock,
    dummy_config: dict,
    mock_graph: Mock,
) -> None:
    """Test synchronous execution of MultiAgent."""
    mock_dispatch.return_value = mock_graph

    agent = MultiAgent(
        successors=[],
        predecessors=[],
        tools=[],
        state=mock_state,
        multi_agent_config=dummy_config,
    )

    result = agent._execute(mock_state)

    assert result == {"result": "sync_result"}
    mock_graph.compile.assert_called_once()
    mock_graph.compile().invoke.assert_called_once_with(mock_state)


@patch("arklex.env.agents.multi_agent.dispatch_pattern")
@pytest.mark.asyncio
async def test_multi_agent_async_execution(
    mock_dispatch: Mock,
    mock_state: Mock,
    dummy_config: dict,
    mock_graph_async: Mock,
) -> None:
    """Test asynchronous execution with ainvoke support."""
    dummy_config["node_specific_data"]["is_async"] = True
    mock_dispatch.return_value = mock_graph_async

    agent = MultiAgent(
        successors=[],
        predecessors=[],
        tools=[],
        state=mock_state,
        multi_agent_config=dummy_config,
    )

    result = await agent._async_execute(mock_state)

    assert result == {"result": "async_result"}
    mock_graph_async.compile.assert_called_once()
    mock_graph_async.compile().ainvoke.assert_awaited_once_with(mock_state)


@patch("arklex.env.agents.multi_agent.dispatch_pattern")
@pytest.mark.asyncio
async def test_multi_agent_async_falls_back_to_sync(
    mock_dispatch: Mock,
    mock_state: Mock,
    dummy_config: dict,
    mock_graph: Mock,
) -> None:
    """Test fallback to sync execution when ainvoke is not available."""
    dummy_config["node_specific_data"]["is_async"] = True
    mock_dispatch.return_value = mock_graph

    agent = MultiAgent(
        successors=[],
        predecessors=[],
        tools=[],
        state=mock_state,
        multi_agent_config=dummy_config,
    )

    result = await agent._async_execute(mock_state)

    assert result == {"result": "sync_result"}
    mock_graph.compile.assert_called_once()
    mock_graph.compile().invoke.assert_called_once_with(mock_state)


def test_multi_agent_is_async_true(mock_state: Mock) -> None:
    config = {"node_specific_data": {"type": "agents_as_tools", "is_async": True}}
    agent = MultiAgent(
        successors=[],
        predecessors=[],
        tools=[],
        state=mock_state,
        multi_agent_config=config,
    )
    assert agent.is_async() is True


def test_multi_agent_is_async_false_when_missing(mock_state: Mock) -> None:
    # is_async not provided
    config = {"node_specific_data": {"type": "agents_as_tools"}}
    agent = MultiAgent(
        successors=[],
        predecessors=[],
        tools=[],
        state=mock_state,
        multi_agent_config=config,
    )
    assert agent.is_async() is False


def test_multi_agent_raises_when_config_missing(mock_state: Mock) -> None:
    with pytest.raises(ValueError, match="MultiAgent config not found"):
        MultiAgent(
            successors=[],
            predecessors=[],
            tools=[],
            state=mock_state,
            multi_agent_config=None,  # Triggers the ValueError
        )


def test_multi_agent_execute_exception_handling(mock_state: Mock) -> None:
    """Test that _execute handles exceptions gracefully."""
    mock_workflow = Mock()
    mock_workflow.compile.side_effect = Exception("compile error")

    with patch("arklex.env.agents.multi_agent.dispatch_pattern") as mock_dispatch:
        mock_dispatch.return_value = mock_workflow
        agent = MultiAgent(
            successors=[],
            predecessors=[],
            tools=[],
            state=mock_state,
            multi_agent_config={"node_specific_data": {}},
        )

        result = agent._execute(mock_state)

        assert "[MultiAgent Error]" in mock_state.response
        assert result == mock_state.model_dump()


@pytest.mark.asyncio
async def test_multi_agent_async_execute_exception_handling(mock_state: Mock) -> None:
    """Test that _async_execute handles exceptions gracefully."""
    mock_workflow = Mock()
    mock_workflow.compile.side_effect = Exception("compile error")

    with patch("arklex.env.agents.multi_agent.dispatch_pattern") as mock_dispatch:
        mock_dispatch.return_value = mock_workflow
        agent = MultiAgent(
            successors=[],
            predecessors=[],
            tools=[],
            state=mock_state,
            multi_agent_config={"node_specific_data": {}},
        )

        result = await agent._async_execute(mock_state)

        assert "[MultiAgent Error]" in mock_state.response
        assert result == mock_state.model_dump()
