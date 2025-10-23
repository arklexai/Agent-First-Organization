import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import requests

from integration_tests.utils.base import BaseTestOrchestrator, ChatRole


def create_mock_response(url: str, method: str, **kwargs: dict[str, Any]) -> Mock:
    mock_response = Mock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()

    request_body = kwargs.get("json", {})

    if url == "https://api.arklex.test/service":
        if method == "GET":
            # Mock response for queryService
            json_data = {
                "services": [
                    {
                        "name": "graph-based chatbots",
                        "description": "Advanced conversational AI using graph structures",
                        "category": "AI Solutions",
                    },
                    {
                        "name": "agents",
                        "description": "Intelligent autonomous agents for various tasks",
                        "category": "AI Solutions",
                    },
                    {
                        "name": "user simulator",
                        "description": "Simulation tools for user behavior modeling",
                        "category": "Testing Tools",
                    },
                ]
            }
        elif method == "POST":
            # Mock response for contactTeam
            json_data = {
                "status": "success",
                "message": "Service request submitted successfully",
                "request_id": "REQ-12345",
                "submitted_services": request_body.get("service", []),
                "total_budget": sum(
                    item["budget"] for item in request_body.get("service", [])
                ),
            }
        else:
            # Unsupported method
            json_data = {
                "error": "Method Not Allowed",
                "message": f"Method {method} not supported",
            }
            mock_response.status_code = 405
    else:
        # Unknown endpoint
        json_data = {"error": "Not Found", "message": f"Endpoint {url} not found"}
        mock_response.status_code = 404

    mock_response.json.return_value = json_data
    mock_response.text = json.dumps(json_data)
    return mock_response


def create_mock_agent_result(trajectory: list[dict[str, Any]]) -> AsyncMock:
    from agents import MessageOutputItem, ToolCallItem, ToolCallOutputItem

    user_messages = [msg for msg in trajectory if msg.get("role") == "user"]
    last_user_message = user_messages[-1]["content"] if user_messages else ""

    mock_result = AsyncMock()
    mock_agent = Mock()
    mock_agent.name = "Arklex Agent"

    if last_user_message == "What services does your company provide?":
        mock_tool_call = Mock(spec=ToolCallItem)
        mock_tool_call.agent = mock_agent
        mock_tool_call.tool_call_id = f"call_{str(uuid.uuid4()).replace('-', '')}"
        mock_tool_call.function_name = "queryService"
        mock_tool_call.arguments = "{}"

        mock_tool_output = Mock(spec=ToolCallOutputItem)
        mock_tool_output.agent = mock_agent
        mock_tool_output.tool_call_id = mock_tool_call.tool_call_id
        mock_tool_output.output = json.dumps(
            {
                "services": [
                    {
                        "name": "graph-based chatbots",
                        "description": "Advanced conversational AI using graph structures",
                        "category": "AI Solutions",
                    },
                    {
                        "name": "agents",
                        "description": "Intelligent autonomous agents for various tasks",
                        "category": "AI Solutions",
                    },
                    {
                        "name": "user simulator",
                        "description": "Simulation tools for user behavior modeling",
                        "category": "Testing Tools",
                    },
                ]
            }
        )

        mock_message = Mock(spec=MessageOutputItem)
        mock_message.agent = mock_agent
        # Create raw_item with content structure expected by ItemHelpers.text_message_output
        mock_raw_item = Mock()
        mock_raw_item.content = [
            Mock(
                text="Arklex provides graph-based chatbots, agents, and user simulators."
            )
        ]
        mock_message.raw_item = mock_raw_item
        mock_message.content = [
            Mock(
                text="Arklex provides graph-based chatbots, agents, and user simulators."
            )
        ]

        mock_result.new_items = [mock_tool_call, mock_tool_output, mock_message]
        mock_result.to_input_list.return_value = trajectory + [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": mock_tool_call.tool_call_id,
                        "type": "function",
                        "function": {"name": "queryService", "arguments": "{}"},
                    }
                ],
                "content": None,
            },
            {
                "role": "tool",
                "tool_call_id": mock_tool_call.tool_call_id,
                "content": mock_tool_output.output,
            },
            {
                "role": "assistant",
                "content": "Arklex provides graph-based chatbots, agents, and user simulators.",
            },
        ]

    elif last_user_message == "I'm interested in user simulator with a budget of 1000":
        tool_call_id = f"call_{str(uuid.uuid4()).replace('-', '')}"

        mock_tool_call = Mock(spec=ToolCallItem)
        mock_tool_call.agent = mock_agent
        mock_tool_call.tool_call_id = tool_call_id
        mock_tool_call.function_name = "contactTeam"
        mock_tool_call.arguments = (
            '{"ServiceInfo":[{"service":"user simulator","budget":1000}]}'
        )

        mock_tool_output = Mock(spec=ToolCallOutputItem)
        mock_tool_output.agent = mock_agent
        mock_tool_output.tool_call_id = tool_call_id
        mock_tool_output.output = json.dumps(
            {
                "status": "success",
                "message": "Service request submitted successfully",
                "request_id": "REQ-12345",
            }
        )

        mock_message = Mock(spec=MessageOutputItem)
        mock_message.agent = mock_agent
        # Create raw_item with content structure expected by ItemHelpers.text_message_output
        mock_raw_item = Mock()
        mock_raw_item.content = [
            Mock(
                text="Service request for user simulator with a budget of 1000 submitted successfully."
            )
        ]
        mock_message.raw_item = mock_raw_item
        mock_message.content = [
            Mock(
                text="Service request for user simulator with a budget of 1000 submitted successfully."
            )
        ]

        mock_result.new_items = [mock_tool_call, mock_tool_output, mock_message]
        mock_result.to_input_list.return_value = trajectory + [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "contactTeam",
                            "arguments": mock_tool_call.arguments,
                        },
                    }
                ],
                "content": None,
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": mock_tool_output.output,
            },
            {
                "role": "assistant",
                "content": "Service request for user simulator with a budget of 1000 submitted successfully.",
            },
        ]
    else:
        raise ValueError(f"Unknown request type: {last_user_message}")

    return mock_result


@patch("arklex.env.tools.custom_tools.http_tool.requests.request")
@patch("arklex.env.agents.openai_agent.Runner.run")
async def test_http_tool_agent(mock_runner_run: Mock, mock_request: Mock) -> None:
    mock_request.side_effect = create_mock_response

    async def mock_run(agent: object, trajectory: list[dict[str, Any]]) -> AsyncMock:
        return create_mock_agent_result(trajectory)

    mock_runner_run.side_effect = mock_run

    orchestrator = BaseTestOrchestrator(
        "integration_tests/taskgraphs/http_tool_agent_taskgraph.json"
    )
    params = BaseTestOrchestrator.init_params()
    chat_history, params = params["chat_history"], params["parameters"]

    # Test: Agent start message (handled by agent, no tool call)
    text = "<start>"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    assert (
        output["answer"]
        == "This is agent developed by Arklex, how can I assist you today?"
    )

    # Test Case 1: HTTP tool (queryService)
    text = "What services does your company provide?"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]

    assert mock_request.call_count >= 1
    get_calls = [
        call for call in mock_request.call_args_list if call[1].get("method") == "GET"
    ]
    assert len(get_calls) == 1
    get_call = get_calls[0]
    assert get_call[1]["url"] == "https://api.arklex.test/service"
    assert get_call[1]["method"] == "GET"
    assert get_call[1]["headers"]["Authorization"] == "Bearer test-token"
    assert get_call[1]["headers"]["Content-Type"] == "application/json"

    response_text = output["answer"]
    assert (
        response_text
        == "Arklex provides graph-based chatbots, agents, and user simulators."
    )

    # Test Case 2: HTTP tool (contactTeam)
    text = "I'm interested in user simulator with a budget of 1000"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]

    post_calls = [
        call for call in mock_request.call_args_list if call[1].get("method") == "POST"
    ]
    assert len(post_calls) >= 1
    post_call = post_calls[0]
    assert post_call[1]["url"] == "https://api.arklex.test/service"
    assert post_call[1]["method"] == "POST"
    assert post_call[1]["headers"]["Authorization"] == "Bearer test-token"
    assert post_call[1]["headers"]["Content-Type"] == "application/json"
    assert post_call[1]["json"] == {
        "service": [
            {"service": "user simulator", "budget": 1000},
        ],
    }

    # Verify the agent's final response
    assert (
        output["answer"]
        == "Service request for user simulator with a budget of 1000 submitted successfully."
    )
