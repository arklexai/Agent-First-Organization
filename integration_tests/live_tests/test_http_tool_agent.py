import json
from typing import Any
from unittest.mock import Mock, patch

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
                "submitted_services": request_body["service"],
                "total_budget": sum(item["budget"] for item in request_body["service"]),
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


@patch("arklex.resources.tools.custom_tools.http_tool.requests.request")
async def test_http_tool_agent(mock_request: Mock) -> None:
    orchestrator = BaseTestOrchestrator(
        "integration_tests/taskgraphs/http_tool_agent_taskgraph.json"
    )
    params = BaseTestOrchestrator.init_params()
    chat_history, params = params["chat_history"], params["parameters"]

    # agent start message
    text = "<start>"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    assert (
        output["answer"]
        == "This is agent developed by Arklex, how can I assist you today?"
    )

    # Test case 1: HTTP tool (queryService)
    mock_request.side_effect = create_mock_response
    text = "What services does your company provide?"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]

    assert mock_request.call_count == 1
    get_calls = [
        call for call in mock_request.call_args_list if call[1].get("method") == "GET"
    ]
    assert len(get_calls) == 1
    get_call = get_calls[0]
    assert get_call[1]["url"] == "https://api.arklex.test/service"
    assert get_call[1]["method"] == "GET"
    assert get_call[1]["headers"]["Authorization"] == "Bearer test-token"
    assert get_call[1]["headers"]["Content-Type"] == "application/json"

    response_text = output["answer"].lower()
    assert "graph-based chatbots" in response_text or "chatbots" in response_text
    assert "agents" in response_text
    assert "user simulator" in response_text or "simulator" in response_text

    # Test Case 2: HTTP tool (contactTeam)
    mock_request.side_effect = create_mock_response
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

    # Verify successful submission response
    assert (
        "success" in output["answer"].lower() or "submitted" in output["answer"].lower()
    )
