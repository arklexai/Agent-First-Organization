from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from arklex.resources.resource_types import WorkerItem
from integration_tests.utils.base import BaseTestOrchestrator, ChatRole


def mock_model_invoke(messages: list[Any]) -> AIMessage:
    # Get the last message content
    last_message = messages[-1]
    content = last_message.content

    # RAG question reformulation requests
    if "formulate a standalone question" in content:
        if "What products do you offer?" in content:
            return AIMessage(content="What products do you offer?")
        elif "What is your company's culture?" in content:
            return AIMessage(content="What is your company's culture?")
        else:
            raise ValueError(
                f"Unknown RAG question reformulation request: {content[-100:]}"
            )

    # Retriever needed requests
    elif "decide whether information retrieval is needed" in content:
        if "What products do you offer?" in content:  # noqa: SIM114
            return AIMessage(content="yes")
        elif "What is your company's culture?" in content:  # noqa: SIM114
            return AIMessage(content="yes")
        else:
            raise ValueError(f"Unknown retriever needed request: {content[-100:]}")

    return AIMessage(content="This is a demo worker response.")


def mock_structured_output_invoke(messages: list[Any]) -> dict[str, str]:
    # Get the last message content
    last_message = messages[-1]
    content = last_message.content

    # Intent detection responses
    if "Given the following intents and their definitions" in content:
        if content.endswith("How is the weather?\n"):
            return {"intent": "ask about weather"}
        elif content.endswith("Which car would you like to buy?\n"):
            return {"intent": "ask about car options"}
        elif content.endswith("What products do you offer?\n"):
            return {"intent": "ask about company products"}
        elif content.endswith("What is your company's culture?\n"):
            return {"intent": "ask about company culture"}
        elif content.endswith("Connect me with a human agent\n"):
            return {"intent": "user want to connect with a human agent"}
        else:
            raise ValueError(f"Unknown intent detection request: {content[-100:]}")

    # Default response for other structured outputs
    return {"intent": "others"}


def create_mock_llm() -> MagicMock:
    mock_llm = MagicMock()

    mock_llm.invoke = MagicMock(side_effect=mock_model_invoke)

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke = MagicMock(side_effect=mock_structured_output_invoke)
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured_llm)

    return mock_llm


@patch("arklex.models.model_service.load_llm")
async def test_workers(mock_load_llm: MagicMock) -> None:
    # Configure load_llm to return our mock LLM
    mock_load_llm.return_value = create_mock_llm()

    orchestrator = BaseTestOrchestrator(
        "integration_tests/taskgraphs/worker_taskgraph.json"
    )
    params = BaseTestOrchestrator.init_params()
    chat_history, params = params["chat_history"], params["parameters"]

    # start message (direct message)
    text = "<start>"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    assert (
        output["answer"]
        == "Hello! I'm here to assist you with any customer service inquiries."
    )

    # message worker (undirected message)
    text = "How is the weather?"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    assert len(output["answer"]) > 1

    # multiple choice worker
    text = "Which car would you like to buy?"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    assert output["answer"] == "Which car would you like to buy?"
    assert output["choice_list"] == ["Car A", "Car B", "Car C"]

    # Milvus RAG worker
    text = "What products do you offer?"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    last_trajectory = params["memory"]["trajectory"][-1][0]
    assert last_trajectory["info"]["resource"]["id"] == WorkerItem.MILVUS_RAG_WORKER
    assert len(last_trajectory["steps"]) > 0

    # RAG message worker
    text = "What is your company's culture?"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    last_trajectory = params["memory"]["trajectory"][-1][0]
    assert last_trajectory["info"]["resource"]["id"] == WorkerItem.RAG_MESSAGE_WORKER
    assert len(last_trajectory["steps"]) > 0

    # Human in the loop worker
    text = "Connect me with a human agent"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    assert output["answer"] == "I'll connect you to a representative!"
    assert output["human_in_the_loop"] == "live"
