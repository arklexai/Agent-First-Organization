from arklex.resources.resource_types import WorkerItem
from integration_tests.utils.base import BaseTestOrchestrator, ChatRole


async def test_workers() -> None:
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
    assert "weather" in output["answer"].lower()

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
    assert (
        "agent" in output["answer"].lower() or "evaluation" in output["answer"].lower()
    )
    last_trajectory = params["memory"]["trajectory"][-1][0]
    assert last_trajectory["info"]["resource"]["id"] == WorkerItem.MILVUS_RAG_WORKER
    assert last_trajectory["steps"] is not None

    # RAG message worker
    text = "What is your company's culture?"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    assert (
        "fast" in output["answer"].lower()
        or "curious" in output["answer"].lower()
        or "purpose" in output["answer"].lower()
    )
    last_trajectory = params["memory"]["trajectory"][-1][0]
    assert last_trajectory["info"]["resource"]["id"] == WorkerItem.RAG_MESSAGE_WORKER

    # Human in the loop worker
    text = "Connect me with a human agent"
    output = await orchestrator.get_response(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    assert output["answer"] == "I'll connect you to a representative!"
    assert output["human_in_the_loop"] == "live"
