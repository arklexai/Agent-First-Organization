from integration_tests.base import BaseTestOrchestrator, ChatRole


def test_workers() -> None:
    orchestrator = BaseTestOrchestrator(
        "integration_tests/taskgraphs/worker_taskgraph.json"
    )
    params = BaseTestOrchestrator.init_params()
    chat_history, params = params["chat_history"], params["parameters"]

    # start message (direct message)
    text = "<start>"
    output = orchestrator.get_resopnse(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    assert (
        output["answer"]
        == "Hello! I'm here to assist you with any customer service inquiries."
    )

    # message worker (undirected message)
    text = "How is the weather?"
    output = orchestrator.get_resopnse(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    assert len(output["answer"]) > 1

    # multiple choice worker
    text = "Which car would you like to buy?"
    output = orchestrator.get_resopnse(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    assert output["answer"] == "Which car would you like to buy?"
    assert output["choice_list"] == ["Car A", "Car B", "Car C"]

    # TODO: Milvus RAG worker

    # TODO: RAG message worker

    # Human in the loop worker
    text = "Connect me with a human agent"
    output = orchestrator.get_resopnse(text, chat_history, params)
    chat_history.append({"role": ChatRole.USER, "content": text})
    chat_history.append({"role": ChatRole.ASSISTANT, "content": output["answer"]})
    params = output["parameters"]
    assert output["answer"] == "I'll connect you to a representative!"
    assert output["human_in_the_loop"] == "live"
