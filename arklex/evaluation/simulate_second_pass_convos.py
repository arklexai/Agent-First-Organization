"""Second-pass conversation simulation for evaluation in the Arklex framework.

This module provides functionality for generating labeled conversations based on intent paths
derived from first-pass conversations. It includes utilities for sampling conversation paths,
interacting with the chatbot system, and generating labeled conversations with specific
intents. The module supports path sampling with weighted edges and natural conversation flow
while maintaining intent control.
"""

import json
import random
from typing import Any

import anthropic
import networkx as nx
from openai import OpenAI

from arklex.evaluation.chatgpt_utils import (
    chatgpt_chatbot,
    filter_convo,
    flip_hist,
    query_chatbot,
)
from arklex.evaluation.extract_conversation_info import build_intent_graph


def sampling_paths(
    start_node: str,
    graph: nx.DiGraph,
    path_length: int,
    max_turns: int,
    intents: list[str],
) -> list[str]:
    children: list[str] = list(graph.successors(start_node))
    if path_length >= max_turns or len(children) == 0:
        return intents
    weights: list[float] = []
    for c in children:
        weights.append(graph.get_edge_data(start_node, c)["weight"])
    next_node: str = random.choices(children, weights)[0]
    intents.append(next_node)
    return sampling_paths(next_node, graph, path_length + 1, max_turns, intents)


def get_paths(G: nx.DiGraph, num_paths: int, max_turns: int) -> list[list[str]]:
    my_paths: list[list[str]] = []
    for _i in range(num_paths):
        my_path: list[str] = sampling_paths("start", G, 0, max_turns, ["start"])
        my_paths.append(my_path[1:])
    return my_paths


def interact(
    intent_path: list[str],
    summary: str,
    model_api: str,
    model_params: dict[str, Any],
    client: OpenAI | anthropic.Anthropic,
    env_config: dict[str, Any],
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    instructional_prompt: str = (
        "Replicate the behavior of a human customer. You are interacting with customer service chatbot for the following company: "
        + summary
    )
    start_text: str = (
        "Begin the conversation as a human customer with the following intent: "
        + intent_path[0]
    )
    history.append({"role": "system", "content": instructional_prompt})
    history.append({"role": "user", "content": start_text})
    for i in range(len(intent_path)):
        intent: str = intent_path[i]
        output: str = chatgpt_chatbot(history, client)
        history.append({"role": "assistant", "content": output, "intent": intent})
        response_data: dict[str, Any] = query_chatbot(
            model_api, filter_convo(history), model_params, env_config
        )
        answer: str = response_data["answer"]
        answer = answer.replace("\n", " ")
        model_params = response_data.get("parameters", model_params)
        if i < len(intent_path) - 1:
            intent = intent_path[i + 1]
        history.append(
            {
                "role": "user",
                "content": answer
                + "\nRespond to this utterance with the following intent: "
                + intent
                + "\nMake sure your response is natural and follows the flow of the conversation. For example, if the bot asks you a question make sure you answer it.",
            }
        )
    return history


def generate_labeled_convos(
    intent_paths: list[list[str]],
    summary: str,
    model_api: str,
    model_params: dict[str, Any],
    client: OpenAI | anthropic.Anthropic,
    env_config: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    convos: list[list[dict[str, Any]]] = []
    model_params = {}
    for intent_path in intent_paths:
        convo: list[dict[str, Any]] = interact(
            intent_path, summary, model_api, model_params, client, env_config
        )
        convos.append(flip_hist(filter_convo(convo)))
    return convos


def get_labeled_convos(
    first_pass_data: list[dict[str, Any]],
    model_api: str,
    synthetic_data_params: dict[str, Any],
    model_params: dict[str, Any],
    config: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    intent_graph: nx.DiGraph = build_intent_graph(first_pass_data)
    intent_paths: list[list[str]] = get_paths(
        intent_graph,
        synthetic_data_params["num_convos"],
        synthetic_data_params["max_turns"],
    )
    summary: str = config["intro"]
    client: OpenAI | anthropic.Anthropic = config["client"]
    env_config: dict[str, Any] = {
        "workers": config["workers"],
        "tools": config["tools"],
        "client": config["client"],
    }
    convos: list[list[dict[str, Any]]] = generate_labeled_convos(
        intent_paths, summary, model_api, model_params, client, env_config
    )
    return convos


def main() -> None:
    with open("temp_files/p1_sample_convos_labeled.json") as f:
        data: list[dict[str, Any]] = json.load(f)

    with open("temp_files/richtech_config.json") as f:
        config: dict[str, Any] = json.load(f)

    model_api: str = "http://adaptation.cs.columbia.edu:55231/qa/richtech/v1alpha1"
    synthetic_data_params: dict[str, Any] = {
        "num_convos": 2,
        "num_goals": 3,
        "max_turns": 10,
    }
    model_params: dict[str, Any] = {}

    labeled_convos: list[list[dict[str, Any]]] = get_labeled_convos(
        data, model_api, synthetic_data_params, model_params, config
    )

    with open("files/p2_sample_convos.json", "w") as f:
        json.dump(labeled_convos, f, indent=5)


if __name__ == "__main__":
    main()
