"""Conversation information extraction and analysis for evaluation in the Arklex framework.

This module provides functionality for analyzing conversations and extracting metrics for
evaluation purposes. It includes utilities for building intent graphs, tracking conversation
flows, checking goal completion, and calculating various performance metrics such as task
completion rates and efficiency. The module supports both user and bot goal tracking,
conversation filtering, and statistical analysis of conversation patterns.
"""

import json
from typing import Any

import anthropic
import networkx as nx
from openai import OpenAI

from arklex.evaluation.chatgpt_utils import (
    chatgpt_chatbot,
    filter_convo,
    flip_hist_content_only,
    format_chat_history_str,
)


def get_edges_and_counts(data: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    edge_counts: dict[tuple[str, str], int] = {}
    for convo in data:
        filtered_convo = filter_convo(convo["convo"])
        for i in range(len(filtered_convo)):
            if filtered_convo[i]["role"] == "assistant":
                continue
            prev_intent: str = "start" if i == 0 else filtered_convo[i - 2]["intent"]
            current_intent: str = filtered_convo[i]["intent"]
            edge_counts[(prev_intent, current_intent)] = (
                edge_counts.get((prev_intent, current_intent), 0) + 1
            )
    return edge_counts


def build_intent_graph(data: list[dict[str, Any]]) -> nx.DiGraph:
    G: nx.DiGraph = nx.DiGraph()
    edge_counts: dict[tuple[str, str], int] = get_edges_and_counts(data)
    for key in edge_counts:
        G.add_edge(key[0], key[1], weight=edge_counts[key])
    return G


def check_bot_goal(
    convo: list[dict[str, Any]], bot_goal: str, client: OpenAI | anthropic.Anthropic
) -> bool:
    convo_str: str = format_chat_history_str(flip_hist_content_only(convo))
    prompt: str = f"Here is a conversation between a user and a customer service chatbot assistant:\n{convo_str}\n\nThe chatbot's goal is the following: {bot_goal}\nOutput True if the bot was able to achieve its goal. Output False otherwise. Only output True or False and nothing else."
    output: str = chatgpt_chatbot([{"role": "user", "content": prompt}], client)
    return output == "True"


def num_user_turns(convo: list[dict[str, Any]]) -> int:
    user_turns: int = 0
    for turn in convo:
        if turn.get("role", None) == "user":
            user_turns += 1
    return user_turns


def extract_task_completion_metrics(
    data: list[dict[str, Any]],
    client: OpenAI | anthropic.Anthropic,
    bot_goal: str | None = None,
) -> dict[str, float] | str:
    num_convos: int = len(data)
    if num_convos == 0:
        return "Error while extracting task completion metrics"
    goal_completetions: int = 0
    bot_goal_completions: int = 0
    completion_efficiency: int = 0
    for convo in data:
        convo_history: list[dict[str, Any]] = convo["convo"]
        completion_efficiency += num_user_turns(convo_history)
        if convo["goal_completion"]:
            goal_completetions += 1
        if bot_goal is not None and check_bot_goal(convo_history, bot_goal, client):
            bot_goal_completions += 1
    metrics: dict[str, float] = {
        "user_task_completion": goal_completetions / num_convos,
        "user_task_completion_efficiency": completion_efficiency / num_convos,
    }
    if bot_goal is not None:
        metrics["bot_goal_completion"] = bot_goal_completions / num_convos
    return metrics


def print_edge_weights_from_file(path: str) -> None:
    with open(path) as f:
        data: list[dict[str, Any]] = json.load(f)
    G: nx.DiGraph = build_intent_graph(data)
    for e in list(G.edges()):
        print(f"Weight for edge {e}: {G.get_edge_data(e[0], e[1])['weight']}")


if __name__ == "__main__":
    print_edge_weights_from_file("files/p1_sample_convos_labeled.json")
