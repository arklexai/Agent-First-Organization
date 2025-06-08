"""ChatGPT and LLM utility functions for evaluation in the Arklex framework.

This module provides utility functions for interacting with various language models (OpenAI,
Anthropic, Gemini) during evaluation. It includes functionality for client creation,
message handling, conversation history management, and goal generation. The module supports
multiple LLM providers, conversation role flipping, history formatting, and chatbot querying
with parameter management.
"""

import os
import random
import json
import requests
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI
import anthropic
from dotenv import load_dotenv

from arklex.utils.model_config import MODEL

load_dotenv()


def create_client() -> Union[OpenAI, anthropic.Anthropic]:
    """Create a client for interacting with language models.

    This function creates and returns a client for the configured language model provider
    (OpenAI, Anthropic, or Gemini). It handles API key configuration and organization
    settings.

    Returns:
        Union[OpenAI, anthropic.Anthropic]: A client instance for the configured LLM provider.

    Raises:
        KeyError: If required environment variables are not set.
    """
    try:
        org_key: Optional[str] = os.environ["OPENAI_ORG_ID"]
    except:
        org_key = None
    if MODEL["llm_provider"] == "openai" or MODEL["llm_provider"] == "gemini":
        client: OpenAI = OpenAI(
            api_key=os.environ[f"{MODEL['llm_provider'].upper()}_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            if MODEL["llm_provider"] == "gemini"
            else None,
            organization=org_key,
        )
    elif MODEL["llm_provider"] == "anthropic":
        client: anthropic.Anthropic = anthropic.Anthropic()
    return client


def chatgpt_chatbot(
    messages: List[Dict[str, str]],
    client: Union[OpenAI, anthropic.Anthropic],
    model: str = MODEL["model_type_or_path"],
) -> str:
    """Send messages to a language model and get a response.

    This function sends a list of messages to the specified language model and returns
    its response. It handles different providers (OpenAI, Anthropic) and their specific
    API requirements.

    Args:
        messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'.
        client (Union[OpenAI, anthropic.Anthropic]): The LLM client to use.
        model (str, optional): The model to use. Defaults to MODEL["model_type_or_path"].

    Returns:
        str: The model's response text.
    """
    if MODEL["llm_provider"] != "anthropic":
        answer: str = (
            client.chat.completions.create(
                model=MODEL["model_type_or_path"], messages=messages, temperature=0.1
            )
            .choices[0]
            .message.content.strip()
        )
    else:
        kwargs: Dict[str, Any] = {
            "model": MODEL["model_type_or_path"],
            "messages": messages if messages[0]["role"] != "system" else [messages[1]],
            "temperature": 0.1,
            "max_tokens": 1024,
            **(
                {"system": messages[0]["content"]}
                if messages[0]["role"] == "system"
                else {}
            ),
        }
        answer: str = client.messages.create(**kwargs).content[0].text.strip()

    return answer


# flip roles in convo history, only keep role and content
def flip_hist_content_only(hist: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Flip roles in conversation history, keeping only role and content.

    This function takes a conversation history and flips the roles (user becomes assistant
    and vice versa), while keeping only the 'role' and 'content' fields. System messages
    are removed.

    Args:
        hist (List[Dict[str, Any]]): The conversation history to process.

    Returns:
        List[Dict[str, str]]: The processed conversation history with flipped roles.
    """
    new_hist: List[Dict[str, str]] = []
    for turn in hist:
        if turn["role"] == "system":
            continue
        elif turn["role"] == "user":
            new_hist.append({"role": "assistant", "content": turn["content"]})
        else:
            new_hist.append({"role": "user", "content": turn["content"]})
    return new_hist


# flip roles in convo history, keep all other keys the same
def flip_hist(hist: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flip roles in conversation history, preserving all fields.

    This function takes a conversation history and flips the roles (user becomes assistant
    and vice versa), while preserving all other fields in the message dictionaries.
    System messages are removed.

    Args:
        hist (List[Dict[str, Any]]): The conversation history to process.

    Returns:
        List[Dict[str, Any]]: The processed conversation history with flipped roles.
    """
    new_hist: List[Dict[str, Any]] = []
    for turn in hist.copy():
        if "role" not in turn.keys():
            new_hist.append(turn)
        elif turn["role"] == "system":
            continue
        elif turn["role"] == "user":
            turn["role"] = "assistant"
            new_hist.append(turn)
        else:
            turn["role"] = "user"
            new_hist.append(turn)
    return new_hist


def query_chatbot(
    model_api: str,
    history: List[Dict[str, Any]],
    params: Dict[str, Any],
    env_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Query a chatbot API with conversation history and parameters.

    This function sends a request to a chatbot API with the provided conversation history,
    parameters, and environment configuration. It flips the roles in the history before
    sending the request.

    Args:
        model_api (str): The URL of the chatbot API.
        history (List[Dict[str, Any]]): The conversation history.
        params (Dict[str, Any]): Additional parameters for the request.
        env_config (Dict[str, Any]): Environment configuration including workers and tools.

    Returns:
        Dict[str, Any]: The API response as a dictionary.
    """
    history = flip_hist_content_only(history)
    data: Dict[str, Any] = {
        "history": history,
        "parameters": params,
        "workers": env_config["workers"],
        "tools": env_config["tools"],
    }
    data_str: str = json.dumps(data)
    response = requests.post(
        model_api, headers={"Content-Type": "application/json"}, data=data_str
    )
    return response.json()


def format_chat_history_str(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history into a single string.

    This function takes a list of chat messages and formats them into a single string,
    with each message prefixed by its role in uppercase.

    Args:
        chat_history (List[Dict[str, str]]): List of chat messages with 'role' and 'content'.

    Returns:
        str: The formatted chat history string.
    """
    formatted_hist: str = ""
    for turn in chat_history:
        formatted_hist += turn["role"].upper() + ": " + turn["content"] + " "
    return formatted_hist.strip()


# filter prompts out of bot utterances
def filter_convo(
    convo: List[Dict[str, Any]], delim: str = "\n", filter_turns: bool = True
) -> List[Dict[str, Any]]:
    """Filter and process a conversation.

    This function processes a conversation by:
    1. Skipping the first two turns
    2. Optionally filtering out turns without a 'role' field
    3. For assistant messages, keeping them as is
    4. For user messages, truncating content at the specified delimiter

    Args:
        convo (List[Dict[str, Any]]): The conversation to process.
        delim (str, optional): Delimiter for truncating user messages. Defaults to "\n".
        filter_turns (bool, optional): Whether to filter turns without roles. Defaults to True.

    Returns:
        List[Dict[str, Any]]: The processed conversation.
    """
    filtered_convo: List[Dict[str, Any]] = []
    for i, turn in enumerate(convo):
        if i <= 1:
            continue
        elif "role" not in turn.keys() and filter_turns:
            continue
        elif "role" not in turn.keys():
            filtered_convo.append(turn)
        elif turn["role"] == "assistant":
            filtered_convo.append(turn)
        else:
            idx: int = turn["content"].find(delim)
            new_turn: Dict[str, Any] = {}
            for key in turn.keys():
                if key == "content":
                    new_turn[key] = turn[key][:idx]
                else:
                    new_turn[key] = turn[key]
            filtered_convo.append(new_turn)
    return filtered_convo


def adjust_goal(doc_content: str, goal: str) -> str:
    """Adjust a goal based on document content.

    This function uses a language model to adjust a goal by replacing specific product
    mentions with products from the provided document content. If no specific products
    are mentioned in the goal, it returns the original goal unchanged.

    Args:
        doc_content (str): The document content to reference for product replacements.
        goal (str): The original goal to adjust.

    Returns:
        str: The adjusted goal.
    """
    message: str = f"Pretend you have the following goal in the mind. If the goal including some specific product, such as floss, mug, iphone, etc., then please replace it with the product from the following document content. Otherwise, don't need to change it and just return the original goal. The document content is as follows:\n{doc_content}\n\nThe original goal is as follows:\n{goal}\n\nOnly give the answer to the question in your response."

    return chatgpt_chatbot(
        [{"role": "user", "content": message}], model=MODEL["model_type_or_path"]
    )


def generate_goal(doc_content: str, client: Union[OpenAI, anthropic.Anthropic]) -> str:
    """Generate a goal based on document content.

    This function uses a language model to generate a goal or information request
    based on the provided document content. The goal is generated in first person
    and represents what a user might want to achieve when interacting with a chatbot
    about the document's content.

    Args:
        doc_content (str): The document content to base the goal on.
        client (Union[OpenAI, anthropic.Anthropic]): The LLM client to use.

    Returns:
        str: The generated goal.
    """
    message: str = f"Pretend you have just read the following website:\n{doc_content}\nThis website also has a chatbot. What is some information you want to get from this chatbot or a goal you might have when chatting with this chatbot based on the website content? Answer the question in the first person. Only give the answer to the question in your response."

    return chatgpt_chatbot(
        [{"role": "user", "content": message}],
        client,
        model=MODEL["model_type_or_path"],
    )


def generate_goals(
    documents: List[Dict[str, str]],
    params: Dict[str, Any],
    client: Union[OpenAI, anthropic.Anthropic],
) -> List[str]:
    """Generate multiple goals based on a collection of documents.

    This function generates a specified number of goals by randomly selecting documents
    and using the language model to generate goals based on their content.

    Args:
        documents (List[Dict[str, str]]): List of documents with their content.
        params (Dict[str, Any]): Parameters including 'num_goals' to generate.
        client (Union[OpenAI, anthropic.Anthropic]): The LLM client to use.

    Returns:
        List[str]: List of generated goals.
    """
    goals: List[str] = []
    for i in range(params["num_goals"]):
        doc: Dict[str, str] = random.choice(documents)
        goals.append(generate_goal(doc["content"], client))
    return goals


if __name__ == "__main__":
    from get_documents import get_all_documents, filter_documents

    documents = get_all_documents()
    documents = filter_documents(documents)
    client = create_client()
    params = {"num_goals": 1}
    print(generate_goals(documents, params, client))
