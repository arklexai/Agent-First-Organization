"""ChatGPT and LLM utility functions for evaluation in the Arklex framework.

This module provides utility functions for interacting with various language models (OpenAI,
Anthropic, Gemini) during evaluation. It includes functionality for client creation,
message handling, conversation history management, and goal generation. The module supports
multiple LLM providers, conversation role flipping, history formatting, and chatbot querying
with parameter management.
"""

import json
import os
import random
from typing import Any

import anthropic
import requests
from dotenv import load_dotenv
from google.generativeai import GenerativeModel
from openai import OpenAI

from arklex.utils.model_config import MODEL
from arklex.utils.provider_utils import get_api_key_for_provider

load_dotenv()


def create_client() -> OpenAI | anthropic.Anthropic | GenerativeModel:
    """Create a client for interacting with language models.

    This function creates and returns a client for the configured language model provider
    (OpenAI, Anthropic, or Gemini). It handles API key configuration and organization
    settings.

    Returns:
        Union[OpenAI, anthropic.Anthropic, GenerativeModel]: A client instance for the configured LLM provider.

    Raises:
        KeyError: If required environment variables are not set.
        ValueError: If the configured provider is not supported.
    """
    try:
        org_key: str | None = os.environ["OPENAI_ORG_ID"]
    except KeyError:
        org_key = None

    provider = MODEL["llm_provider"]

    if not provider:
        raise ValueError(
            "llm_provider must be explicitly specified in MODEL configuration"
        )

    if provider == "openai":
        client: OpenAI = OpenAI(
            api_key=get_api_key_for_provider("openai"),
            organization=org_key,
        )
        return client

    elif provider == "google":
        # Set the API key for Google Generative AI
        import google.generativeai as genai

        genai.configure(api_key=get_api_key_for_provider("google"))
        # Use the model from MODEL configuration instead of hardcoding
        model_name = MODEL.get("model_type_or_path", "gemini-1.5-flash")
        client = GenerativeModel(model_name)
        return client

    elif provider == "anthropic":
        client: anthropic.Anthropic = anthropic.Anthropic(
            api_key=get_api_key_for_provider("anthropic")
        )
        return client

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def chatgpt_chatbot(
    messages: list[dict[str, str]],
    client: OpenAI | anthropic.Anthropic | GenerativeModel,
    model: str = None,
) -> str:
    """Send messages to a language model and get a response.

    This function sends a list of messages to the specified language model and returns
    its response. It handles different providers (OpenAI, Anthropic, Gemini) and their specific
    API requirements.

    Args:
        messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'.
        client (Union[OpenAI, anthropic.Anthropic, GenerativeModel]): The LLM client to use.
        model (str, optional): The model to use. Defaults to MODEL["model_type_or_path"].

    Returns:
        str: The model's response text.
    """
    # Get model from MODEL config if not provided
    if model is None:
        model = MODEL["model_type_or_path"]

    provider = MODEL["llm_provider"]

    if not provider:
        raise ValueError(
            "llm_provider must be explicitly specified in MODEL configuration"
        )

    # Ensure model is not empty
    if not model:
        raise ValueError(
            "Model parameter cannot be empty. Please check MODEL configuration."
        )

    if provider == "openai":
        answer: str = (
            client.chat.completions.create(
                model=model, messages=messages, temperature=0.1
            )
            .choices[0]
            .message.content.strip()
        )
    elif provider == "anthropic":
        kwargs: dict[str, Any] = {
            "model": model,
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
    elif provider == "google":
        # Convert messages to Gemini format
        gemini_messages = _convert_messages_to_gemini_format(messages)

        response = client.generate_content(
            gemini_messages,
            generation_config={"temperature": 0.1, "max_output_tokens": 1024},
        )
        answer: str = response.text.strip()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    return answer


def _convert_messages_to_gemini_format(
    messages: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Convert OpenAI/Anthropic message format to Gemini format.

    This function converts a list of messages from OpenAI/Anthropic format to
    Gemini's expected format. System messages are prepended to the first user message
    since Gemini doesn't support system messages directly.

    Args:
        messages (List[Dict[str, str]]): List of messages in OpenAI/Anthropic format.

    Returns:
        List[Dict[str, Any]]: List of messages in Gemini format.
    """
    gemini_messages = []
    system_content = ""

    # Extract system message if present
    if messages and messages[0]["role"] == "system":
        system_content = messages[0]["content"]
        messages = messages[1:]  # Remove system message from processing

    # Convert remaining messages
    for msg in messages:
        if msg["role"] == "user":
            gemini_messages.append(
                {"role": "user", "parts": [{"text": msg["content"]}]}
            )
        elif msg["role"] == "assistant":
            gemini_messages.append(
                {"role": "model", "parts": [{"text": msg["content"]}]}
            )

    # Prepend system content to first user message if it exists
    if system_content and gemini_messages and gemini_messages[0]["role"] == "user":
        gemini_messages[0]["parts"][0]["text"] = (
            f"{system_content}\n\n{gemini_messages[0]['parts'][0]['text']}"
        )

    return gemini_messages


# flip roles in convo history, only keep role and content
def flip_hist_content_only(hist: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Flip roles in conversation history, keeping only role and content.

    This function takes a conversation history and flips the roles (user becomes assistant
    and vice versa), while keeping only the 'role' and 'content' fields. System messages
    are removed.

    Args:
        hist (List[Dict[str, Any]]): The conversation history to process.

    Returns:
        List[Dict[str, str]]: The processed conversation history with flipped roles.
    """
    new_hist: list[dict[str, str]] = []
    for turn in hist:
        if turn["role"] == "system":
            continue
        elif turn["role"] == "user":
            new_hist.append({"role": "assistant", "content": turn["content"]})
        else:
            new_hist.append({"role": "user", "content": turn["content"]})
    return new_hist


# flip roles in convo history, keep all other keys the same
def flip_hist(hist: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flip roles in conversation history, preserving all fields.

    This function takes a conversation history and flips the roles (user becomes assistant
    and vice versa), while preserving all other fields in the message dictionaries.
    System messages are removed.

    Args:
        hist (List[Dict[str, Any]]): The conversation history to process.

    Returns:
        List[Dict[str, Any]]: The processed conversation history with flipped roles.
    """
    new_hist: list[dict[str, Any]] = []
    for turn in hist.copy():
        if "role" not in turn:
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
    history: list[dict[str, Any]],
    params: dict[str, Any],
    env_config: dict[str, Any],
) -> dict[str, Any]:
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
    data: dict[str, Any] = {
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


def format_chat_history_str(chat_history: list[dict[str, str]]) -> str:
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
    convo: list[dict[str, Any]], delim: str = "\n", filter_turns: bool = True
) -> list[dict[str, Any]]:
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
    filtered_convo: list[dict[str, Any]] = []
    for i, turn in enumerate(convo):
        if i <= 1 or "role" not in turn and filter_turns:
            continue
        elif "role" not in turn or turn["role"] == "assistant":
            filtered_convo.append(turn)
        else:
            idx: int = turn["content"].find(delim)
            new_turn: dict[str, Any] = {}
            for key in turn:
                if key == "content":
                    if idx == -1:
                        new_turn[key] = turn[key]
                    else:
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

    client = create_client()
    return chatgpt_chatbot(
        [{"role": "user", "content": message}],
        client,
        model=MODEL["model_type_or_path"],
    )


def generate_goal(
    doc_content: str, client: OpenAI | anthropic.Anthropic | GenerativeModel
) -> str:
    """Generate a goal based on document content.

    This function uses a language model to generate a goal or information request
    based on the provided document content. The goal is generated in first person
    and represents what a user might want to achieve when interacting with a chatbot
    about the document's content.

    Args:
        doc_content (str): The document content to base the goal on.
        client (Union[OpenAI, anthropic.Anthropic, GenerativeModel]): The LLM client to use.

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
    documents: list[dict[str, str]],
    params: dict[str, Any],
    client: OpenAI | anthropic.Anthropic | GenerativeModel,
) -> list[str]:
    """Generate multiple goals based on a collection of documents.

    This function generates a specified number of goals by randomly selecting documents
    and using the language model to generate goals based on their content.

    Args:
        documents (List[Dict[str, str]]): List of documents with their content.
        params (Dict[str, Any]): Parameters including 'num_goals' to generate.
        client (Union[OpenAI, anthropic.Anthropic, GenerativeModel]): The LLM client to use.

    Returns:
        List[str]: List of generated goals.
    """
    goals: list[str] = []
    for _ in range(params["num_goals"]):
        doc: dict[str, str] = random.choice(documents)
        goals.append(generate_goal(doc["content"], client))
    return goals


def _print_goals(goals: list[str]) -> None:
    print(goals)


def main() -> None:
    from get_documents import filter_documents, get_all_documents

    documents = get_all_documents()
    documents = filter_documents(documents)
    client = create_client()
    params = {"num_goals": 1}
    _print_goals(generate_goals(documents, params, client))


if __name__ == "__main__":
    main()
