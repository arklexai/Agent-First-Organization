"""Model provider configuration for the Arklex framework.

This module defines the configuration and mapping for different language model providers
supported by the framework, including OpenAI, Google Gemini, Anthropic, and HuggingFace.
It provides functions and mappings for initializing language models and embeddings from
different providers, ensuring consistent interface across different model backends.

Key Components:
1. LLM_PROVIDERS: List of supported language model providers
2. PROVIDER_MAP: Mapping of provider names to their LLM classes
3. PROVIDER_EMBEDDINGS: Mapping of provider names to their embedding classes
4. PROVIDER_EMBEDDING_MODELS: Default embedding models for each provider
5. get_huggingface_llm: Function to initialize HuggingFace models

Usage:
    from arklex.utils.model_provider_config import (
        LLM_PROVIDERS,
        PROVIDER_MAP,
        PROVIDER_EMBEDDINGS,
        PROVIDER_EMBEDDING_MODELS
    )

    # Initialize a model
    provider = "openai"
    if provider in LLM_PROVIDERS:
        llm_class = PROVIDER_MAP[provider]
        model = llm_class(model_name="gpt-4")

    # Get embeddings
    embedding_class = PROVIDER_EMBEDDINGS[provider]
    embedding_model = PROVIDER_EMBEDDING_MODELS[provider]
    embeddings = embedding_class(model=embedding_model)
"""

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_huggingface_llm(model: str, **kwargs: object) -> ChatHuggingFace:
    """Initialize a HuggingFace language model.

    This function creates a HuggingFace language model instance using the specified model
    and configuration parameters. It sets up the model for text generation tasks.

    Args:
        model (str): The HuggingFace model identifier to use.
        **kwargs (object): Additional configuration parameters for the model.

    Returns:
        ChatHuggingFace: A configured HuggingFace chat model instance.

    Example:
        llm = get_huggingface_llm(
            model="gpt2",
            temperature=0.7,
            max_length=100
        )
    """
    llm: HuggingFaceEndpoint = HuggingFaceEndpoint(
        repo_id=model, task="text-generation", **kwargs
    )
    return ChatHuggingFace(llm=llm)


class DummyLLM:
    def __init__(self, *args: object, **kwargs: object) -> None:
        # Set model_name from kwargs if provided
        self.model_name = kwargs.get("model", "dummy-model")
        # Also set model attribute for compatibility
        self.model = kwargs.get("model", "dummy-model")

    def invoke(self, messages: object) -> object:
        class Response:
            content = "dummy response"

        return Response()


# List of supported language model providers
LLM_PROVIDERS: list[str] = [
    "openai",  # OpenAI's language models
    "google",  # Google's models
    "anthropic",  # Anthropic's Claude models
    "huggingface",  # HuggingFace's open-source models
]

# Mapping of provider names to their LLM classes
PROVIDER_MAP: dict[str, type] = {
    "anthropic": ChatAnthropic,  # Anthropic's Claude models
    "google": ChatGoogleGenerativeAI,  # Google's models
    "openai": ChatOpenAI,  # OpenAI's GPT models
    "huggingface": get_huggingface_llm,  # HuggingFace's models
    "dummy": DummyLLM,  # Dummy provider for tests
}

# Mapping of provider names to their embedding classes
PROVIDER_EMBEDDINGS: dict[str, type] = {
    "anthropic": HuggingFaceEmbeddings,  # Anthropic uses HuggingFace embeddings
    "google": GoogleGenerativeAIEmbeddings,  # Google's embeddings
    "openai": OpenAIEmbeddings,  # OpenAI's embeddings
    "huggingface": HuggingFaceEmbeddings,  # HuggingFace's embeddings
}

# Mapping of provider names to their default embedding model identifiers
PROVIDER_EMBEDDING_MODELS: dict[str, str] = {
    "anthropic": "sentence-transformers/sentence-t5-base",  # T5-based embeddings
    "google": "models/embedding-001",  # Google embeddings
    "openai": "text-embedding-ada-002",  # OpenAI's Ada embeddings
    "huggingface": "sentence-transformers/all-mpnet-base-v2",  # MPNet embeddings
}
