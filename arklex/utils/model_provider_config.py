"""Model provider configuration for the Arklex framework.

This module defines the configuration and mapping for different language model providers
supported by the framework, including OpenAI, Google Gemini, Anthropic, and HuggingFace.
It provides functions and mappings for initializing language models and embeddings from
different providers, ensuring consistent interface across different model backends.
"""

from typing import Dict, Type, Any, List
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


def get_huggingface_llm(model: str, **kwargs: Any) -> ChatHuggingFace:
    llm = HuggingFaceEndpoint(repo_id=model, task="text-generation", **kwargs)
    return ChatHuggingFace(llm=llm)


LLM_PROVIDERS: List[str] = ["openai", "gemini", "anthropic", "huggingface"]

PROVIDER_MAP: Dict[str, Type] = {
    "anthropic": ChatAnthropic,
    "gemini": ChatGoogleGenerativeAI,
    "openai": ChatOpenAI,
    "huggingface": get_huggingface_llm,
}

PROVIDER_EMBEDDINGS: Dict[str, Type] = {
    "anthropic": HuggingFaceEmbeddings,
    "gemini": GoogleGenerativeAIEmbeddings,
    "openai": OpenAIEmbeddings,
    "huggingface": HuggingFaceEmbeddings,
}

PROVIDER_EMBEDDING_MODELS: Dict[str, str] = {
    "anthropic": "sentence-transformers/sentence-t5-base",
    "gemini": "models/embedding-001",
    "openai": "text-embedding-ada-002",
    "huggingface": "sentence-transformers/all-mpnet-base-v2",
}
