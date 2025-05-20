from typing import Dict, List, Any, Callable, Type, Union
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


def get_huggingface_llm(model: str, **kwargs: Any) -> ChatHuggingFace:
    llm: HuggingFaceEndpoint = HuggingFaceEndpoint(
        repo_id=model, task="text-generation", **kwargs
    )
    return ChatHuggingFace(llm=llm)


LLM_PROVIDERS: List[str] = ["openai", "gemini", "anthropic", "huggingface"]

PROVIDER_MAP: Dict[
    str,
    Union[
        Type[ChatOpenAI],
        Type[ChatGoogleGenerativeAI],
        Type[ChatAnthropic],
        Callable[..., ChatHuggingFace],
    ],
] = {
    "anthropic": ChatAnthropic,
    "gemini": ChatGoogleGenerativeAI,
    "openai": ChatOpenAI,
    "huggingface": get_huggingface_llm,
}

PROVIDER_EMBEDDINGS: Dict[
    str,
    Type[Union[OpenAIEmbeddings, GoogleGenerativeAIEmbeddings, HuggingFaceEmbeddings]],
] = {
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
