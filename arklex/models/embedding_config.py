from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    embedding_provider: str


def load_embedding(embedding_config: EmbeddingConfig) -> any:
    """Load an embedding model based on the configuration."""
    if embedding_config.embedding_provider == "openai":
        return OpenAIEmbeddings(model="text-embedding-ada-002")
    elif embedding_config.embedding_provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    elif embedding_config.embedding_provider == "huggingface":
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")
    else:
        raise ValueError(
            f"Unsupported embedding provider: {embedding_config.embedding_provider}"
        )
