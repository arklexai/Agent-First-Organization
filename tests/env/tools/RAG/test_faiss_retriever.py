import pytest
from unittest.mock import patch
from arklex.env.tools.RAG.retrievers.faiss_retriever import FaissRetrieverExecutor


def test_faiss_retriever_file_not_found() -> None:
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            FaissRetrieverExecutor.load_docs(database_path="/tmp", llm_config=None)
