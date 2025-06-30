"""Document loading and processing component for the Arklex framework.

This module provides the DocumentLoader class that handles loading and processing
of documentation from various sources. It ensures proper document validation and
formatting for task generation.

Key Features:
- Document loading from multiple sources
- Document validation and error handling
- Document caching for performance
- Support for task and instruction documents
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from arklex.orchestrator.generator.docs.document_validator import DocumentValidator
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class DocumentLoader:
    """Document loader for loading and validating documents."""

    def __init__(self, cache_dir: Path, validate_documents: bool = True) -> None:
        """Initialize the document loader.

        Args:
            cache_dir (Path): Directory for caching documents
            validate_documents (bool): Whether to validate documents
        """
        self._cache_dir = cache_dir
        self._validate_documents = validate_documents
        self._cache: dict[str, dict[str, Any]] = {}
        self._validator = DocumentValidator()

    def load_document(self, doc_path: Path) -> dict[str, Any]:
        """Load a document from file.

        Args:
            doc_path (Path): Path to the document file

        Returns:
            Dict[str, Any]: The loaded document

        Raises:
            FileNotFoundError: If the document file doesn't exist
            json.JSONDecodeError: If the document is not valid JSON
        """
        if str(doc_path) in self._cache:
            return self._cache[str(doc_path)]
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")
        try:
            if hasattr(doc_path, "read_text"):
                content = doc_path.read_text()
            else:
                with open(doc_path) as f:
                    content = f.read()
            document = json.loads(content)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in document: {doc_path}", e.doc, e.pos
            ) from e
        if self._validate_documents and not self.validate_document(document):
            raise ValueError(f"Invalid document structure: {doc_path}")
        return document

    def load_task_document(self, doc_path: Path | str) -> dict[str, Any]:
        """Load a task document.

        Args:
            doc_path (Union[Path, str]): Path or URL to the task document

        Returns:
            Dict[str, Any]: The loaded task document

        Raises:
            FileNotFoundError: If the document file doesn't exist
            ValueError: If the document structure is invalid
        """
        cache_key = str(doc_path)
        if cache_key in self._cache:
            document = self._cache[cache_key]
        else:
            if isinstance(doc_path, str) and doc_path.startswith("http"):
                response = requests.get(doc_path)
                response.raise_for_status()
                url_hash = hashlib.md5(doc_path.encode("utf-8")).hexdigest()
                ext = os.path.splitext(doc_path)[-1] or ".html"
                filename = f"url_{url_hash}{ext}"
                file_path = os.path.join(self._cache_dir, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                from pathlib import Path

                doc_path = Path(file_path)
            if isinstance(doc_path, str):
                from pathlib import Path

                doc_path = Path(doc_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Document not found: {doc_path}")
            if hasattr(doc_path, "read_text"):
                content = doc_path.read_text()
            else:
                with open(doc_path) as f:
                    content = f.read()
            try:
                document = json.loads(content)
            except json.JSONDecodeError:
                try:
                    soup = BeautifulSoup(content, "html.parser")
                    document = {
                        "task_id": "html_task",
                        "name": soup.title.string if soup.title else "HTML Document",
                        "description": "Document parsed from HTML",
                        "steps": [
                            {"step_id": i, "description": p.get_text()}
                            for i, p in enumerate(soup.find_all("p"), 1)
                        ],
                    }
                except Exception as e:
                    raise ValueError(
                        f"Content at {doc_path} is neither valid JSON nor parseable HTML. Error: {e}"
                    ) from e
            self._cache[cache_key] = document
        if not self._validate_task_document(document):
            raise ValueError(f"Invalid task document structure: {doc_path}")
        return document

    def load_instruction_document(self, doc_path: Path) -> dict[str, Any]:
        """Load an instruction document.

        Args:
            doc_path (Path): Path to the instruction document

        Returns:
            Dict[str, Any]: The loaded instruction document

        Raises:
            FileNotFoundError: If the document file doesn't exist
            ValueError: If the document structure is invalid
        """
        if str(doc_path) in self._cache:
            document = self._cache[str(doc_path)]
        else:
            if not doc_path.exists():
                raise FileNotFoundError(f"Document not found: {doc_path}")
            if hasattr(doc_path, "read_text"):
                content = doc_path.read_text()
            else:
                with open(doc_path) as f:
                    content = f.read()
            document = json.loads(content)
            self._cache[str(doc_path)] = document
        if not self._validate_instruction_document(document):
            raise ValueError(f"Invalid instruction document structure: {doc_path}")
        return document

    def cache_document(self, doc_path: Path, document: dict[str, Any]) -> None:
        """Cache a document.

        Args:
            doc_path (Path): Path to the document
            document (Dict[str, Any]): Document to cache
        """
        self._cache[str(doc_path)] = document

    def validate_document(self, document: dict[str, Any]) -> bool:
        """Validate document structure.

        Args:
            document (Dict[str, Any]): Document to validate

        Returns:
            bool: True if document is valid
        """
        if not isinstance(document, dict):
            return False
        required_fields = ["title", "sections"]
        if not all(field in document for field in required_fields):
            return False
        if not isinstance(document["sections"], list):
            return False
        for section in document["sections"]:
            if not isinstance(section, dict):
                return False
            if "name" not in section or "content" not in section:
                return False
            if "requirements" in section and not isinstance(
                section["requirements"], list
            ):
                return False
        return True

    def _validate_task_document(self, document: dict[str, Any]) -> bool:
        """Validate task document structure.

        Args:
            document (Dict[str, Any]): Document to validate

        Returns:
            bool: True if document is valid
        """
        if not isinstance(document, dict):
            return False
        required_fields = ["task_id", "name", "description", "steps"]
        if not all(field in document for field in required_fields):
            return False
        if not isinstance(document["steps"], list):
            return False
        for step in document["steps"]:
            if not isinstance(step, dict):
                return False
            if "step_id" not in step or "description" not in step:
                return False
            if "required_fields" in step and not isinstance(
                step["required_fields"], list
            ):
                return False
        return True

    def _validate_instruction_document(self, document: dict[str, Any]) -> bool:
        """Validate instruction document structure.

        Args:
            document (Dict[str, Any]): Document to validate

        Returns:
            bool: True if document is valid
        """
        if not isinstance(document, dict):
            return False
        required_fields = ["instruction_id", "title", "content", "sections"]
        if not all(field in document for field in required_fields):
            return False
        if not isinstance(document["sections"], list):
            return False
        for section in document["sections"]:
            if not isinstance(section, dict):
                return False
            if "section_id" not in section or "title" not in section:
                return False
            if "steps" in section and not isinstance(section["steps"], list):
                return False
        return True
