"""Document processing component for the Arklex framework.

This module provides the DocumentProcessor class that handles the processing
and transformation of documents into a format suitable for task generation.
"""

from typing import Any


class DocumentProcessor:
    """Processes and transforms documents for task generation.

    This class handles the processing of raw document content, extracting
    relevant information, and formatting it for use in task generation.
    It ensures that documents are properly structured and contain all
    necessary information for task creation.

    Attributes:
        _processed_documents (Dict[str, Any]): Cache of processed documents
    """

    def __init__(self) -> None:
        """Initialize the DocumentProcessor."""
        self._processed_documents: dict[str, Any] = {}

    def process_document(self, document: dict[str, Any]) -> dict[str, Any]:
        """Process a document and output 'processed_sections' for test compatibility.

        Args:
            document (Dict[str, Any]): The document to process

        Returns:
            Dict[str, Any]: The processed document with 'processed_sections'
        """
        processed_doc = {
            "title": document.get("title", ""),
            "processed_sections": [],
            "metadata": document.get("metadata", {}),
        }
        for section in document.get("sections", []):
            processed_section = dict(section)
            processed_doc["processed_sections"].append(processed_section)
        return processed_doc

    def extract_requirements(self, document: dict[str, Any]) -> list[str]:
        """Extract requirements from a document.

        Args:
            document (Dict[str, Any]): Document to extract requirements from

        Returns:
            List[str]: List of requirements
        """
        requirements = []
        for section_key in ["sections", "processed_sections"]:
            if section_key in document:
                for section in document[section_key]:
                    if "requirements" in section and isinstance(
                        section["requirements"], list
                    ):
                        requirements.extend(section["requirements"])
        return requirements

    def format_for_tasks(self, document: dict[str, Any]) -> dict[str, Any]:
        """Format a document for task generation.

        Args:
            document (Dict[str, Any]): The document to format

        Returns:
            Dict[str, Any]: The formatted document
        """
        formatted_doc = {
            "title": document.get("title", ""),
            "formatted_sections": [],
            "metadata": document.get("metadata", {}),
        }

        for section in document.get("sections", []):
            formatted_section = {
                "name": section.get("name", ""),
                "content": section.get("content", ""),
                "requirements": section.get("requirements", []),
                "steps": self._extract_steps(section),
            }
            formatted_doc["formatted_sections"].append(formatted_section)

        return formatted_doc

    def handle_specific_requirements(self, requirements: list[str]) -> list[str]:
        """Handle specific requirements in a standardized way.

        Args:
            requirements (List[str]): List of requirements to process

        Returns:
            List[str]: Processed requirements
        """
        processed_reqs = []
        for req in requirements:
            # Standardize requirement format
            processed_req = req.strip().capitalize()
            if processed_req not in processed_reqs:
                processed_reqs.append(processed_req)
        return processed_reqs

    def _process_section(self, section: dict[str, Any]) -> dict[str, Any]:
        """Process a document section.

        Args:
            section (Dict[str, Any]): The section to process

        Returns:
            Dict[str, Any]: The processed section
        """
        return {
            "name": section.get("name", ""),
            "content": section.get("content", ""),
            "requirements": section.get("requirements", []),
            "processed_content": self._process_content(section.get("content", "")),
        }

    def _process_content(self, content: str) -> str:
        """Process section content.

        Args:
            content (str): The content to process

        Returns:
            str: The processed content
        """
        # Basic content processing
        return content.strip()

    def _extract_steps(self, section: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract steps from a section.

        Args:
            section (Dict[str, Any]): The section to process

        Returns:
            List[Dict[str, Any]]: List of extracted steps
        """
        steps = []
        if "steps" in section:
            for step in section["steps"]:
                if isinstance(step, dict):
                    steps.append(step)
                else:
                    steps.append({"task": str(step)})
        return steps
