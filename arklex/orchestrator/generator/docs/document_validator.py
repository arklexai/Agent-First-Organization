"""Document validation component for the Arklex framework.

This module provides the DocumentValidator class that handles the validation
of document structure, content, and consistency.
"""

from typing import Any


class DocumentValidator:
    """Validates documents for task generation.

    This class ensures that documents meet the required format and contain
    all necessary information for task generation. It performs various
    validation checks and provides detailed error messages.

    Attributes:
        _required_fields (List[str]): List of required document fields
        _section_fields (List[str]): List of required section fields
    """

    def __init__(self) -> None:
        """Initialize the DocumentValidator."""
        self._required_fields = ["title", "sections"]
        self._section_fields = ["name", "content"]

    def validate_structure(self, document: dict[str, Any]) -> bool:
        """Validate document structure.

        Args:
            document (Dict[str, Any]): The document to validate

        Returns:
            bool: True if document structure is valid
        """
        # Check required fields
        if not all(field in document for field in self._required_fields):
            return False

        # Check sections
        if not isinstance(document["sections"], list):
            return False

        # Check each section
        return all(self._validate_section(section) for section in document["sections"])

    def validate_required_fields(self, document: dict[str, Any]) -> bool:
        """Validate required fields in document.

        Args:
            document (Dict[str, Any]): The document to validate

        Returns:
            bool: True if all required fields are present
        """
        # Check document fields
        if not all(field in document for field in self._required_fields):
            return False

        # Check section fields
        for section in document.get("sections", []):
            if not all(field in section for field in self._section_fields):
                return False

        return True

    def validate_consistency(self, document: dict[str, Any]) -> bool:
        """Validate document consistency.

        Args:
            document (Dict[str, Any]): The document to validate

        Returns:
            bool: True if document is consistent
        """
        # Check section names are unique
        section_names = [
            section.get("name", "") for section in document.get("sections", [])
        ]
        if len(section_names) != len(set(section_names)):
            return False

        # Check requirements are non-empty
        for section in document.get("sections", []):
            if not section.get("requirements"):
                return False

        return True

    def get_error_messages(self, document: dict[str, Any]) -> list[str]:
        """Get validation error messages.

        Args:
            document (Dict[str, Any]): The document to validate

        Returns:
            List[str]: List of error messages
        """
        errors = []

        # Check structure
        if not self.validate_structure(document):
            errors.append("Invalid document structure")

        # Check required fields
        if not self.validate_required_fields(document):
            errors.append("Missing required fields")

        # Check consistency
        if not self.validate_consistency(document):
            errors.append("Document consistency issues")

        return errors

    def _validate_section(self, section: dict[str, Any]) -> bool:
        """Validate a document section.

        Args:
            section (Dict[str, Any]): The section to validate

        Returns:
            bool: True if section is valid
        """
        # Check required fields
        if not all(field in section for field in self._section_fields):
            return False

        # Check content is non-empty
        if not section.get("content"):
            return False

        # Check requirements
        return not (
            "requirements" in section and not isinstance(section["requirements"], list)
        )
