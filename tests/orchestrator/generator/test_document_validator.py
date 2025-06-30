"""Tests for the DocumentValidator component.

This module provides comprehensive tests for the DocumentValidator class,
including structure, required fields, consistency, and error reporting.
"""

from typing import Any

import pytest

from arklex.orchestrator.generator.docs.document_validator import DocumentValidator


@pytest.fixture
def validator() -> DocumentValidator:
    """Fixture for DocumentValidator instance."""
    return DocumentValidator()


@pytest.fixture
def valid_document() -> dict[str, Any]:
    """Fixture for a valid document structure."""
    return {
        "title": "Test Document",
        "sections": [
            {"name": "Section 1", "content": "Some content", "requirements": ["req1"]},
            {"name": "Section 2", "content": "Other content", "requirements": ["req2"]},
        ],
    }


def make_doc_missing_title(doc: dict[str, Any]) -> dict[str, Any]:
    d = doc.copy()
    d.pop("title", None)
    return d


def make_doc_invalid_sections_type(doc: dict[str, Any]) -> dict[str, Any]:
    d = doc.copy()
    d["sections"] = "not a list"
    return d


def make_doc_invalid_section(doc: dict[str, Any]) -> dict[str, Any]:
    d = doc.copy()
    d["sections"][0] = {"name": "Section 1"}  # missing content
    return d


def make_doc_missing_section_field(doc: dict[str, Any]) -> dict[str, Any]:
    d = doc.copy()
    d["sections"][0] = {"content": "no name"}
    return d


def make_doc_duplicate_section_names(doc: dict[str, Any]) -> dict[str, Any]:
    d = doc.copy()
    d["sections"][1]["name"] = "Section 1"
    return d


def make_doc_empty_requirements(doc: dict[str, Any]) -> dict[str, Any]:
    d = doc.copy()
    d["sections"][0]["requirements"] = []
    return d


class TestDocumentValidator:
    """Test suite for DocumentValidator component."""

    def test_validate_structure_valid(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test valid document structure."""
        assert validator.validate_structure(valid_document) is True

    def test_validate_structure_missing_field(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test structure validation with missing title."""
        doc = make_doc_missing_title(valid_document)
        assert validator.validate_structure(doc) is False

    def test_validate_structure_invalid_sections_type(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test structure validation with invalid sections type."""
        doc = make_doc_invalid_sections_type(valid_document)
        assert validator.validate_structure(doc) is False

    def test_validate_structure_invalid_section(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test structure validation with invalid section."""
        doc = make_doc_invalid_section(valid_document)
        assert validator.validate_structure(doc) is False

    def test_validate_required_fields_valid(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test required fields validation for valid document."""
        assert validator.validate_required_fields(valid_document) is True

    def test_validate_required_fields_missing_doc_field(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test required fields validation with missing doc field."""
        doc = make_doc_missing_title(valid_document)
        assert validator.validate_required_fields(doc) is False

    def test_validate_required_fields_missing_section_field(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test required fields validation with missing section field."""
        doc = make_doc_missing_section_field(valid_document)
        assert validator.validate_required_fields(doc) is False

    def test_validate_consistency_valid(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test consistency validation for valid document."""
        assert validator.validate_consistency(valid_document) is True

    def test_validate_consistency_duplicate_section_names(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test consistency validation with duplicate section names."""
        doc = make_doc_duplicate_section_names(valid_document)
        assert validator.validate_consistency(doc) is False

    def test_validate_consistency_empty_requirements(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test consistency validation with empty requirements."""
        doc = make_doc_empty_requirements(valid_document)
        assert validator.validate_consistency(doc) is False

    def test_get_error_messages_all_valid(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test error messages for a valid document (should be empty)."""
        assert validator.get_error_messages(valid_document) == []

    def test_get_error_messages_invalid_structure(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test error messages for invalid structure."""
        doc = make_doc_missing_title(valid_document)
        errors = validator.get_error_messages(doc)
        assert "Invalid document structure" in errors

    def test_get_error_messages_missing_required_fields(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test error messages for missing required fields."""
        doc = make_doc_missing_section_field(valid_document)
        errors = validator.get_error_messages(doc)
        assert "Missing required fields" in errors

    def test_get_error_messages_consistency(
        self, validator: DocumentValidator, valid_document: dict[str, Any]
    ) -> None:
        """Test error messages for consistency issues."""
        doc = make_doc_duplicate_section_names(valid_document)
        errors = validator.get_error_messages(doc)
        assert "Document consistency issues" in errors

    def test__validate_section_valid(self, validator: DocumentValidator) -> None:
        """Test private section validation for valid section."""
        section = {"name": "Section", "content": "Content", "requirements": ["req"]}
        assert validator._validate_section(section) is True

    def test__validate_section_missing_field(
        self, validator: DocumentValidator
    ) -> None:
        """Test private section validation with missing field."""
        section = {"content": "Content"}
        assert validator._validate_section(section) is False

    def test__validate_section_empty_content(
        self, validator: DocumentValidator
    ) -> None:
        """Test private section validation with empty content."""
        section = {"name": "Section", "content": "", "requirements": ["req"]}
        assert validator._validate_section(section) is False

    def test__validate_section_invalid_requirements(
        self, validator: DocumentValidator
    ) -> None:
        """Test private section validation with invalid requirements type."""
        section = {"name": "Section", "content": "Content", "requirements": "notalist"}
        assert validator._validate_section(section) is False
