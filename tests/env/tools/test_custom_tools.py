"""
Tests for custom tools.

This module contains comprehensive tests for all custom tools including
HTTP request functionality and placeholder replacement.
"""

from unittest.mock import Mock, patch

import pytest
import requests

from arklex.env.tools.custom_tools.http_tool import http_tool, replace_placeholders
from arklex.utils.exceptions import ToolExecutionError


class TestReplacePlaceholders:
    """Test the replace_placeholders function."""

    def test_replace_placeholders_dict(self) -> None:
        """Test placeholder replacement in dictionaries."""
        data = {"name": "{{user_name}}", "age": "{{user_age}}"}
        slot_map = {"user_name": "John", "user_age": "30"}
        result = replace_placeholders(data, slot_map)
        assert result == {"name": "John", "age": "30"}

    def test_replace_placeholders_list(self) -> None:
        """Test placeholder replacement in lists."""
        data = ["{{user_name}}", "{{user_age}}"]
        slot_map = {"user_name": "John", "user_age": "30"}
        result = replace_placeholders(data, slot_map)
        assert result == ["John", "30"]

    def test_replace_placeholders_string_full_placeholder(self) -> None:
        """Test placeholder replacement for full string placeholders."""
        data = "{{user_name}}"
        slot_map = {"user_name": "John"}
        result = replace_placeholders(data, slot_map)
        assert result == "John"

    def test_replace_placeholders_string_partial_placeholder(self) -> None:
        """Test placeholder replacement for partial string placeholders."""
        data = "Hello {{user_name}}, you are {{user_age}} years old"
        slot_map = {"user_name": "John", "user_age": "30"}
        result = replace_placeholders(data, slot_map)
        assert result == "Hello John, you are 30 years old"

    def test_replace_placeholders_missing_slot(self) -> None:
        """Test placeholder replacement with missing slot."""
        data = "{{user_name}}"
        slot_map = {}
        result = replace_placeholders(data, slot_map)
        assert result is None

    def test_replace_placeholders_none_value(self) -> None:
        """Test placeholder replacement with None value."""
        data = "{{user_name}}"
        slot_map = {"user_name": None}
        result = replace_placeholders(data, slot_map)
        assert result is None

    def test_replace_placeholders_non_string(self) -> None:
        """Test placeholder replacement with non-string values."""
        data = {"count": "{{count}}", "active": "{{active}}"}
        slot_map = {"count": 42, "active": True}
        result = replace_placeholders(data, slot_map)
        # The actual implementation returns the original values, not converted to strings
        assert result == {"count": 42, "active": True}

    def test_replace_placeholders_nested_dict(self) -> None:
        """Test placeholder replacement in nested dictionaries."""
        data = {
            "user": {"name": "{{user_name}}", "age": "{{user_age}}"},
            "settings": {"theme": "{{theme}}"},
        }
        slot_map = {"user_name": "John", "user_age": "30", "theme": "dark"}
        result = replace_placeholders(data, slot_map)
        expected = {
            "user": {"name": "John", "age": "30"},
            "settings": {"theme": "dark"},
        }
        assert result == expected


class TestHTTPTool:
    """Test the HTTP tool functionality."""

    @patch("requests.request")
    def test_http_tool_basic_request(self, mock_request: Mock) -> None:
        """Test basic HTTP request functionality."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        result = http_tool().func(
            method="GET",
            endpoint="https://api.example.com/test",
            headers={"Content-Type": "application/json"},
            body={},
            params={},
        )

        assert "success" in result
        mock_request.assert_called_once()

    @patch("requests.request")
    def test_http_tool_with_slots(self, mock_request: Mock) -> None:
        """Test HTTP request with slot parameters."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        slots = [
            {"name": "user_id", "value": 123, "target": "params"},
            {"name": "user_name", "value": "John", "target": "body"},
        ]

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/users",
            headers={"Content-Type": "application/json"},
            body={"name": "{{user_name}}"},
            params={},
            slots=slots,
        )

        assert "success" in result
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[1]["params"]["user_id"] == 123
        assert call_args[1]["json"]["name"] == "John"

    @patch("requests.request")
    def test_http_tool_with_body_placeholders(self, mock_request: Mock) -> None:
        """Test HTTP request with body placeholders."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        # Create slots with the correct structure
        slots = [
            {"name": "user_name", "value": "John", "target": "body"},
            {"name": "user_age", "value": "30", "target": "body"},
        ]

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/test",
            headers={"Content-Type": "application/json"},
            body={"name": "{{user_name}}", "age": "{{user_age}}"},
            params={},
            slots=slots,
        )

        assert "success" in result
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        # Placeholders should be replaced with slot values
        assert call_args[1]["json"]["name"] == "John"
        assert call_args[1]["json"]["age"] == "30"

    @patch("requests.request")
    def test_http_tool_remove_placeholder_params(self, mock_request: Mock) -> None:
        """Test that placeholder params are removed."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        result = http_tool().func(
            method="GET",
            endpoint="https://api.example.com/test",
            headers={"Content-Type": "application/json"},
            body={},
            params={"optional": "{{optional_param}}"},
        )

        assert "success" in result
        # Verify that placeholder params were removed
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert "optional" not in call_args[1]["params"]

    @patch("requests.request")
    def test_http_tool_remove_placeholder_body(self, mock_request: Mock) -> None:
        """Test that placeholder body fields are removed."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/test",
            headers={"Content-Type": "application/json"},
            body={"name": "test", "optional": "{{optional_field}}"},
            params={},
        )

        assert "success" in result
        # Verify that placeholder body fields were removed
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert "optional" not in call_args[1]["json"]

    @patch("requests.request")
    def test_http_tool_request_exception(self, mock_request: Mock) -> None:
        """Test HTTP tool with request exception."""
        mock_request.side_effect = requests.exceptions.RequestException("Network error")

        with pytest.raises(ToolExecutionError):
            http_tool().func(
                method="GET",
                endpoint="https://api.example.com/test",
                headers={"Content-Type": "application/json"},
                body={},
                params={},
            )

    @patch("requests.request")
    def test_http_tool_general_exception(self, mock_request: Mock) -> None:
        """Test HTTP tool with general exception."""
        mock_request.side_effect = Exception("Unexpected error")

        with pytest.raises(ToolExecutionError):
            http_tool().func(
                method="GET",
                endpoint="https://api.example.com/test",
                headers={"Content-Type": "application/json"},
                body={},
                params={},
            )

    @patch("requests.request")
    def test_http_tool_with_slot_objects(self, mock_request: Mock) -> None:
        """Test HTTP tool with slot objects."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        class SlotObject:
            def __init__(
                self, name: str, value: str | int | bool | None, target: str
            ) -> None:
                self.name = name
                self.value = value
                self.target = target

        slots = [
            SlotObject("user_id", 123, "params"),
            SlotObject("user_name", "John", "body"),
        ]

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/users",
            headers={"Content-Type": "application/json"},
            body={"name": "{{user_name}}"},
            params={},
            slots=slots,
        )

        assert "success" in result
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[1]["params"]["user_id"] == 123
        assert call_args[1]["json"]["name"] == "John"

    @patch("requests.request")
    def test_http_tool_with_mixed_slot_types(self, mock_request: Mock) -> None:
        """Test HTTP tool with mixed slot types."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        class SlotObject:
            def __init__(
                self, name: str, value: str | int | bool | None, target: str
            ) -> None:
                self.name = name
                self.value = value
                self.target = target

        slots = [
            {"name": "user_id", "value": 123, "target": "params"},
            SlotObject("user_name", "John", "body"),
        ]

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/users",
            headers={"Content-Type": "application/json"},
            body={"name": "{{user_name}}"},
            params={},
            slots=slots,
        )

        assert "success" in result
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[1]["params"]["user_id"] == 123
        assert call_args[1]["json"]["name"] == "John"

    @patch("requests.request")
    def test_http_tool_with_invalid_slots(self, mock_request: Mock) -> None:
        """Test HTTP tool with invalid slot structure."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        # Slots with missing required attributes
        slots = [
            {"name": "user_id"},  # Missing value and target
            {"value": "John"},  # Missing name and target
        ]

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/users",
            headers={"Content-Type": "application/json"},
            body={},
            params={},
            slots=slots,
        )

        # Should still work without the invalid slots
        assert "success" in result
        mock_request.assert_called_once()

    @patch("requests.request")
    def test_http_tool_with_none_slots(self, mock_request: Mock) -> None:
        """Test HTTP tool with None slots parameter."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        result = http_tool().func(
            method="GET",
            endpoint="https://api.example.com/test",
            headers={"Content-Type": "application/json"},
            body={},
            params={},
            slots=None,
        )

        assert "success" in result
        mock_request.assert_called_once()


class TestHTTPToolIntegration:
    """Integration tests for HTTP tool."""

    @patch("requests.request")
    def test_http_tool_complete_workflow(self, mock_request: Mock) -> None:
        """Test complete HTTP tool workflow."""
        mock_response = Mock()
        mock_response.json.return_value = {"id": 1, "name": "John", "status": "active"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        slots = [
            {"name": "user_id", "value": 1, "target": "params"},
            {"name": "user_name", "value": "John", "target": "body"},
        ]

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/users",
            headers={"Content-Type": "application/json"},
            body={"name": "{{user_name}}"},
            params={"id": "{{user_id}}"},
            slots=slots,
        )

        assert "id" in result
        assert "name" in result
        assert "status" in result
        mock_request.assert_called_once()

        # Verify the request was made with correct parameters
        call_args = mock_request.call_args
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["url"] == "https://api.example.com/users"
        assert call_args[1]["json"]["name"] == "John"
        assert call_args[1]["params"]["user_id"] == 1
