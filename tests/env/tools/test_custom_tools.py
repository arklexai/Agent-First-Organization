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
        slot_map = {"user_name": {"value": "John"}, "user_age": {"value": "30"}}
        result = replace_placeholders(data, slot_map)
        assert result == {"name": "John", "age": "30"}

    def test_replace_placeholders_list(self) -> None:
        """Test placeholder replacement in lists."""
        data = ["{{user_name}}", "{{user_age}}"]
        slot_map = {"user_name": {"value": "John"}, "user_age": {"value": "30"}}
        result = replace_placeholders(data, slot_map)
        assert result == ["John", "30"]

    def test_replace_placeholders_string_full_placeholder(self) -> None:
        """Test placeholder replacement for full string placeholders."""
        data = "{{user_name}}"
        slot_map = {"user_name": {"value": "John"}}
        result = replace_placeholders(data, slot_map)
        assert result == "John"

    def test_replace_placeholders_string_partial_placeholder(self) -> None:
        """Test placeholder replacement for partial string placeholders."""
        data = "Hello {{user_name}}, you are {{user_age}} years old"
        slot_map = {"user_name": {"value": "John"}, "user_age": {"value": "30"}}
        result = replace_placeholders(data, slot_map)
        assert result == "Hello John, you are 30 years old"

    def test_replace_placeholders_missing_slot(self) -> None:
        """Test placeholder replacement with missing slot."""
        data = "{{user_name}}"
        slot_map = {}
        result = replace_placeholders(data, slot_map)
        assert result == ""

    def test_replace_placeholders_none_value(self) -> None:
        """Test placeholder replacement with None value."""
        data = "{{user_name}}"
        slot_map = {"user_name": {"value": None}}
        result = replace_placeholders(data, slot_map)
        assert result is None

    def test_replace_placeholders_non_string(self) -> None:
        """Test placeholder replacement with non-string values."""
        data = {"count": "{{count}}", "active": "{{active}}"}
        slot_map = {"count": {"value": 42}, "active": {"value": True}}
        result = replace_placeholders(data, slot_map)
        assert result == {"count": 42, "active": True}

    def test_replace_placeholders_nested_dict(self) -> None:
        """Test placeholder replacement in nested dictionaries."""
        data = {
            "user": {"name": "{{user_name}}", "age": "{{user_age}}"},
            "settings": {"theme": "{{theme}}"},
        }
        slot_map = {"user_name": {"value": "John"}, "user_age": {"value": "30"}, "theme": {"value": "dark"}}
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
        # Verify that placeholder body fields were set to empty string
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[1]["json"]["optional"] == ""

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


class TestHTTPToolDataCleaning:
    """Test the data cleaning functionality in HTTP tool."""

    def test_clean_json_data_nested_dict(self) -> None:
        """Test cleaning nested dictionary data."""
        from arklex.env.tools.custom_tools.http_tool import clean_json_data
        
        data = {
            "level1": {
                "level2": {
                    "normal": "value",
                    "placeholder": "{{test_placeholder}}",
                    "mixed": "Hello {{name}}, how are you?"
                }
            },
            "list_data": [
                {"item": "{{item1}}"},
                {"item": "normal_item"}
            ]
        }
        
        result = clean_json_data(data)
        
        # Placeholders should be replaced with empty strings
        assert result["level1"]["level2"]["placeholder"] == ""
        assert result["level1"]["level2"]["mixed"] == "Hello , how are you?"
        assert result["list_data"][0]["item"] == ""
        assert result["list_data"][1]["item"] == "normal_item"

    def test_clean_json_data_non_dict_input(self) -> None:
        """Test clean_json_data with non-dict input."""
        from arklex.env.tools.custom_tools.http_tool import clean_json_data
        
        # Should return input as-is for non-dict types
        assert clean_json_data("string") == "string"
        assert clean_json_data(123) == 123
        assert clean_json_data(None) is None
        assert clean_json_data(["list"]) == ["list"]

    def test_clean_json_data_empty_dict(self) -> None:
        """Test clean_json_data with empty dictionary."""
        from arklex.env.tools.custom_tools.http_tool import clean_json_data
        
        result = clean_json_data({})
        assert result == {}

    def test_clean_json_data_with_list_values(self) -> None:
        """Test clean_json_data with list values containing placeholders."""
        from arklex.env.tools.custom_tools.http_tool import clean_json_data
        
        data = {
            "items": [
                "normal_item",
                "{{placeholder_item}}",
                {"nested": "{{nested_placeholder}}"}
            ]
        }
        
        result = clean_json_data(data)
        assert result["items"][0] == "normal_item"
        assert result["items"][1] == ""
        assert result["items"][2]["nested"] == ""

    def test_validate_request_body_success(self) -> None:
        """Test validate_request_body with valid data."""
        from arklex.env.tools.custom_tools.http_tool import validate_request_body
        
        body = {
            "name": "test",
            "age": 30,
            "active": True
        }
        
        result = validate_request_body(body)
        assert result == body

    def test_validate_request_body_with_placeholders(self) -> None:
        """Test validate_request_body with placeholders."""
        from arklex.env.tools.custom_tools.http_tool import validate_request_body
        
        body = {
            "name": "{{user_name}}",
            "age": "{{user_age}}",
            "normal": "value"
        }
        
        result = validate_request_body(body)
        assert result["name"] == ""
        assert result["age"] == ""
        assert result["normal"] == "value"

    def test_validate_request_body_none_input(self) -> None:
        """Test validate_request_body with None input."""
        from arklex.env.tools.custom_tools.http_tool import validate_request_body
        
        result = validate_request_body(None)
        assert result is None

    def test_validate_request_body_invalid_json(self) -> None:
        """Test validate_request_body with data that can't be JSON serialized."""
        from arklex.env.tools.custom_tools.http_tool import validate_request_body
        
        # Create an object that can't be JSON serialized
        class NonSerializable:
            pass
        
        body = {
            "normal": "value",
            "problematic": NonSerializable()
        }
        
        result = validate_request_body(body)
        assert "error" in result
        assert "Invalid request body" in result["error"]


class TestReplacePlaceholdersAdvanced:
    """Test advanced placeholder replacement scenarios."""

    def test_replace_placeholders_with_type_defaults(self) -> None:
        """Test placeholder replacement with type-based defaults."""
        from arklex.env.tools.custom_tools.http_tool import replace_placeholders
        
        data = "{{user_name}}"
        slot_map = {
            "user_name": {"type": "str", "value": None}
        }
        
        result = replace_placeholders(data, slot_map)
        assert result == ""

    def test_replace_placeholders_with_int_type_default(self) -> None:
        """Test placeholder replacement with int type default."""
        from arklex.env.tools.custom_tools.http_tool import replace_placeholders
        
        data = "{{user_age}}"
        slot_map = {
            "user_age": {"type": "int", "value": None}
        }
        
        result = replace_placeholders(data, slot_map)
        assert result == 0

    def test_replace_placeholders_with_float_type_default(self) -> None:
        """Test placeholder replacement with float type default."""
        from arklex.env.tools.custom_tools.http_tool import replace_placeholders
        
        data = "{{user_score}}"
        slot_map = {
            "user_score": {"type": "float", "value": None}
        }
        
        result = replace_placeholders(data, slot_map)
        assert result == 0.0

    def test_replace_placeholders_with_bool_type_default(self) -> None:
        """Test placeholder replacement with bool type default."""
        from arklex.env.tools.custom_tools.http_tool import replace_placeholders
        
        data = "{{user_active}}"
        slot_map = {
            "user_active": {"type": "bool", "value": None}
        }
        
        result = replace_placeholders(data, slot_map)
        assert result is False

    def test_replace_placeholders_with_list_type_default(self) -> None:
        """Test placeholder replacement with list type default."""
        from arklex.env.tools.custom_tools.http_tool import replace_placeholders
        
        data = "{{user_tags}}"
        slot_map = {
            "user_tags": {"type": "list", "value": None}
        }
        
        result = replace_placeholders(data, slot_map)
        assert result == []

    def test_replace_placeholders_with_unknown_type_default(self) -> None:
        """Test placeholder replacement with unknown type default."""
        from arklex.env.tools.custom_tools.http_tool import replace_placeholders
        
        data = "{{user_data}}"
        slot_map = {
            "user_data": {"type": "unknown", "value": None}
        }
        
        result = replace_placeholders(data, slot_map)
        assert result is None

    def test_replace_placeholders_partial_with_type_defaults(self) -> None:
        """Test partial placeholder replacement with type defaults."""
        from arklex.env.tools.custom_tools.http_tool import replace_placeholders
        
        data = "Hello {{user_name}}, your age is {{user_age}}"
        slot_map = {
            "user_name": {"type": "str", "value": None},
            "user_age": {"type": "int", "value": None}
        }
        
        result = replace_placeholders(data, slot_map)
        assert result == "Hello , your age is 0"

    def test_replace_placeholders_with_boolean_string_values(self) -> None:
        """Test placeholder replacement with boolean string values."""
        from arklex.env.tools.custom_tools.http_tool import replace_placeholders
        
        data = "{{is_active}}"
        slot_map = {
            "is_active": {"value": True, "type": "bool"}
        }
        
        result = replace_placeholders(data, slot_map)
        assert result is True

    def test_replace_placeholders_with_numeric_string_values(self) -> None:
        """Test placeholder replacement with numeric string values."""
        from arklex.env.tools.custom_tools.http_tool import replace_placeholders
        
        data = "{{count}}"
        slot_map = {
            "count": {"value": 42, "type": "int"}
        }
        
        result = replace_placeholders(data, slot_map)
        assert result == 42


class TestHTTPToolAdvanced:
    """Test advanced HTTP tool functionality."""

    @patch("requests.request")
    def test_http_tool_with_invalid_json_body(self, mock_request: Mock) -> None:
        """Test HTTP tool with invalid JSON body."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        # Create a body with non-serializable content
        class NonSerializable:
            pass

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/test",
            headers={"Content-Type": "application/json"},
            body={"normal": "value", "problematic": NonSerializable()},
            params={},
        )

        assert "success" in result
        mock_request.assert_called_once()

    @patch("requests.request")
    def test_http_tool_with_complex_placeholder_removal(self, mock_request: Mock) -> None:
        """Test HTTP tool with complex placeholder removal scenarios."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/test",
            headers={"Content-Type": "application/json"},
            body={
                "normal_field": "value",
                "placeholder_field": "{{missing_placeholder}}",
                "mixed_field": "Hello {{name}}, welcome!",
                "nested": {
                    "inner_placeholder": "{{inner_missing}}"
                }
            },
            params={
                "normal_param": "value",
                "placeholder_param": "{{missing_param}}"
            },
        )

        assert "success" in result
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        
        # Check that placeholders were handled correctly
        body = call_args[1]["json"]
        params = call_args[1]["params"]
        
        assert body["normal_field"] == "value"
        assert body["placeholder_field"] == ""
        assert body["mixed_field"] == "Hello , welcome!"
        assert body["nested"]["inner_placeholder"] == ""
        assert params["normal_param"] == "value"
        assert "placeholder_param" not in params  # Should be removed

    @patch("requests.request")
    def test_http_tool_with_empty_slots(self, mock_request: Mock) -> None:
        """Test HTTP tool with empty slots list."""
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
            slots=[],
        )

        assert "success" in result
        mock_request.assert_called_once()

    @patch("requests.request")
    def test_http_tool_with_slot_missing_attributes(self, mock_request: Mock) -> None:
        """Test HTTP tool with slots missing required attributes."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        # Slots with missing attributes
        slots = [
            {"name": "user_id", "value": 123},  # Missing target
            {"value": "John", "target": "body"},  # Missing name
            {"name": "user_name", "target": "body"},  # Missing value
        ]

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/test",
            headers={"Content-Type": "application/json"},
            body={},
            params={},
            slots=slots,
        )

        assert "success" in result
        mock_request.assert_called_once()

    @patch("requests.request")
    def test_http_tool_with_none_body_and_params(self, mock_request: Mock) -> None:
        """Test HTTP tool with None body and params."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        result = http_tool().func(
            method="GET",
            endpoint="https://api.example.com/test",
            headers={"Content-Type": "application/json"},
            body=None,
            params=None,
        )

        assert "success" in result
        mock_request.assert_called_once()

    @patch("requests.request")
    def test_http_tool_with_complex_slot_objects(self, mock_request: Mock) -> None:
        """Test HTTP tool with complex slot objects."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        class ComplexSlotObject:
            def __init__(self, name: str, value: str, target: str, type: str = "str", description: str = ""):
                self.name = name
                self.value = value
                self.target = target
                self.type = type
                self.description = description

        slots = [
            ComplexSlotObject("user_id", 123, "params", "int", "User ID"),
            ComplexSlotObject("user_name", "John", "body", "str", "User name"),
            ComplexSlotObject("is_active", True, "body", "bool", "Active status"),
        ]

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/test",
            headers={"Content-Type": "application/json"},
            body={"name": "{{user_name}}", "active": "{{is_active}}"},
            params={},
            slots=slots,
        )

        assert "success" in result
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert call_args[1]["params"]["user_id"] == 123
        assert call_args[1]["json"]["name"] == "John"
        assert call_args[1]["json"]["active"] is True

    @patch("requests.request")
    def test_http_tool_with_mixed_slot_types_and_placeholders(self, mock_request: Mock) -> None:
        """Test HTTP tool with mixed slot types and placeholders."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        slots = [
            {"name": "user_id", "value": 123, "target": "params", "type": "int"},
            {"name": "user_name", "value": "John", "target": "body", "type": "str"},
            {"name": "missing_field", "value": None, "target": "body", "type": "str"},
        ]

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/test",
            headers={"Content-Type": "application/json"},
            body={
                "name": "{{user_name}}",
                "missing": "{{missing_field}}",
                "unresolved": "{{unresolved_placeholder}}"
            },
            params={"id": "{{user_id}}"},
            slots=slots,
        )

        assert "success" in result
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        
        # Check that resolved placeholders are replaced
        assert call_args[1]["json"]["name"] == "John"
        assert call_args[1]["json"]["missing"] == ""
        assert call_args[1]["json"]["unresolved"] == ""
        assert call_args[1]["params"]["user_id"] == 123

    @patch("requests.request")
    def test_http_tool_with_validation_errors(self, mock_request: Mock) -> None:
        """Test HTTP tool with JSON validation errors."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        # Create a body that would cause JSON serialization issues
        class ProblematicObject:
            def __str__(self):
                raise Exception("Serialization error")

        result = http_tool().func(
            method="POST",
            endpoint="https://api.example.com/test",
            headers={"Content-Type": "application/json"},
            body={"problematic": ProblematicObject()},
            params={},
        )

        # Should still work, but with error handling
        assert "success" in result
        mock_request.assert_called_once()
