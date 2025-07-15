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

    def test_replace_placeholders_string_partial_placeholder_returns_original(self) -> None:
        """Test that partial placeholder strings are returned unchanged (no partial support)."""
        data = "Hello {{user_name}}, you are {{user_age}} years old"
        slot_map = {"user_name": {"value": "John"}, "user_age": {"value": "30"}}
        result = replace_placeholders(data, slot_map)
        assert result == data

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
        assert result == data

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

    def test_replace_placeholders_all_slot_types(self) -> None:
        """Test replace_placeholders with all slot_type scenarios."""
        # Test slot_type="list" - should return []
        data_list = "{{list_slot}}"
        slot_map_list = {"list_slot": {"type": "list", "value": None}}
        assert replace_placeholders(data_list, slot_map_list) == []
        
        # Test slot_type="float" - should return 0.0
        data_float = "{{float_slot}}"
        slot_map_float = {"float_slot": {"type": "float", "value": None}}
        assert replace_placeholders(data_float, slot_map_float) == 0.0
        
        # Test slot_type="bool" - should return False
        data_bool = "{{bool_slot}}"
        slot_map_bool = {"bool_slot": {"type": "bool", "value": None}}
        assert replace_placeholders(data_bool, slot_map_bool) is False
        
        # Test slot_type="int" - should return 0
        data_int = "{{int_slot}}"
        slot_map_int = {"int_slot": {"type": "int", "value": None}}
        assert replace_placeholders(data_int, slot_map_int) == 0
        
        # Test slot_type="str" - should return ""
        data_str = "{{str_slot}}"
        slot_map_str = {"str_slot": {"type": "str", "value": None}}
        assert replace_placeholders(data_str, slot_map_str) == ""
        
        # Test slot_type="string" - should return ""
        data_string = "{{string_slot}}"
        slot_map_string = {"string_slot": {"type": "string", "value": None}}
        assert replace_placeholders(data_string, slot_map_string) == ""
        
        # Test slot_type="integer" - should return 0
        data_integer = "{{integer_slot}}"
        slot_map_integer = {"integer_slot": {"type": "integer", "value": None}}
        assert replace_placeholders(data_integer, slot_map_integer) == 0
        
        # Test slot_type="unsigned int" - should return original string (not handled)
        data_uint = "{{uint_slot}}"
        slot_map_uint = {"uint_slot": {"type": "unsigned int", "value": None}}
        assert replace_placeholders(data_uint, slot_map_uint) == data_uint
        
        # Test slot_type=None - should return original string
        data_none = "{{none_slot}}"
        slot_map_none = {"none_slot": {"type": None, "value": None}}
        assert replace_placeholders(data_none, slot_map_none) == data_none
        
        # Test slot_type="unknown" - should return original string
        data_unknown = "{{unknown_slot}}"
        slot_map_unknown = {"unknown_slot": {"type": "unknown", "value": None}}
        assert replace_placeholders(data_unknown, slot_map_unknown) == data_unknown
        
        # Test partial string replacements: should return original string
        data_partial = "Hello {{list_slot}}, your score is {{float_slot}}, active: {{bool_slot}}"
        slot_map_partial = {
            "list_slot": {"type": "list", "value": None},
            "float_slot": {"type": "float", "value": None},
            "bool_slot": {"type": "bool", "value": None}
        }
        result_partial = replace_placeholders(data_partial, slot_map_partial)
        assert result_partial == data_partial

    def test_replace_placeholders_partial_string_str_and_int_types_returns_original(self) -> None:
        """Test that partial string with slot types returns original string (no partial support)."""
        # str type
        data = "User: {{str_slot}}!"
        slot_map = {"str_slot": {"type": "str", "value": None}}
        result = replace_placeholders(data, slot_map)
        assert result == data

        # string type
        data2 = "User: {{string_slot}}!"
        slot_map2 = {"string_slot": {"type": "string", "value": None}}
        result2 = replace_placeholders(data2, slot_map2)
        assert result2 == data2

        # int type
        data3 = "Count: {{int_slot}} apples"
        slot_map3 = {"int_slot": {"type": "int", "value": None}}
        result3 = replace_placeholders(data3, slot_map3)
        assert result3 == data3

        # integer type
        data4 = "Count: {{integer_slot}} bananas"
        slot_map4 = {"integer_slot": {"type": "integer", "value": None}}
        result4 = replace_placeholders(data4, slot_map4)
        assert result4 == data4

    def test_replace_placeholders_unknown_slot_and_type(self) -> None:
        """Test replace_placeholders with unknown slots and types."""
        # Unknown type and missing slot
        data = "{{unknown}}"
        slot_map = {"known": {"value": None, "type": "str"}, "other": {"value": None, "type": "weirdtype"}}
        # Should return original string for missing slot
        assert replace_placeholders(data, slot_map) == ""
        # Unknown type branch
        data2 = "{{other}}"
        assert replace_placeholders(data2, slot_map) == data2
        # Partial string with unknown slot: should return original string
        data3 = "Hello {{unknown}}"
        assert replace_placeholders(data3, slot_map) == data3
        # Inner repl unknown type: should return original string
        data4 = "Value: {{foo}}"
        slot_map4 = {"foo": {"value": None, "type": "strange"}}
        assert replace_placeholders(data4, slot_map4) == data4
        # Non-dict, non-list, non-str
        assert replace_placeholders(123, {}) == 123
        assert replace_placeholders(12.5, {}) == 12.5


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
    def test_http_tool_with_slot_objects(self, mock_request: Mock) -> None:
        """Test HTTP tool with slot objects."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        class SlotObject:
            def __init__(self, name: str, value: str | int | bool | None, target: str) -> None:
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
            def __init__(self, name: str, value: str | int | bool | None, target: str) -> None:
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
    def test_http_tool_with_slot_object_missing_attributes(self, mock_request: Mock) -> None:
        """Test HTTP tool with slot object missing attributes."""
        mock_response = Mock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        class SlotObj:
            pass  # No name, value, or target

        slots = [SlotObj()]
        result = http_tool().func(
            method="GET",
            endpoint="http://x", 
            headers={}, 
            body={}, 
            params={}, 
            slots=slots
        )
        assert "ok" in result

    @patch("requests.request")
    def test_http_tool_with_slot_object_none_name(self, mock_request: Mock) -> None:
        """Test HTTP tool with slot object having None name."""
        mock_response = Mock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        class SlotObj:
            def __init__(self) -> None:
                self.name = None
                self.value = "test"
                self.target = "body"

        slots = [SlotObj()]
        result = http_tool().func(
            method="GET", 
            endpoint="http://x", 
            headers={}, 
            body={}, 
            params={},
            slots=slots
        )
        assert "ok" in result

    @patch("requests.request")
    def test_http_tool_remove_placeholders_and_valid_json(self, mock_request: Mock) -> None:
        """Test remove_placeholders with empty dict and http_tool with valid JSON body."""
        mock_response = Mock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        # Test remove_placeholders with empty dict
        result = http_tool().func(
            method="POST",
            endpoint="http://x",
            headers={"Content-Type": "application/json"},
            body={},  # Empty dict
            params={},  # Empty dict
        )
        assert "ok" in result
        
        # Test http_tool with valid JSON body
        result2 = http_tool().func(
            method="POST",
            endpoint="http://x",
            headers={"Content-Type": "application/json"},
            body={"name": "test", "age": 30, "active": True},  # Valid JSON
            params={"page": 1, "limit": 10},  # Valid params
        )
        assert "ok" in result2

    @patch("requests.request")
    def test_http_tool_invalid_json_scenarios(self, mock_request: Mock) -> None:
        """Test http_tool with invalid JSON scenarios to cover error handling branches."""
        mock_response = Mock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        # Test with non-serializable object that raises exception in __str__
        class NonSerializable:
            def __str__(self) -> str:
                raise Exception("Cannot serialize")

        result = http_tool().func(
            method="POST",
            endpoint="http://x",
            headers={"Content-Type": "application/json"},
            body={"problematic": NonSerializable()},
            params={},
        )
        assert "ok" in result
        
        # Test with complex nested structure that might cause JSON issues
        class CircularReference:
            def __init__(self) -> None:
                self.self_ref = self
        
        result2 = http_tool().func(
            method="POST",
            endpoint="http://x",
            headers={"Content-Type": "application/json"},
            body={"circular": CircularReference()},
            params={},
        )
        assert "ok" in result2
        
        # Test with function object (not JSON serializable)
        def some_function() -> None:
            pass
        
        result3 = http_tool().func(
            method="POST",
            endpoint="http://x",
            headers={"Content-Type": "application/json"},
            body={"function": some_function},
            params={},
        )
        assert "ok" in result3

    def test_http_tool_json_parsing_error(self) -> None:
        """Test HTTP tool with JSON parsing error."""
        # Mock requests to return non-JSON response
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "Raw text response"
        mock_response.raise_for_status.return_value = None
    
        with patch('requests.request', return_value=mock_response):
            result = http_tool().func(
                method="GET",
                endpoint="https://api.example.com/test",
                headers={},
                body=None,
                params={}
            )
            assert "raw_response" in result
            assert "error" in result

    @patch("requests.request")
    def test_http_tool_remove_placeholders_with_unreplaced_placeholders(self, mock_request: Mock) -> None:
        """Test remove_placeholders with unreplaced placeholders."""
        mock_response = Mock()
        mock_response.json.return_value = {"ok": True}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        # Test with unreplaced placeholders in params and body
        result = http_tool().func(
            method="POST",
            endpoint="http://x",
            headers={"Content-Type": "application/json"},
            body={
                "normal_field": "value",
                "unreplaced_placeholder": "{{unreplaced_slot}}",
                "mixed": "Hello {{unreplaced_name}}, welcome!"
            },
            params={
                "normal_param": "value",
                "unreplaced_param": "{{unreplaced_param}}"
            },
        )
        assert "ok" in result
        mock_request.assert_called_once()
        
        # Verify that unreplaced placeholders were removed from params
        call_args = mock_request.call_args
        params = call_args[1]["params"]
        body = call_args[1]["json"]
        
        # Unreplaced placeholders should be removed from params
        assert "normal_param" in params
        assert "unreplaced_param" not in params
        
        # Unreplaced placeholders should be set to empty string in body
        assert body["normal_field"] == "value"
        assert body["unreplaced_placeholder"] == ""
        assert body["mixed"] == ""


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
        assert result["level1"]["level2"]["mixed"] == ""
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
        assert result["items"][1] == "{{placeholder_item}}"
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
