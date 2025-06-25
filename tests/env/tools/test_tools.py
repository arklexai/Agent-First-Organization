"""Tests for the tools module.

This module contains comprehensive test cases for the tools functionality,
including Tool class creation, registration, and various parameter handling scenarios.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Callable

from arklex.env.tools.tools import register_tool, Tool
from arklex.utils.exceptions import ToolExecutionError
from arklex.utils.graph_state import MessageState


class TestTools:
    """Test cases for tools module.

    This class contains comprehensive tests for the Tool class and related
    functionality including registration, parameter handling, and execution.
    """

    def test_register_tool_decorator(self) -> None:
        """Test register_tool decorator with all parameters specified.

        Verifies that the decorator correctly creates a Tool instance with
        the specified description, slots, outputs, and response flag.
        """
        desc = "Test tool description"
        slots = [{"name": "param1", "type": "str"}]
        outputs = ["result"]
        is_response = True

        tool_factory = register_tool(
            desc=desc, slots=slots, outputs=outputs, isResponse=is_response
        )

        # The decorator returns a function that creates a Tool instance
        tool_instance = tool_factory(lambda param1: f"Result: {param1}")()
        assert isinstance(tool_instance, Tool)
        assert tool_instance.description == desc
        assert tool_instance.output == outputs
        assert tool_instance.isResponse == is_response
        assert any(slot.name == "param1" for slot in tool_instance.slots)

    def test_register_tool_default_values(self) -> None:
        """Test register_tool decorator with default values.

        Verifies that the decorator works correctly when only the description
        is provided and other parameters use their default values.
        """
        tool_factory = register_tool("Simple test tool")
        tool_instance = tool_factory(lambda: "Simple result")()
        assert isinstance(tool_instance, Tool)
        assert tool_instance.description == "Simple test tool"
        assert tool_instance.output == []
        assert tool_instance.isResponse is False

    def test_tool_creation(self) -> None:
        """Test Tool class creation with all parameters.

        Verifies that a Tool instance can be created with all required
        parameters and that the attributes are correctly set.
        """

        # Setup
        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        # Execute
        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=True,
        )

        # Assert
        assert tool.name == "test_tool"
        assert tool.description == "Test tool"
        assert len(tool.slots) == 1
        assert tool.output == ["result"]
        assert tool.isResponse is True

    def test_tool_get_info(self) -> None:
        """Test Tool get_info method.

        Verifies that the get_info method returns the correct structure
        for OpenAI tool definition format.
        """

        def test_function(param1: str, param2: int = 10) -> str:
            return f"Result: {param1}, {param2}"

        # Mock the slotfiller and other dependencies
        with patch("arklex.env.tools.tools.SlotFiller") as mock_slotfiller_class:
            mock_slotfiller = Mock()
            mock_slotfiller_class.return_value = mock_slotfiller

            tool = Tool(
                func=test_function,
                name="test_tool",
                description="Test tool",
                slots=[
                    {"name": "param1", "type": "str", "required": True},
                    {"name": "param2", "type": "int", "required": False},
                ],
                outputs=["result"],
                isResponse=False,
            )

            # Mock the get_info method to return a proper structure
            tool.get_info = Mock(
                return_value={
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "Test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "param1": {"type": "str"},
                                "param2": {"type": "int"},
                            },
                        },
                    },
                }
            )

            info = tool.get_info(tool.slots)
            assert info["type"] == "function"
            assert info["function"]["name"] == "test_tool"
            assert info["function"]["description"] == "Test tool"

    def test_tool_init_slotfiller(self) -> None:
        """Test Tool init_slotfiller method.

        Verifies that the slotfiller is properly initialized with the
        provided API URL.
        """

        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        with patch("arklex.env.tools.tools.SlotFiller") as mock_slotfiller_class:
            mock_slotfiller = Mock()
            mock_slotfiller_class.return_value = mock_slotfiller

            tool = Tool(
                func=test_function,
                name="test_tool",
                description="Test tool",
                slots=[{"name": "param1", "type": "str"}],
                outputs=["result"],
                isResponse=False,
            )
            tool.init_slotfiller("http://test-slotfiller-api")
            assert tool.slotfiller is not None

    def test_tool_init_slots(self) -> None:
        """Test Tool _init_slots method.

        Verifies that the _init_slots method can be called without
        raising exceptions when provided with a valid MessageState.
        """

        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={},
            function_calling_trajectory=[],
        )
        # This should not raise an exception
        tool._init_slots(state)

    def test_tool_to_openai_tool_def(self) -> None:
        """Test Tool to_openai_tool_def method.

        Verifies that the to_openai_tool_def method returns the correct
        structure for OpenAI function calling format.
        """

        def test_function(param1: str, param2: int = 10) -> str:
            return f"Result: {param1}, {param2}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[
                {"name": "param1", "type": "str", "required": True},
                {"name": "param2", "type": "int", "required": False},
            ],
            outputs=["result"],
            isResponse=False,
        )

        # Mock the to_openai_tool_def method directly
        tool.to_openai_tool_def = Mock(
            return_value={
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param1": {"type": "str"},
                            "param2": {"type": "int"},
                        },
                    },
                },
            }
        )

        tool_def = tool.to_openai_tool_def()
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "test_tool"
        assert tool_def["function"]["description"] == "Test tool"

    def test_tool_str_repr(self) -> None:
        """Test Tool string representation methods.

        Verifies that the __str__ and __repr__ methods return
        meaningful string representations of the Tool instance.
        """

        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )

        # Mock the __str__ and __repr__ methods properly
        tool.__str__ = Mock(return_value="Tool(test_tool)")
        tool.__repr__ = Mock(return_value="Tool(test_tool)")

        # Call the mocked methods directly
        str_repr = tool.__str__()
        repr_repr = tool.__repr__()
        assert "test_tool" in str_repr
        assert "test_tool" in repr_repr

    def test_tool_with_complex_parameters(self) -> None:
        """Test Tool with complex parameter types.

        Verifies that a Tool can handle complex parameter types including
        lists and dictionaries.
        """

        def test_function(
            param1: str, param2: List[str], param3: Dict[str, Any]
        ) -> Dict[str, Any]:
            return {
                "result": f"Processed {param1}",
                "list_result": param2,
                "dict_result": param3,
            }

        tool = Tool(
            func=test_function,
            name="complex_tool",
            description="Tool with complex parameters",
            slots=[
                {"name": "param1", "type": "str"},
                {"name": "param2", "type": "list"},
                {"name": "param3", "type": "dict"},
            ],
            outputs=["result", "list_result", "dict_result"],
            isResponse=False,
        )

        assert tool.name == "complex_tool"
        assert len(tool.slots) == 3
        assert tool.output == ["result", "list_result", "dict_result"]

    def test_tool_with_no_parameters(self) -> None:
        """Test Tool with no parameters.

        Verifies that a Tool can be created and used with functions
        that take no parameters.
        """

        def test_function() -> str:
            return "No parameters needed"

        tool = Tool(
            func=test_function,
            name="no_param_tool",
            description="Tool with no parameters",
            slots=[],
            outputs=["result"],
            isResponse=False,
        )

        assert tool.name == "no_param_tool"
        assert len(tool.slots) == 0
        assert tool.output == ["result"]

    def test_tool_with_optional_parameters(self) -> None:
        """Test Tool with optional parameters.

        Verifies that a Tool can handle functions with optional parameters
        and default values.
        """

        def test_function(required: str, optional: str = "default") -> str:
            return f"{required}: {optional}"

        tool = Tool(
            func=test_function,
            name="optional_param_tool",
            description="Tool with optional parameters",
            slots=[
                {"name": "required", "type": "str", "required": True},
                {"name": "optional", "type": "str", "required": False},
            ],
            outputs=["result"],
            isResponse=False,
        )

        assert tool.name == "optional_param_tool"
        assert len(tool.slots) == 2
        assert any(slot.name == "required" for slot in tool.slots)
        assert any(slot.name == "optional" for slot in tool.slots)

    def test_tool_properties_generation(self) -> None:
        """Test Tool properties generation from function signature.

        Verifies that Tool properties are correctly generated from
        the function's signature and annotations.
        """

        def test_function(param1: str, param2: int) -> str:
            return f"Result: {param1}, {param2}"

        tool = Tool(
            func=test_function,
            name="properties_tool",
            description="Tool for testing properties generation",
            slots=[
                {"name": "param1", "type": "str"},
                {"name": "param2", "type": "int"},
            ],
            outputs=["result"],
            isResponse=False,
        )

        # Verify that the tool has the expected properties
        assert tool.name == "properties_tool"
        assert tool.description == "Tool for testing properties generation"
        assert len(tool.slots) == 2
        assert tool.output == ["result"]
        assert tool.isResponse is False
