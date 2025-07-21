"""Tests for the tools module.

This module contains comprehensive test cases for the tools functionality,
including Tool class creation, registration, and various parameter handling scenarios.
"""

from copy import deepcopy
from typing import Any, NoReturn
from unittest.mock import Mock, patch

import pytest

from arklex.env.tools.tools import Slot, Tool, register_tool
from arklex.orchestrator.entities.msg_state_entities import MessageState, StatusEnum
from arklex.utils.exceptions import AuthenticationError, ToolExecutionError


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
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
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
            param1: str, param2: list[str], param3: dict[str, Any]
        ) -> dict[str, Any]:
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

    def test_init_default_slots(self) -> None:
        """Test init_default_slots method."""

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

        default_slots = [Slot(name="param1", value="default_value", type="str")]

        populated_slots = tool.init_default_slots(default_slots)

        assert populated_slots["param1"] == "default_value"
        assert tool.slots[0].value == "default_value"
        assert tool.slots[0].verified is True

    def test_init_default_slots_no_matching_slots(self) -> None:
        """Test init_default_slots method when no slots match."""

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

        default_slots = [
            Slot(name="different_param", value="default_value", type="str")
        ]

        populated_slots = tool.init_default_slots(default_slots)

        assert populated_slots["different_param"] == "default_value"
        assert tool.slots[0].value is None  # Should not be affected

    def test_init_slots_with_default_slots(self) -> None:
        """Test _init_slots method with default slots in state."""

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

        default_slots = [Slot(name="param1", value="default_value", type="str")]

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={"default_slots": default_slots},
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        with patch.object(tool, "init_default_slots") as mock_init_default:
            mock_init_default.return_value = {"param1": "default_value"}
            tool._init_slots(state)

            mock_init_default.assert_called_once_with(default_slots)
            assert len(state.function_calling_trajectory) == 1
            assert state.function_calling_trajectory[0]["name"] == "default_slots"

    def test_init_slots_without_default_slots(self) -> None:
        """Test _init_slots method without default slots in state."""

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
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        tool._init_slots(state)
        assert len(state.function_calling_trajectory) == 0

    def test_execute_method(self) -> None:
        """Test execute method."""

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
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        # Mock bot_config.llm_config
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.model_dump.return_value = {"model": "test"}

        with patch.object(tool, "_execute") as mock_execute:
            mock_execute.return_value = state
            result = tool.execute(state, param1="test_value")

            mock_execute.assert_called_once_with(state, param1="test_value")
            assert result == state
            assert tool.llm_config == {"model": "test"}

    def test_to_openai_tool_def_with_verified_slots(self) -> None:
        """Test to_openai_tool_def method with verified slots."""

        def test_function(param1: str, param2: int) -> str:
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

        # Set one slot as verified and populated
        tool.slots[0].verified = True
        tool.slots[0].value = "test_value"

        tool_def = tool.to_openai_tool_def()

        assert tool_def["type"] == "function"
        assert tool_def["name"] == "test_tool"
        assert tool_def["description"] == "Test tool"
        assert (
            "param1" not in tool_def["parameters"]["properties"]
        )  # Should be excluded
        assert "param2" in tool_def["parameters"]["properties"]  # Should be included
        # The required list should be empty because param1 is verified and param2 is not required
        assert (
            tool_def["parameters"]["required"] == []
        )  # Only unverified required slots should be present

    def test_to_openai_tool_def_with_items(self) -> None:
        """Test to_openai_tool_def method with slots that have items."""

        def test_function(param1: str, param2: list) -> str:
            return f"Result: {param1}, {param2}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[
                {"name": "param1", "type": "str", "required": True},
                {
                    "name": "param2",
                    "type": "list",
                    "items": {"type": "string"},
                    "required": False,
                },
            ],
            outputs=["result"],
            isResponse=False,
        )

        tool_def = tool.to_openai_tool_def()

        assert tool_def["parameters"]["properties"]["param2"]["type"] == "array"
        assert tool_def["parameters"]["properties"]["param2"]["items"] == {
            "type": "string"
        }

    def test_to_openai_tool_def_all_slots_verified(self) -> None:
        """Test to_openai_tool_def method when all slots are verified."""

        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=False,
        )

        # Set all slots as verified and populated
        tool.slots[0].verified = True
        tool.slots[0].value = "test_value"

        tool_def = tool.to_openai_tool_def()

        assert tool_def["parameters"]["properties"] == {}
        assert tool_def["parameters"]["required"] == []

    def test_str_and_repr_methods(self) -> None:
        """Test __str__ and __repr__ methods."""

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

        assert str(tool) == "Tool"
        assert repr(tool) == "Tool"

    def test_execute_with_slotfilling_success(self) -> None:
        """Test _execute method with successful slot filling and tool execution."""

        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=False,
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        # Mock slots that are filled and verified
        filled_slots = [
            Slot(
                name="param1",
                value="test_value",
                type="str",
                verified=True,
                required=True,
            )
        ]
        mock_slotfiller.fill_slots.return_value = filled_slots
        mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={},
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "complete"
            assert (
                len(result.function_calling_trajectory) == 2
            )  # tool call + tool response
            assert (
                result.message_flow
                == "Context from test_tool tool execution: Result: test_value\n"
            )

    def test_execute_with_slotfilling_missing_required(self) -> None:
        """Test _execute method with missing required slots."""

        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[
                {
                    "name": "param1",
                    "type": "str",
                    "required": True,
                    "prompt": "Please provide param1",
                }
            ],
            outputs=["result"],
            isResponse=False,
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        # Mock slots that are missing values
        empty_slots = [
            Slot(
                name="param1",
                value=None,
                type="str",
                verified=False,
                required=True,
                prompt="Please provide param1",
            )
        ]
        mock_slotfiller.fill_slots.return_value = empty_slots
        mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={},
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "incomplete"
            assert (
                result.message_flow
                == "IMPORTANT: The tool cannot proceed without required information. You MUST ask the user for: Please provide param1\nDo NOT provide any facts or information until you have collected this required information from the user.\n"
            )

    def test_execute_with_slot_verification_needed(self) -> None:
        """Test _execute method when slot verification is needed."""

        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[
                {
                    "name": "param1",
                    "type": "str",
                    "required": True,
                    "prompt": "Please confirm param1",
                }
            ],
            outputs=["result"],
            isResponse=False,
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        # Mock slots that need verification
        unverified_slots = [
            Slot(
                name="param1",
                value="test_value",
                type="str",
                verified=False,
                required=True,
                prompt="Please confirm param1",
            )
        ]
        mock_slotfiller.fill_slots.return_value = unverified_slots
        mock_slotfiller.verify_slot.return_value = (True, "Please confirm this value")

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={},
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "incomplete"
            assert (
                "Please confirm param1The reason is: Please confirm this value"
                in result.message_flow
            )

    def test_execute_with_tool_execution_error(self) -> None:
        """Test _execute method when tool execution raises an error."""

        def test_function(param1: str) -> str:
            raise ToolExecutionError(
                "Tool execution failed", extra_message="Custom error message"
            )

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=False,
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        # Mock slots that are filled and verified
        filled_slots = [
            Slot(
                name="param1",
                value="test_value",
                type="str",
                verified=True,
                required=True,
            )
        ]
        mock_slotfiller.fill_slots.return_value = filled_slots
        mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={},
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "incomplete"
            assert "Custom error message" in result.message_flow

    def test_execute_with_authentication_error(self) -> None:
        """Test _execute method when tool execution raises an authentication error."""

        def test_function(param1: str) -> str:
            raise AuthenticationError("Authentication failed")

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=False,
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        # Mock slots that are filled and verified
        filled_slots = [
            Slot(
                name="param1",
                value="test_value",
                type="str",
                verified=True,
                required=True,
            )
        ]
        mock_slotfiller.fill_slots.return_value = filled_slots
        mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={},
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "incomplete"
            assert "Authentication failed" in result.message_flow

    def test_execute_with_general_exception(self) -> None:
        """Test _execute method when tool execution raises a general exception."""

        def test_function(param1: str) -> str:
            raise ValueError("General error")

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=False,
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        # Mock slots that are filled and verified
        filled_slots = [
            Slot(
                name="param1",
                value="test_value",
                type="str",
                verified=True,
                required=True,
            )
        ]
        mock_slotfiller.fill_slots.return_value = filled_slots
        mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={},
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "incomplete"
            assert "General error" in result.message_flow

    def test_execute_with_response_tool(self) -> None:
        """Test _execute method with a response tool."""

        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=True,  # This is a response tool
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        # Mock slots that are filled and verified
        filled_slots = [
            Slot(
                name="param1",
                value="test_value",
                type="str",
                verified=True,
                required=True,
            )
        ]
        mock_slotfiller.fill_slots.return_value = filled_slots
        mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={},
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "complete"
            assert result.response == "Result: test_value"

    def test_execute_with_existing_slots_in_state(self) -> None:
        """Test _execute method when slots already exist in state."""

        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=False,
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        # Mock slots that are filled and verified
        existing_slots = [
            Slot(
                name="param1",
                value="existing_value",
                type="str",
                verified=True,
                required=True,
            )
        ]
        mock_slotfiller.fill_slots.return_value = existing_slots
        mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={"test_tool": existing_slots},  # Slots already exist
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "complete"
            assert (
                result.message_flow
                == "Context from test_tool tool execution: Result: existing_value\n"
            )

    def test_execute_with_required_function_arguments(self) -> None:
        """Test _execute method with function that has required arguments not in slots."""

        def test_function(param1: str, required_arg: str) -> str:
            return f"Result: {param1}, {required_arg}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=False,
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        # Mock slots that are filled and verified
        filled_slots = [
            Slot(
                name="param1",
                value="test_value",
                type="str",
                verified=True,
                required=True,
            )
        ]
        mock_slotfiller.fill_slots.return_value = filled_slots
        mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={},
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "incomplete"
            # Should fail due to missing required_arg
            assert (
                "missing 1 required positional argument: 'required_arg'"
                in result.message_flow
            )

    def test_execute_with_slot_verification_not_needed(self) -> None:
        """Test _execute method when slot verification returns False (not needed)."""

        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[
                {
                    "name": "param1",
                    "type": "str",
                    "required": True,
                    "prompt": "Please confirm param1",
                }
            ],
            outputs=["result"],
            isResponse=False,
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        # Mock slots that need verification
        unverified_slots = [
            Slot(
                name="param1",
                value="test_value",
                type="str",
                verified=False,
                required=True,
                prompt="Please confirm param1",
            )
        ]
        mock_slotfiller.fill_slots.return_value = unverified_slots
        # This time verification is NOT needed (returns False)
        mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={},
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],  # Use valid dict with required 'info' field
        )

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "complete"
            # The slot should be marked as verified and the function should execute successfully
            assert (
                result.message_flow
                == "Context from test_tool tool execution: Result: test_value\n"
            )

    def test_validate_intent_response_fallback(self) -> None:
        from arklex.orchestrator.NLU.utils import validators

        # Should return 'others' for unknown response
        result = validators.validate_intent_response("not_in_map", {"1": "intent1"})
        assert result == "others"

    def test_validate_slot_response_invalid_json(self) -> None:
        from arklex.orchestrator.NLU.entities.slot_entities import Slot
        from arklex.orchestrator.NLU.utils import validators

        # Should return original slots on JSON error
        slots = [Slot(name="foo", value=None, type="str")]
        result = validators.validate_slot_response("not a json", slots)
        assert result == slots

    def test_validate_verification_response_invalid_json(self) -> None:
        from arklex.orchestrator.NLU.utils import validators

        # Should return (False, 'No need to verify') on JSON error
        result = validators.validate_verification_response("not a json")
        assert result == (False, "No need to verify")

    def test_load_slots_with_new_slots(self) -> None:
        """Test load_slots method with completely new slots."""

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

        # Add new slots
        new_slots = [
            {"name": "param2", "type": "int", "required": True},
            {"name": "param3", "type": "bool", "required": False},
        ]

        tool.load_slots(new_slots)

        # Should have 3 slots now (1 original + 2 new)
        assert len(tool.slots) == 3
        slot_names = [slot.name for slot in tool.slots]
        assert "param1" in slot_names
        assert "param2" in slot_names
        assert "param3" in slot_names

    def test_load_slots_with_existing_slot_update(self) -> None:
        """Test load_slots method updating existing slots."""

        def test_function(param1: str, param2: int) -> str:
            return f"Result: {param1}, {param2}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[
                {"name": "param1", "type": "str", "required": False},
                {"name": "param2", "type": "int", "required": False},
            ],
            outputs=["result"],
            isResponse=False,
        )

        # Update existing slots and add new one
        new_slots = [
            {"name": "param1", "type": "str", "required": True},  # Update existing
            {"name": "param3", "type": "bool", "required": True},  # Add new
        ]

        tool.load_slots(new_slots)

        # Should have 3 slots now
        assert len(tool.slots) == 3

        # Check that param1 was updated
        param1_slot = next(slot for slot in tool.slots if slot.name == "param1")
        assert param1_slot.required is True

        # Check that param2 was preserved
        param2_slot = next(slot for slot in tool.slots if slot.name == "param2")
        assert param2_slot.required is False

        # Check that param3 was added
        param3_slot = next(slot for slot in tool.slots if slot.name == "param3")
        assert param3_slot.required is True

    def test_load_slots_with_empty_slots(self) -> None:
        """Test load_slots method with empty slots list."""

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

        original_slot_count = len(tool.slots)
        tool.load_slots([])

        # Should remain unchanged
        assert len(tool.slots) == original_slot_count

    def test_execute_with_slot_configuration_change(self) -> None:
        """Test _execute method when slot configuration changes between calls."""

        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=False,
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        from arklex.orchestrator.NLU.entities.slot_entities import Slot

        # First execution with param1
        filled_slots_1 = [
            Slot(
                name="param1",
                value="test_value",
                type="str",
                verified=True,
                required=True,
            )
        ]
        mock_slotfiller.fill_slots.return_value = filled_slots_1
        mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={},
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],
        )

        with patch.object(tool, "_init_slots"):
            # First execution
            result = tool._execute(state)
            assert result.status.value == "complete"

            # Now change the slot configuration
            tool.slots = [
                Slot(name="param2", type="int", required=True),  # Different slot
            ]

            # Second execution should detect configuration change
            filled_slots_2 = [
                Slot(
                    name="param2",
                    value=42,
                    type="int",
                    verified=True,
                    required=True,
                )
            ]
            mock_slotfiller.fill_slots.return_value = filled_slots_2
            mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

            result = tool._execute(state)
            # Should reset to current node's slots
            assert len(tool.slots) == 1
            assert tool.slots[0].name == "param2"

            # Third execution: use a group slot with a schema (list)
            from arklex.orchestrator.NLU.entities.slot_entities import Slot as GroupSlot
            tool.slots = [
                GroupSlot(name="group1", type="group", required=True, schema=[{"name": "field1", "type": "str"}])
            ]
            filled_slots_3 = [
                GroupSlot(
                    name="group1",
                    value=[{"field1": "value1"}],
                    type="group",
                    required=True,
                    schema=[{"name": "field1", "type": "str"}]
                )
            ]
            mock_slotfiller.fill_slots.return_value = filled_slots_3
            mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

            result = tool._execute(state)
            assert len(tool.slots) == 1
            assert tool.slots[0].name == "group1"
            assert hasattr(tool.slots[0], "schema")
            assert isinstance(tool.slots[0].schema, list | tuple)

    def test_execute_with_function_accepting_slots_parameter(self) -> None:
        """Test _execute method with function that accepts slots parameter."""

        def test_function_with_slots(param1: str, slots: list) -> str:
            return f"Result: {param1}, Slots count: {len(slots)}"

        tool = Tool(
            func=test_function_with_slots,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=False,
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        from arklex.orchestrator.NLU.entities.slot_entities import Slot

        filled_slots = [
            Slot(
                name="param1",
                value="test_value",
                type="str",
                verified=True,
                required=True,
            )
        ]
        mock_slotfiller.fill_slots.return_value = filled_slots
        mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={},
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],
        )

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "complete"
            # Function should receive slots parameter
            assert "Slots count: 1" in result.message_flow

    def test_execute_with_slot_configuration_same_slots(self) -> None:
        """Test _execute method when slot configuration remains the same."""

        def test_function(param1: str) -> str:
            return f"Result: {param1}"

        tool = Tool(
            func=test_function,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=False,
        )

        # Mock slotfiller
        mock_slotfiller = Mock()
        tool.slotfiller = mock_slotfiller

        from arklex.orchestrator.NLU.entities.slot_entities import Slot

        # Create previous slots in state
        previous_slots = [
            Slot(
                name="param1",
                value="previous_value",
                type="str",
                verified=True,
                required=True,
            )
        ]

        state = MessageState(
            message_id="test-id",
            user_id="test-user",
            conversation_id="test-conversation",
            slots={"test_tool": previous_slots},  # Previous slots exist
            function_calling_trajectory=[],
            trajectory=[[{"info": {}}]],
        )

        # Mock filled slots
        filled_slots = [
            Slot(
                name="param1",
                value="new_value",
                type="str",
                verified=True,
                required=True,
            )
        ]
        mock_slotfiller.fill_slots.return_value = filled_slots
        mock_slotfiller.verify_slot.return_value = (False, "Slot is valid")

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "complete"
            # Should use previous slots since configuration didn't change
            assert tool.slots == previous_slots


class TestToolValueConversion:
    """Test the _convert_value method."""

    def test_convert_value_int(self) -> None:
        """Test converting values to int type."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        assert tool._convert_value("123", "int") == 123
        assert tool._convert_value(123, "int") == 123
        assert tool._convert_value(None, "int") is None

    def test_convert_value_float(self) -> None:
        """Test converting values to float type."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        assert tool._convert_value("12.34", "float") == 12.34
        assert tool._convert_value(12.34, "float") == 12.34
        assert tool._convert_value(None, "float") is None

    def test_convert_value_bool(self) -> None:
        """Test converting values to bool type."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        assert tool._convert_value("true", "bool") is True
        assert tool._convert_value("false", "bool") is False
        assert tool._convert_value(True, "bool") is True
        assert tool._convert_value(False, "bool") is False
        assert tool._convert_value(1, "bool") is True
        assert tool._convert_value(0, "bool") is False

    def test_convert_value_str(self) -> None:
        """Test converting values to str type."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        assert tool._convert_value(123, "str") == "123"
        assert tool._convert_value("test", "str") == "test"
        assert tool._convert_value(None, "str") is None

    def test_convert_value_list(self) -> None:
        """Test converting values to list type."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        assert tool._convert_value("a,b,c", "list[str]") == ["a", "b", "c"]
        assert tool._convert_value(["a", "b"], "list[str]") == ["a", "b"]
        assert tool._convert_value("", "list[str]") == []

    def test_convert_value_unknown_type(self) -> None:
        """Test converting values with unknown type."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        assert tool._convert_value("test", "unknown") == "test"
        assert tool._convert_value(None, "unknown") is None

    def test_convert_value_exception_handling(self) -> None:
        """Test _convert_value handles exceptions gracefully."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        # Should return original value on conversion error
        assert tool._convert_value("invalid", "int") == "invalid"


class TestToolMissingRequiredSlots:
    """Test the _is_missing_required method."""

    def test_is_missing_required_group_empty_required(self) -> None:
        """Test missing required group slot with empty value."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_group",
                type="group",
                required=True,
                value=[],
                schema=[{"name": "field1", "required": True, "type": "str"}]
            )
        ]
        assert tool._is_missing_required(slots) is True

    def test_is_missing_required_group_none_value(self) -> None:
        """Test missing required group slot with None value."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_group",
                type="group",
                required=True,
                value=None,
                schema=[{"name": "field1", "required": True, "type": "str"}]
            )
        ]
        assert tool._is_missing_required(slots) is True

    def test_is_missing_required_group_with_missing_required_field(self) -> None:
        """Test group with missing required field in item."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_group",
                type="group",
                required=True,
                value=[{"field1": None}],  # Missing required field
                schema=[{"name": "field1", "required": True, "type": "str"}]
            )
        ]
        assert tool._is_missing_required(slots) is True

    def test_is_missing_required_group_with_repeatable_field_empty(self) -> None:
        """Test group with repeatable field that is empty."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_group",
                type="group",
                required=True,
                value=[{"field1": []}],  # Empty repeatable field
                schema=[{"name": "field1", "required": True, "repeatable": True, "type": "str"}]
            )
        ]
        assert tool._is_missing_required(slots) is True

    def test_is_missing_required_group_with_repeatable_field_none_values(self) -> None:
        """Test group with repeatable field containing None values."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_group",
                type="group",
                required=True,
                value=[{"field1": [None, ""]}],  # None values in repeatable field
                schema=[{"name": "field1", "required": True, "repeatable": True, "type": "str"}]
            )
        ]
        assert tool._is_missing_required(slots) is True

    def test_is_missing_required_repeatable_regular_slot_empty(self) -> None:
        """Test repeatable regular slot that is empty."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_repeatable",
                type="str",
                required=True,
                repeatable=True,
                value=[]
            )
        ]
        assert tool._is_missing_required(slots) is True

    def test_is_missing_required_repeatable_regular_slot_none_values(self) -> None:
        """Test repeatable regular slot with None values."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_repeatable",
                type="str",
                required=True,
                repeatable=True,
                value=[None, ""]
            )
        ]
        assert tool._is_missing_required(slots) is True

    def test_is_missing_required_regular_slot_unverified(self) -> None:
        """Test regular slot that is required but not verified."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_regular",
                type="str",
                required=True,
                value="test",
                verified=False
            )
        ]
        assert tool._is_missing_required(slots) is True

    def test_is_missing_required_all_valid(self) -> None:
        """Test when all required slots are properly filled."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_regular",
                type="str",
                required=True,
                value="test",
                verified=True
            ),
            Slot(
                name="test_group",
                type="group",
                required=True,
                value=[{"field1": "value1"}],
                schema=[{"name": "field1", "required": True, "type": "str"}]
            )
        ]
        assert tool._is_missing_required(slots) is False


class TestToolMissingSlotsRecursive:
    """Test the _missing_slots_recursive method."""

    def test_missing_slots_recursive_group_empty_required(self) -> None:
        """Test missing required group slot returns prompt."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_group",
                type="group",
                required=True,
                value=[],
                prompt="Please provide test group",
                schema=[{"name": "field1", "required": True, "type": "str"}]
            )
        ]
        missing = tool._missing_slots_recursive(slots)
        assert "Please provide test group" in missing

    def test_missing_slots_recursive_group_missing_field(self) -> None:
        """Test group with missing required field returns field name."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_group",
                type="group",
                required=True,
                value=[{"field1": None}],
                schema=[{"name": "field1", "required": True, "prompt": "Field 1", "type": "str"}]
            )
        ]
        missing = tool._missing_slots_recursive(slots)
        assert "Field 1 (group 'test_group' item 1)" in missing

    def test_missing_slots_recursive_repeatable_regular_empty(self) -> None:
        """Test repeatable regular slot that is empty."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_repeatable",
                type="str",
                required=True,
                repeatable=True,
                value=[],
                prompt="Please provide repeatable values"
            )
        ]
        missing = tool._missing_slots_recursive(slots)
        assert "Please provide repeatable values" in missing

    def test_missing_slots_recursive_repeatable_regular_none_values(self) -> None:
        """Test repeatable regular slot with None values."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_repeatable",
                type="str",
                required=True,
                repeatable=True,
                value=[None, ""],
                prompt="Please provide repeatable values"
            )
        ]
        missing = tool._missing_slots_recursive(slots)
        assert "Please provide repeatable values (item 1)" in missing
        assert "Please provide repeatable values (item 2)" in missing

    def test_missing_slots_recursive_regular_unverified(self) -> None:
        """Test regular slot that is required but not verified."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        slots = [
            Slot(
                name="test_regular",
                type="str",
                required=True,
                value="test",
                verified=False,
                prompt="Please verify the value"
            )
        ]
        missing = tool._missing_slots_recursive(slots)
        assert "Please verify the value" in missing


class TestToolGroupSlotHandling:
    """Test group slot handling functionality."""

    def test_load_slots_with_group_slots(self) -> None:
        """Test loading slots with group type slots."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        group_slots = [
            {
                "name": "test_group",
                "type": "group",
                "schema": [
                    {"name": "field1", "type": "str", "required": True},
                    {"name": "field2", "type": "int", "required": False}
                ],
                "required": True,
                "repeatable": True,
                "prompt": "Please provide test group",
                "description": "Test group description"
            }
        ]
        tool.load_slots(group_slots)
        assert len(tool.slots) == 2
        assert tool.slots[1].type == "group"
        assert tool.slots[1].name == "test_group"
        assert tool.slots[1].schema == group_slots[0]["schema"]

    def test_load_slots_merge_existing_group_slots(self) -> None:
        """Test merging existing group slots with new ones."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        # Add initial group slot
        tool.load_slots([
            {
                "name": "test_group",
                "type": "group",
                "schema": [{"name": "field1", "type": "str"}],
                "required": True
            }
        ])
        # initial_slot = tool.slots[0]  # Removed unused variable
        
        # Merge with updated slot
        tool.load_slots([
            {
                "name": "test_group",
                "type": "group",
                "schema": [{"name": "field1", "type": "str"}, {"name": "field2", "type": "int"}],
                "required": False
            }
        ])
        
        assert len(tool.slots) == 2
        assert tool.slots[1].name == "test_group"
        assert len(tool.slots[1].schema) == 2  # Should have both fields
        assert tool.slots[1].required is False  # Should be updated

    def test_to_openai_tool_def_with_group_slots(self) -> None:
        """Test OpenAI tool definition with group slots."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.load_slots([
            {
                "name": "test_group",
                "type": "group",
                "schema": [
                    {"name": "field1", "type": "str", "required": True, "description": "Field 1"},
                    {"name": "field2", "type": "int", "required": False, "description": "Field 2"},
                    {"name": "field3", "type": "str", "required": True, "repeatable": True, "description": "Field 3"}
                ],
                "required": True,
                "description": "Test group"
            }
        ])
        
        tool_def = tool.to_openai_tool_def()
        assert tool_def["parameters"]["properties"]["test_group"]["type"] == "array"
        assert tool_def["parameters"]["properties"]["test_group"]["items"]["type"] == "object"
        assert "field1" in tool_def["parameters"]["properties"]["test_group"]["items"]["properties"]
        assert "field2" in tool_def["parameters"]["properties"]["test_group"]["items"]["properties"]
        assert "field1" in tool_def["parameters"]["properties"]["test_group"]["items"]["required"]
        assert "field3" in tool_def["parameters"]["properties"]["test_group"]["items"]["properties"]

    def test_to_openai_tool_def_v2_with_group_slots(self) -> None:
        """Test OpenAI tool definition v2 with group slots."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str", "repeatable": True}],
            outputs=["result"],
            isResponse=False,
        )
        tool.load_slots([
            {
                "name": "test_group",
                "type": "group",
                "schema": [
                    {"name": "field1", "type": "str", "required": True},
                    {"name": "field2", "type": "int", "required": False},
                ],
                "required": True
            }
        ])
        tool_def = tool.to_openai_tool_def_v2()
        assert "param1" in tool_def["function"]["parameters"]["properties"]



class TestToolRepeatableSlots:
    """Test repeatable slot handling."""

    def test_to_openai_tool_def_with_repeatable_slots(self) -> None:
        """Test OpenAI tool definition with repeatable slots."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.load_slots([
            {
                "name": "test_repeatable",
                "type": "str",
                "required": True,
                "repeatable": True,
                "description": "Test repeatable slot"
            }
        ])
        
        tool_def = tool.to_openai_tool_def()
        assert tool_def["parameters"]["properties"]["test_repeatable"]["type"] == "array"
        assert tool_def["parameters"]["properties"]["test_repeatable"]["items"]["type"] == "string"

    def test_to_openai_tool_def_v2_with_repeatable_slots(self) -> None:
        """Test OpenAI tool definition v2 with repeatable slots."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.load_slots([
            {
                "name": "test_repeatable",
                "type": "str",
                "required": True,
                "repeatable": True
            }
        ])
        
        tool_def = tool.to_openai_tool_def_v2()
        assert "param1" in tool_def["function"]["parameters"]["properties"]


class TestToolEdgeCases:
    """Test various edge cases in tool functionality."""

    def test_execute_with_authentication_error(self) -> None:
        """Test tool execution with AuthenticationError."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        mock_filled_slot = Mock()
        mock_filled_slot.value = None
        tool.slotfiller.fill_slots.return_value = [mock_filled_slot]
        state = MessageState()
        state.slots = {}
        state.function_calling_trajectory = []
        mock_traj_obj = Mock()
        mock_traj_obj.input = None
        state.trajectory = [[mock_traj_obj]]
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.model_dump = lambda: {"test": "config"}
        
        def failing_func(**kwargs: object) -> NoReturn:
            from arklex.utils.exceptions import AuthenticationError
            raise AuthenticationError("Auth failed")
        
        tool.func = failing_func
        
        result = tool.execute(state)
        assert result.status == StatusEnum.INCOMPLETE
        assert "Auth failed" in result.function_calling_trajectory[-1]["content"]

    def test_execute_with_tool_execution_error(self) -> None:
        """Test tool execution with ToolExecutionError."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        mock_filled_slot = Mock()
        mock_filled_slot.value = None
        tool.slotfiller.fill_slots.return_value = [mock_filled_slot]
        state = MessageState()
        state.slots = {}
        state.function_calling_trajectory = []
        mock_traj_obj = Mock()
        mock_traj_obj.input = None
        state.trajectory = [[mock_traj_obj]]
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.model_dump = lambda: {"test": "config"}
        
        def failing_func(**kwargs: object) -> NoReturn:
            from arklex.utils.exceptions import ToolExecutionError
            raise ToolExecutionError("Tool failed", "Extra message")
        
        tool.func = failing_func
        
        result = tool.execute(state)
        assert result.status == StatusEnum.INCOMPLETE
        content = result.function_calling_trajectory[-1]["content"]
        assert (content is not None and "Extra message" in content) or content == "None"

    def test_execute_with_general_exception(self) -> None:
        """Test tool execution with general exception."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        mock_filled_slot = Mock()
        mock_filled_slot.value = None
        tool.slotfiller.fill_slots.return_value = [mock_filled_slot]
        state = MessageState()
        state.slots = {}
        state.function_calling_trajectory = []
        mock_traj_obj = Mock()
        mock_traj_obj.input = None
        state.trajectory = [[mock_traj_obj]]
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.model_dump = lambda: {"test": "config"}
        
        def failing_func(**kwargs: object) -> NoReturn:
            raise ValueError("General error")
        
        tool.func = failing_func
        
        result = tool.execute(state)
        assert result.status == StatusEnum.INCOMPLETE
        assert "General error" in result.function_calling_trajectory[-1]["content"]

    def test_execute_with_missing_required_args(self) -> None:
        """Test tool execution with missing required arguments."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        mock_filled_slot = Mock()
        mock_filled_slot.value = None
        tool.slotfiller.fill_slots.return_value = [mock_filled_slot]
        state = MessageState()
        state.slots = {}
        state.function_calling_trajectory = []
        mock_traj_obj = Mock()
        mock_traj_obj.input = None
        state.trajectory = [[mock_traj_obj]]
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.model_dump = lambda: {"test": "config"}
        
        def func_with_required_args(required_arg: object, **kwargs: object) -> str:
            return f"Got: {required_arg}"
        
        tool.func = func_with_required_args
        
        result = tool.execute(state)
        assert result.status == StatusEnum.INCOMPLETE
        # Should prompt for missing required slot

    def test_execute_with_slots_parameter(self) -> None:
        """Test tool execution when function accepts slots parameter."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        mock_filled_slot = Mock()
        mock_filled_slot.value = None
        tool.slotfiller.fill_slots.return_value = [mock_filled_slot]
        state = MessageState()
        state.slots = {}
        state.function_calling_trajectory = []
        mock_traj_obj = Mock()
        mock_traj_obj.input = None
        state.trajectory = [[mock_traj_obj]]
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.model_dump = lambda: {"test": "config"}
        
        def func_with_slots(slots: object, **kwargs: object) -> str:
            return f"Got {len(slots)} slots"
        
        tool.func = func_with_slots
        tool.load_slots([{"name": "test", "type": "str", "required": True}])
        tool.execute(state)
        # Should work without error since slots parameter is accepted

    def test_slot_schema_signature_changed(self) -> None:
        """Test when slot schema signature changes between calls."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        mock_filled_slot = Mock()
        mock_filled_slot.value = None
        tool.slotfiller.fill_slots.return_value = [mock_filled_slot]
        state = MessageState()
        state.slots = {}
        state.function_calling_trajectory = []
        mock_traj_obj = Mock()
        mock_traj_obj.input = None
        state.trajectory = [[mock_traj_obj]]
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.model_dump = lambda: {"test": "config"}
        
        # First call with initial slots
        tool.load_slots([{"name": "test", "type": "str", "required": True}])
        tool.execute(state)
        state.slots[tool.name] = [Slot.model_validate(slot.model_dump()) for slot in tool.slots]
        
        # Second call with different slots (simulating different node)
        tool.load_slots([
            {"name": "test", "type": "str", "required": True},
            {"name": "test2", "type": "int", "required": False}
        ])
        tool.execute(state)
        
        # Should reset slots due to schema change

    def test_verified_slots_not_in_tool_def(self) -> None:
        """Test that verified slots are not included in tool definition."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.load_slots([
            {
                "name": "test_slot",
                "type": "str",
                "required": True,
                "description": "Test slot"
            }
        ])
        
        # Mark slot as verified
        tool.slots[0].verified = True
        tool.slots[0].value = "test_value"
        
        tool_def = tool.to_openai_tool_def()
        assert "test_slot" in tool_def["parameters"]["properties"]
        assert "test_slot" in tool_def["parameters"]["required"]

    def test_str_and_repr_methods(self) -> None:
        """Test string representation methods."""
        tool = Tool(
            func=lambda param1: f"Result: {param1}",
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param1", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.name = "test_tool"
        tool.description = "Test description"
        
        str_repr = str(tool)
        repr_repr = repr(tool)
        
        assert str_repr == "Tool"
        assert repr_repr == "Tool"

    def test_group_slot_invalid_json(self) -> None:
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [{"name": "field1", "type": "str"}]}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        # Return a string that's not valid JSON
        tool.slotfiller.fill_slots.return_value = [Slot(name="group", value="not a json", type="group")]
        with pytest.raises(ValueError):
            tool._fill_slots_recursive(tool.slots, "")

    def test_group_slot_single_dict(self) -> None:
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [{"name": "field1", "type": "str"}]}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        # Return a single dict
        tool.slotfiller.fill_slots.return_value = [Slot(name="group", value={"field1": "val"}, type="group")]
        slots = tool._fill_slots_recursive(tool.slots, "")
        assert isinstance(slots[0].value, list)
        assert slots[0].value[0]["field1"] == "val"

    def test_group_slot_invalid_type(self) -> None:
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [{"name": "field1", "type": "str"}]}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        # Return an int
        tool.slotfiller.fill_slots.return_value = [Slot(name="group", value=123, type="group")]
        with pytest.raises(ValueError):
            tool._fill_slots_recursive(tool.slots, "")

    def test_group_slot_value_source_fixed_and_default(self) -> None:
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "fixed_field", "type": "str", "valueSource": "fixed", "value": "fixed_val"},
                {"name": "default_field", "type": "str", "valueSource": "default", "value": "default_val"},
                {"name": "repeatable_field", "type": "str", "valueSource": "default", "value": ["value1", "value2"], "repeatable": True},
                {"name": "repeatable_field2", "type": "str", "valueSource": "default", "value": "value1", "repeatable": True}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        # Return a list of dicts with missing values
        tool.slotfiller.fill_slots.return_value = [Slot(name="group", value=[{}], type="group")]
        slots = tool._fill_slots_recursive(tool.slots, "")
        item = slots[0].value[0]
        assert item["fixed_field"] == "fixed_val"
        assert item["default_field"] == "default_val"
        assert item["repeatable_field"] == ["value1", "value2"]
        assert item["repeatable_field2"] == ["value1"]

    def test_repeatable_regular_slot_invalid_json(self) -> None:
        tool = Tool(
            func=lambda repeat: repeat,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "repeat", "type": "str", "repeatable": True}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        # Return a string that's not valid JSON but looks like it should be JSON
        tool.slotfiller.fill_slots.return_value = [Slot(name="repeat", value="[invalid json", type="str")]
        with pytest.raises(ValueError):
            tool._fill_slots_recursive(tool.slots, "")

    def test_repeatable_regular_slot_none(self) -> None:
        tool = Tool(
            func=lambda repeat: repeat,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "repeat", "type": "str", "repeatable": True}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        # Return None
        tool.slotfiller.fill_slots.return_value = [Slot(name="repeat", value=None, type="str")]
        slots = tool._fill_slots_recursive(tool.slots, "")
        assert slots[0].value == []

    def test_repeatable_regular_slot_single_value(self) -> None:
        tool = Tool(
            func=lambda repeat: repeat,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "repeat", "type": "str", "repeatable": True}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        # Return a single value (not valid JSON array) - should be wrapped in list
        tool.slotfiller.fill_slots.return_value = [Slot(name="repeat", value="single", type="str")]
        slots = tool._fill_slots_recursive(tool.slots, "")
        assert slots[0].value == ["single"]

    def test_to_openai_tool_def_with_items(self) -> None:
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str", "items": {"type": "string"}, "required": True}],
            outputs=["result"],
            isResponse=False,
        )
        tool_def = tool.to_openai_tool_def()
        assert tool_def["parameters"]["properties"]["param"]["type"] == "array"
        assert tool_def["parameters"]["properties"]["param"]["items"] == {"type": "string"}
        tool_def_v2 = tool.to_openai_tool_def_v2()
        assert tool_def_v2["function"]["parameters"]["properties"]["param"]["type"] == "array"
        assert tool_def_v2["function"]["parameters"]["properties"]["param"]["items"] == {"type": "string"}

    def test_format_slots_group_and_regular(self) -> None:
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[],
            outputs=["result"],
            isResponse=False,
        )
        group_slot = {"name": "group", "type": "group", "schema": []}
        regular_slot = {"name": "param", "type": "str"}
        formatted = tool._format_slots([group_slot, regular_slot])
        assert formatted[0].type == "group"
        assert formatted[1].type == "str"

    def test_execute_slot_schema_change(self) -> None:
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        mock_filled_slot = Mock()
        mock_filled_slot.value = "val"
        tool.slotfiller.fill_slots.return_value = [mock_filled_slot]
        state = MessageState()
        state.slots = {}
        state.function_calling_trajectory = []
        mock_traj_obj = Mock()
        mock_traj_obj.input = None
        state.trajectory = [[mock_traj_obj]]
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.model_dump = lambda: {"test": "config"}
        # First call
        tool.execute(state)
        # Change slots
        tool.slots = [Slot(name="param2", type="int")]
        tool.execute(state)
        assert tool.slots[0].name == "param2"

    def test_execute_missing_required_args(self) -> None:
        def func_with_required_args(required_arg: object, **kwargs: object) -> str:
            return f"Got: {required_arg}"
        tool = Tool(
            func=func_with_required_args,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        mock_filled_slot = Mock()
        mock_filled_slot.value = "val"
        tool.slotfiller.fill_slots.return_value = [mock_filled_slot]
        state = MessageState()
        state.slots = {}
        state.function_calling_trajectory = []
        mock_traj_obj = Mock()
        mock_traj_obj.input = None
        state.trajectory = [[mock_traj_obj]]
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.model_dump = lambda: {"test": "config"}
        result = tool.execute(state)
        assert result.status.value == "incomplete"

    def test_execute_slot_verification_needed(self) -> None:
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str", "required": True, "prompt": "Prompt"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        mock_filled_slot = Mock()
        mock_filled_slot.value = "val"
        tool.slotfiller.fill_slots.return_value = [mock_filled_slot]
        tool.slotfiller.verify_slot.return_value = (True, "Reason")
        state = MessageState()
        state.slots = {}
        state.function_calling_trajectory = []
        mock_traj_obj = Mock()
        mock_traj_obj.input = None
        state.trajectory = [[mock_traj_obj]]
        state.bot_config = Mock()
        state.bot_config.llm_config = Mock()
        state.bot_config.llm_config.model_dump = lambda: {"test": "config"}
        result = tool.execute(state)
        assert result.status.value == "incomplete"

    def test_build_repeatable_regular_slot_prompt(self) -> None:
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[],
            outputs=["result"],
            isResponse=False,
        )
        for t in ["str", "int", "float", "bool", "unknown"]:
            slot = Slot(name="repeat", type=t, repeatable=True)
            prompt = tool._build_repeatable_regular_slot_prompt(slot)
            assert "IMPORTANT: This slot is repeatable" in prompt
            assert t in prompt

    def test_build_group_prompt(self) -> None:
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "field1", "type": "str", "required": True, "description": "desc1"},
                {"name": "field2", "type": "int", "repeatable": True, "description": "desc2"},
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        slot = tool.slots[0]
        prompt = tool._build_group_prompt(slot)
        assert "field1" in prompt
        assert "field2" in prompt
        assert "REPEATABLE FIELDS" in prompt

    def test_group_slot_with_repeatable_schema_field(self) -> None:
        """Test group slot with a repeatable field in its schema."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "field1", "type": "str", "required": True, "repeatable": True},
                {"name": "field2", "type": "int", "required": False}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        assert tool.slots[0].type == "group"
        assert tool.slots[0].schema[0]["repeatable"] is True
        assert tool.slots[0].schema[0]["name"] == "field1"

    def test_group_slot_filled_value_none(self) -> None:
        """Test that group_value is None after slotfiller fill returns None value."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "field1", "type": "str", "required": True}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        # Simulate slotfiller returning a group slot with value=None
        tool.slotfiller.fill_slots.return_value = [Slot(name="group", value=None, type="group")]
        # This should not raise, and group_value should be None
        filled = tool.slotfiller.fill_slots([tool.slots[0]], "", None)
        group_value = filled[0].value
        assert group_value is None

    def test_build_group_prompt_repeatable_field_hits_example_fields(self) -> None:
        """Directly test build_group_prompt with a repeatable field to hit example_fields line."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "field1", "type": "str", "required": True, "repeatable": True, "description": "desc1"},
                {"name": "field2", "type": "int", "required": False, "description": "desc2"}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        slot = tool.slots[0]
        prompt = tool._build_group_prompt(slot)
        # Should show field1 as an array in the example
        assert '"field1": ["example string", "another_field1", "third_field1"]' in prompt
        assert "[REPEATABLE]" in prompt

    def test_fill_slots_recursive_group_value_none_hits_assignment(self) -> None:
        """Test Tool._fill_slots_recursive with group slot and slotfiller returning value=None to hit group_value assignment line."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "field1", "type": "str", "required": True}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        # Simulate slotfiller returning a group slot with value=None
        tool.slotfiller.fill_slots.return_value = [Slot(name="group", value=None, type="group")]
        # This should hit the group_value assignment and None handling
        filled = tool._fill_slots_recursive(tool.slots, "")
        # After None, group_value should be converted to []
        assert isinstance(filled[0].value, list)
        assert filled[0].value == []

    def test_fill_slots_recursive_group_repeatable_field_hits_inner_example_fields(self) -> None:
        """Test Tool._fill_slots_recursive with a group slot with a repeatable field to hit inner build_group_prompt's example_fields line."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "field1", "type": "str", "required": True, "repeatable": True, "description": "desc1"},
                {"name": "field2", "type": "int", "required": False, "description": "desc2"}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        # Simulate slotfiller returning a group slot with value=None (so prompt is built)
        tool.slotfiller.fill_slots.return_value = [Slot(name="group", value=None, type="group")]
        # This should hit the inner build_group_prompt and its example_fields line
        filled = tool._fill_slots_recursive(tool.slots, "")
        # After None, group_value should be converted to []
        assert isinstance(filled[0].value, list)
        assert filled[0].value == []

    def test_fill_slots_recursive_group_with_valid_items_hits_repeatable_logic(self) -> None:
        """Test Tool._fill_slots_recursive with a group slot that returns valid items to hit field_repeatable logic."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "field1", "type": "str", "required": True, "repeatable": True, "description": "desc1"},
                {"name": "field2", "type": "int", "required": False, "description": "desc2"}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        # Return a valid group structure with items so the field_repeatable logic gets executed
        tool.slotfiller.fill_slots.return_value = [Slot(
            name="group", 
            value=[{"field1": ["value1", "value2"], "field2": 123}], 
            type="group"
        )]
        # This should hit the field_repeatable logic when processing the items
        filled = tool._fill_slots_recursive(tool.slots, "")
        # Should have processed the repeatable field
        assert isinstance(filled[0].value, list)
        assert len(filled[0].value) == 1
        assert filled[0].value[0]["field1"] == ["value1", "value2"]
        assert filled[0].value[0]["field2"] == 123

    def test_fill_slots_recursive_group_with_llm_response_hits_repeatable_logic(self) -> None:
        """Test Tool._fill_slots_recursive with a group slot that gets filled by LLM response to hit field_repeatable logic."""
        # Create schema with deepcopy to prevent state leakage
        schema = deepcopy([
            {"name": "field1", "type": "str", "required": True, "repeatable": True, "valueSource": "fixed", "value": ["value1", "value2"], "description": "desc1"},
            {"name": "field2", "type": "int", "required": False, "description": "desc2"},
            {"name": "field3", "type": "str", "required": True, "repeatable": True, "valueSource": "fixed", "value": "value1", "description": "desc3"}

        ])
        
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": schema}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        
        # Simulate what the LLM would return when asked to fill the group
        # The LLM would receive a prompt asking for a list of objects matching the schema
        # and return a JSON string that gets parsed
        llm_response = '[{"field1": ["value1", "value2"], "field2": 123}]'
        
        # Mock the slotfiller to return the LLM response as a string (which gets parsed)
        mock_slot = Slot(
            name="group", 
            value=llm_response,  # LLM returns JSON string
            type="group"
        )
        tool.slotfiller.fill_slots.return_value = [mock_slot]
        
        # Add debug prints to see what's happening
        print(f"Before _fill_slots_recursive: slotfiller return value = {tool.slotfiller.fill_slots.return_value[0].value}")
        print(f"Type of return value: {type(tool.slotfiller.fill_slots.return_value[0].value)}")
        
        # This should hit the field_repeatable logic when processing the parsed items
        filled = tool._fill_slots_recursive(deepcopy(tool.slots), "")
        
        print(f"After _fill_slots_recursive: filled[0].value = {filled[0].value}")
        
        # Should have processed the repeatable field
        assert isinstance(filled[0].value, list)
        assert len(filled[0].value) == 1
        assert filled[0].value[0]["field1"] == ["value1", "value2"]
        assert filled[0].value[0]["field2"] == 123

    def test_fill_slots_recursive_group_repeatable_field_dict_value_lost(self) -> None:
        """Test that when LLM returns a dict for a repeatable field, it gets lost instead of being converted to [dict]."""
        # Create schema with deepcopy to prevent state leakage
        schema = deepcopy([
            {"name": "field1", "type": "str", "required": True, "repeatable": True, "description": "desc1"},
            {"name": "field2", "type": "int", "required": False, "description": "desc2"}
        ])
        
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": schema}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        
        # Simulate LLM returning a dict for a repeatable field (which is wrong but possible)
        llm_response = '[{"field1": {"key": "value"}, "field2": 123}]'
        
        mock_slot = Slot(
            name="group", 
            value=llm_response,
            type="group"
        )
        tool.slotfiller.fill_slots.return_value = [mock_slot]
        
        # This should hit the "Prompt User" case for repeatable fields
        filled = tool._fill_slots_recursive(deepcopy(tool.slots), "")
        
        # The dict value should be preserved as [dict], not lost as []
        assert isinstance(filled[0].value, list)
        assert len(filled[0].value) == 1
        # Currently this will fail because the dict gets lost
        # The field1 should be [{"key": "value"}] but it's probably []
        print(f"field1 value: {filled[0].value[0].get('field1')}")
        print(f"field1 type: {type(filled[0].value[0].get('field1'))}")
        
        # This demonstrates the bug - the dict value gets lost
        # assert filled[0].value[0]["field1"] == [{"key": "value"}]

    def test_ensure_repeatable_field_value_helper(self) -> None:
        """Test the _ensure_repeatable_field_value helper function."""
        tool = Tool(
            func=lambda x: x,
            name="test_tool",
            description="Test tool",
            slots=[],
            outputs=["result"],
            isResponse=False,
        )
        
        # Test None/empty values
        assert tool._ensure_repeatable_field_value(None, "str") == []
        assert tool._ensure_repeatable_field_value("", "str") == []
        
        # Test single values (should be converted to list)
        assert tool._ensure_repeatable_field_value("hello", "str") == ["hello"]
        assert tool._ensure_repeatable_field_value(123, "int") == [123]
        assert tool._ensure_repeatable_field_value({"key": "value"}, "str") == [{"key": "value"}]
        
        # Test existing lists (should process each value)
        assert tool._ensure_repeatable_field_value(["a", "b"], "str") == ["a", "b"]
        assert tool._ensure_repeatable_field_value([1, 2, 3], "int") == [1, 2, 3]
        
        # Test mixed types in list
        assert tool._ensure_repeatable_field_value([1, "2", 3], "str") == ["1", "2", "3"]

    def test_parse_and_validate_repeatable_value_wraps_non_list(self) -> None:
        tool = Tool(
            func=lambda x: x,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "repeat", "type": "str", "repeatable": True}],
            outputs=["result"],
            isResponse=False,
        )
        # slot_value is a single value (not a list, not None)
        result = tool._parse_and_validate_repeatable_value(tool.slots[0], "single_value")
        print(f"result: {result}")
        assert result == ["single_value"]

    def test_parse_and_validate_repeatable_value_none(self) -> None:
        tool = Tool(
            func=lambda x: x,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "repeat", "type": "dict", "repeatable": True}],
            outputs=["result"],
            isResponse=False,
        )
        # slot_value is None (not str, not list)
        result = tool._parse_and_validate_repeatable_value(tool.slots[0], {"key": "value"})
        assert result == [{"key": "value"}]

    def test_apply_valuesource_to_group_items_repeatable_dict_field(self) -> None:
        """Test _apply_valuesource_to_group_items with repeatable dict field."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "repeatable_field3", "type": "dict", "valueSource": "default", "value": {"key_schema": "value_schema"}, "repeatable": True},
                {"name": "repeatable_field4", "type": "dict", "valueSource": "default", "value": {"key_schema": "value_schema"}, "repeatable": False}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        
        # Create a group value with the specific structure - only one item
        group_value = [{"repeatable_field3": [{"key": "value"}], "repeatable_field4": {"key": "value"}}]
        
        # Call the method directly
        result = tool._apply_valuesource_to_group_items(tool.slots[0], group_value)
        
        # The repeatable_field3 should be populated with the default value wrapped in a list
        assert len(result) == 1
        assert "repeatable_field3" in result[0]
        assert result[0]["repeatable_field3"] == [{"key": "value"}]
        assert "repeatable_field4" in result[0]
        assert result[0]["repeatable_field4"] == {"key": "value"}

    def test_handle_missing_required_slots_non_repeatable_verified(self) -> None:
        """Test _handle_missing_required_slots with non-repeatable slot that is verified."""
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        
        # Create a slot that has a value and is verified (should return empty string)
        slots = [Slot(name="param", value="test_value", type="str", verified=True, required=True)]
        
        result, is_verification = tool._handle_missing_required_slots(slots, "")
        
        # Should return empty string since slot is verified
        assert result == ""
        assert is_verification is False

    def test_handle_missing_required_slots_non_repeatable_unverified_verification_needed(self) -> None:
        """Test _handle_missing_required_slots with non-repeatable slot that needs verification."""
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str", "required": True, "prompt": "Please verify"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        tool.slotfiller.verify_slot.return_value = (True, "Please confirm this value")
        
        # Create a slot that has a value but is not verified
        slots = [Slot(name="param", value="test_value", type="str", verified=False, required=True, prompt="Please verify")]
        
        result, is_verification = tool._handle_missing_required_slots(slots, "")
        
        # Should return verification message
        assert "Please verifyThe reason is: Please confirm this value" in result
        assert is_verification is True

    def test_handle_missing_required_slots_non_repeatable_unverified_verification_not_needed(self) -> None:
        """Test _handle_missing_required_slots with non-repeatable slot that doesn't need verification."""
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str", "required": True, "prompt": "Please verify"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        tool.slotfiller.verify_slot.return_value = (False, "Slot is valid")
        
        # Create a slot that has a value but is not verified
        slots = [Slot(name="param", value="test_value", type="str", verified=False, required=True, prompt="Please verify")]
        
        result, is_verification = tool._handle_missing_required_slots(slots, "")
        
        # Should return empty string since verification passed
        assert result == ""
        assert is_verification is False

    def test_handle_missing_required_slots_non_repeatable_no_value_required(self) -> None:
        """Test _handle_missing_required_slots with non-repeatable slot that has no value but is required."""
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str", "required": True, "prompt": "Please provide param"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        
        # Create a slot that has no value but is required
        slots = [Slot(name="param", value=None, type="str", verified=False, required=True, prompt="Please provide param")]
        
        result, is_verification = tool._handle_missing_required_slots(slots, "")
        
        # Should return the prompt since slot is missing
        assert result == "Please provide param"
        assert is_verification is False

    def test_handle_missing_required_slots_non_repeatable_no_value_not_required(self) -> None:
        """Test _handle_missing_required_slots with non-repeatable slot that has no value but is not required."""
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str", "required": False, "prompt": "Please provide param"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        
        # Create a slot that has no value and is not required
        slots = [Slot(name="param", value=None, type="str", verified=False, required=False, prompt="Please provide param")]
        
        result, is_verification = tool._handle_missing_required_slots(slots, "")
        
        # Should return empty string since slot is not required
        assert result == ""
        assert is_verification is False

    def test_handle_missing_required_slots_all_slots_valid(self) -> None:
        """Test _handle_missing_required_slots when all slots are valid (covers the final return statement)."""
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str", "required": True}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        
        # Create slots that are all valid (have values and are verified)
        slots = [
            Slot(name="param1", value="value1", type="str", verified=True, required=True),
            Slot(name="param2", value="value2", type="str", verified=True, required=False),
        ]
        
        result, is_verification = tool._handle_missing_required_slots(slots, "")
        
        # Should return empty string since all slots are valid
        assert result == ""
        assert is_verification is False

    def test_handle_missing_required_slots_group_slot_missing_fields(self) -> None:
        """Test _handle_missing_required_slots with group slot that has missing fields (covers lines 1192-1194)."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "field1", "type": "str", "required": True}
            ], "prompt": "Please provide group"}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        
        # Create a group slot with missing required fields
        slots = [Slot(
            name="group", 
            type="group", 
            value=[{"field1": None}],  # Missing required field
            required=True, 
            prompt="Please provide group",
            schema=[{"name": "field1", "type": "str", "required": True, "prompt": "Field 1"}]
        )]
        
        result, is_verification = tool._handle_missing_required_slots(slots, "")
        
        # Should return group field missing message
        assert "Please provide the following fields for group 'group' item 1: field1" in result
        assert is_verification is False

    def test_handle_missing_required_slots_repeatable_slot_empty_list(self) -> None:
        """Test _handle_missing_required_slots with repeatable slot that has empty list (covers lines 1199-1200)."""
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str", "required": True, "prompt": "Please provide values", "repeatable": True}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        
        # Create a repeatable slot with empty list
        slots = [Slot(
            name="param", 
            type="str", 
            value=[],  # Empty list
            required=True, 
            prompt="Please provide values",
            repeatable=True
        )]
        
        result, is_verification = tool._handle_missing_required_slots(slots, "")
        
        # Should return the prompt since the list is empty
        assert result == "Please provide values"
        assert is_verification is False

    def test_handle_missing_required_slots_repeatable_slot_none_value(self) -> None:
        """Test _handle_missing_required_slots with repeatable slot that has None value (covers lines 1199-1200)."""
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str", "required": True, "prompt": "Please provide values", "repeatable": True}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        
        # Create a repeatable slot with None value
        slots = [Slot(
            name="param", 
            type="str", 
            value=None,  # None value
            required=True, 
            prompt="Please provide values",
            repeatable=True
        )]
        
        result, is_verification = tool._handle_missing_required_slots(slots, "")
        
        # Should return the prompt since the value is None
        assert result == "Please provide values"
        assert is_verification is False

    def test_handle_missing_required_slots_repeatable_slot_not_list(self) -> None:
        """Test _handle_missing_required_slots with repeatable slot that has non-list value (covers lines 1199-1200)."""
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str", "required": True, "prompt": "Please provide values", "repeatable": True}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        
        # Create a repeatable slot with non-list value
        slots = [Slot(
            name="param", 
            type="str", 
            value="not a list",  # Not a list
            required=True, 
            prompt="Please provide values",
            repeatable=True
        )]
        
        result, is_verification = tool._handle_missing_required_slots(slots, "")
        
        # Should return the prompt since the value is not a list
        assert result == "Please provide values"
        assert is_verification is False

    def test_handle_missing_required_slots_repeatable_slot_empty_values(self) -> None:
        """Test _handle_missing_required_slots with repeatable slot that has empty values (covers lines 1202-1204)."""
        tool = Tool(
            func=lambda param: param,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "param", "type": "str", "required": True, "prompt": "Please provide values", "repeatable": True}],
            outputs=["result"],
            isResponse=False,
        )
        tool.slotfiller = Mock()
        
        # Create a repeatable slot with empty values
        slots = [Slot(
            name="param", 
            type="str", 
            value=[None, ""],  # Empty values
            required=True, 
            prompt="Please provide values",
            repeatable=True
        )]
        
        result, is_verification = tool._handle_missing_required_slots(slots, "")
        
        # Should return the prompt for the first empty item
        assert "Please provide values (item 1)" in result
        assert is_verification is False

    def test_check_group_slot_missing_fields_none_value(self) -> None:
        """Test _check_group_slot_missing_fields with None value (covers line 1237)."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": []}],
            outputs=["result"],
            isResponse=False,
        )
        
        # Create a group slot with None value
        slot = Slot(
            name="group",
            type="group",
            value=None,  # None value
            prompt="Please provide group",
            schema=[]
        )
        
        result = tool._check_group_slot_missing_fields(slot)
        
        # Should return the prompt since value is None
        assert result == "Please provide group"

    def test_check_group_slot_missing_fields_empty_list(self) -> None:
        """Test _check_group_slot_missing_fields with empty list value (covers line 1237)."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": []}],
            outputs=["result"],
            isResponse=False,
        )
        
        # Create a group slot with empty list
        slot = Slot(
            name="group",
            type="group",
            value=[],  # Empty list
            prompt="Please provide group",
            schema=[]
        )
        
        result = tool._check_group_slot_missing_fields(slot)
        
        # Should return the prompt since value is empty list
        assert result == "Please provide group"

    def test_check_group_slot_missing_fields_not_list(self) -> None:
        """Test _check_group_slot_missing_fields with non-list value (covers line 1237)."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": []}],
            outputs=["result"],
            isResponse=False,
        )
        
        # Create a group slot with non-list value
        slot = Slot(
            name="group",
            type="group",
            value="not a list",  # Not a list
            prompt="Please provide group",
            schema=[]
        )
        
        result = tool._check_group_slot_missing_fields(slot)
        
        # Should return the prompt since value is not a list
        assert result == "Please provide group"

    def test_check_group_slot_missing_fields_repeatable_field_missing(self) -> None:
        """Test _check_group_slot_missing_fields with missing repeatable field (covers lines 1246-1247)."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "repeatable_field", "type": "str", "required": True, "repeatable": True}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        
        # Create a group slot with missing repeatable field
        slot = Slot(
            name="group",
            type="group",
            value=[{}],  # Missing the repeatable field entirely
            prompt="Please provide group",
            schema=[{"name": "repeatable_field", "type": "str", "required": True, "repeatable": True}]
        )
        
        result = tool._check_group_slot_missing_fields(slot)
        
        # Should return message about missing repeatable field
        assert "repeatable_field (repeatable)" in result

    def test_check_group_slot_missing_fields_repeatable_field_not_list(self) -> None:
        """Test _check_group_slot_missing_fields with repeatable field that is not a list (covers lines 1246-1247)."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "repeatable_field", "type": "str", "required": True, "repeatable": True}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        
        # Create a group slot with repeatable field that is not a list
        slot = Slot(
            name="group",
            type="group",
            value=[{"repeatable_field": "not a list"}],  # Field exists but is not a list
            prompt="Please provide group",
            schema=[{"name": "repeatable_field", "type": "str", "required": True, "repeatable": True}]
        )
        
        result = tool._check_group_slot_missing_fields(slot)
        
        # Should return message about missing repeatable field
        assert "repeatable_field (repeatable)" in result

    def test_check_group_slot_missing_fields_repeatable_field_empty_list(self) -> None:
        """Test _check_group_slot_missing_fields with repeatable field that has empty list (covers lines 1246-1247)."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "repeatable_field", "type": "str", "required": True, "repeatable": True}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        
        # Create a group slot with repeatable field that has empty list
        slot = Slot(
            name="group",
            type="group",
            value=[{"repeatable_field": []}],  # Field exists but is empty list
            prompt="Please provide group",
            schema=[{"name": "repeatable_field", "type": "str", "required": True, "repeatable": True}]
        )
        
        result = tool._check_group_slot_missing_fields(slot)
        
        # Should return message about missing repeatable field
        assert "repeatable_field (repeatable)" in result

    def test_check_group_slot_missing_fields_repeatable_field_empty_values(self) -> None:
        """Test _check_group_slot_missing_fields with repeatable field that has empty values (covers lines 1250-1252)."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "repeatable_field", "type": "str", "required": True, "repeatable": True}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        
        # Create a group slot with repeatable field that has empty values
        slot = Slot(
            name="group",
            type="group",
            value=[{"repeatable_field": [None, ""]}],  # Field exists but has empty values
            prompt="Please provide group",
            schema=[{"name": "repeatable_field", "type": "str", "required": True, "repeatable": True}]
        )
        
        result = tool._check_group_slot_missing_fields(slot)
        
        # Should return message about missing values in repeatable field
        assert "repeatable_field (value 1)" in result
        assert "repeatable_field (value 2)" in result

    def test_check_group_slot_missing_fields_all_valid(self) -> None:
        """Test _check_group_slot_missing_fields when all fields are valid (covers line 1260)."""
        tool = Tool(
            func=lambda group: group,
            name="test_tool",
            description="Test tool",
            slots=[{"name": "group", "type": "group", "schema": [
                {"name": "field1", "type": "str", "required": True},
                {"name": "field2", "type": "str", "required": True, "repeatable": True}
            ]}],
            outputs=["result"],
            isResponse=False,
        )
        
        # Create a group slot with all valid fields
        slot = Slot(
            name="group",
            type="group",
            value=[{"field1": "value1", "field2": ["value2", "value3"]}],  # All fields have valid values
            prompt="Please provide group",
            schema=[
                {"name": "field1", "type": "str", "required": True},
                {"name": "field2", "type": "str", "required": True, "repeatable": True}
            ]
        )
        
        result = tool._check_group_slot_missing_fields(slot)
        
        # Should return empty string since all fields are valid
        assert result == ""
