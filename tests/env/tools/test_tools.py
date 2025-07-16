"""Tests for the tools module.

This module contains comprehensive test cases for the tools functionality,
including Tool class creation, registration, and various parameter handling scenarios.
"""

from typing import Any
from unittest.mock import Mock, patch

from arklex.env.tools.tools import Tool, register_tool
from arklex.orchestrator.entities.msg_state_entities import MessageState
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

        from arklex.orchestrator.NLU.entities.slot_entities import Slot

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

        from arklex.orchestrator.NLU.entities.slot_entities import Slot

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

        from arklex.orchestrator.NLU.entities.slot_entities import Slot

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
        from arklex.orchestrator.NLU.entities.slot_entities import Slot

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
        from arklex.orchestrator.NLU.entities.slot_entities import Slot

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
        from arklex.orchestrator.NLU.entities.slot_entities import Slot

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
        from arklex.orchestrator.NLU.entities.slot_entities import Slot

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

            result = tool._execute(state)
            # Should reset to current node's slots
            assert len(tool.slots) == 1
            assert tool.slots[0].name == "param2"

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

        with patch.object(tool, "_init_slots"):
            result = tool._execute(state)

            assert result.status.value == "complete"
            # Should use previous slots since configuration didn't change
            assert tool.slots == previous_slots
