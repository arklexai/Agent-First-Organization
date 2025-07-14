from typing import Any
from unittest.mock import Mock, patch

import pytest

from arklex.env.agents.agent import BaseAgent, register_agent
from arklex.orchestrator.entities.msg_state_entities import MessageState, StatusEnum


class TestRegisterAgent:
    """Test class for register_agent decorator."""

    def test_register_agent_sets_name(self) -> None:
        """Test that register_agent decorator sets the name attribute."""

        @register_agent
        class TestAgent:
            pass

        assert TestAgent.name == "TestAgent"

    def test_register_agent_returns_class(self) -> None:
        """Test that register_agent decorator returns the original class."""

        @register_agent
        class TestAgent:
            pass

        assert TestAgent.__name__ == "TestAgent"

    def test_register_agent_preserves_existing_attributes(self) -> None:
        """Test that register_agent preserves existing class attributes."""

        @register_agent
        class TestAgent:
            description = "Test description"
            custom_attr = "custom value"

        assert TestAgent.name == "TestAgent"
        assert TestAgent.description == "Test description"
        assert TestAgent.custom_attr == "custom value"


class ConcreteAgent(BaseAgent):  # noqa: D101
    """Concrete implementation of BaseAgent for testing."""

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Mock implementation of _execute method."""
        return {
            "status": StatusEnum.COMPLETE,
            "response": "Test response",
            "message_flow": "",
            "function_calling_trajectory": [],
            "trajectory": msg_state.trajectory,
        }


class FailingAgent(BaseAgent):  # noqa: D101
    """Agent that raises exceptions for testing error handling."""

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Implementation that raises an exception."""
        raise ValueError("Test error")


@pytest.fixture
def mock_state() -> MessageState:
    """Create a mock MessageState for testing."""
    state = Mock(spec=MessageState)
    state.status = StatusEnum.INCOMPLETE
    state.response = ""
    state.message_flow = ""
    state.trajectory = [[Mock()]]
    state.trajectory[0][0].output = None
    return state


@pytest.fixture
def complete_mock_state() -> MessageState:
    """Create a mock MessageState with COMPLETE status."""
    state = Mock(spec=MessageState)
    state.status = StatusEnum.COMPLETE
    state.response = "Existing response"
    state.message_flow = ""
    state.trajectory = [[Mock()]]
    state.trajectory[0][0].output = None
    return state


class TestBaseAgent:
    """Test class for BaseAgent."""

    def test_str_representation(self) -> None:
        """Test __str__ method returns class name."""
        agent = ConcreteAgent()
        assert str(agent) == "ConcreteAgent"

    def test_repr_representation(self) -> None:
        """Test __repr__ method returns class name."""
        agent = ConcreteAgent()
        assert repr(agent) == "ConcreteAgent"

    def test_description_default_none(self) -> None:
        """Test that description defaults to None."""
        agent = ConcreteAgent()
        assert agent.description is None

    def test_description_can_be_set(self) -> None:
        """Test that description can be set on subclasses."""

        class TestAgent(BaseAgent):
            description = "Test agent description"

            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:  # noqa: ANN401
                return {}

        agent = TestAgent()
        assert agent.description == "Test agent description"

    @patch("arklex.env.agents.agent.MessageState.model_validate")
    def test_execute_success(
        self, mock_validate: Mock, mock_state: MessageState
    ) -> None:
        """Test successful execution of agent."""
        # Setup mock return value
        mock_response_state = Mock(spec=MessageState)
        mock_response_state.response = "Test response"
        mock_response_state.message_flow = ""
        mock_response_state.trajectory = [[Mock()]]
        mock_response_state.trajectory[0][0].output = None
        mock_validate.return_value = mock_response_state

        agent = ConcreteAgent()
        result = agent.execute(mock_state)

        # Verify the result
        assert result == mock_response_state
        assert mock_response_state.trajectory[0][0].output == "Test response"
        mock_validate.assert_called_once()

    @patch("arklex.env.agents.agent.MessageState.model_validate")
    def test_execute_with_message_flow_fallback(
        self, mock_validate: Mock, mock_state: MessageState
    ) -> None:  # noqa: E501
        """Test execution when response is empty but message_flow has content."""
        # Setup mock return value with empty response but message_flow content
        mock_response_state = Mock(spec=MessageState)
        mock_response_state.response = ""
        mock_response_state.message_flow = "Message flow content"
        mock_response_state.trajectory = [[Mock()]]
        mock_response_state.trajectory[0][0].output = None
        mock_validate.return_value = mock_response_state

        agent = ConcreteAgent()
        result = agent.execute(mock_state)

        assert result == mock_response_state
        assert mock_response_state.trajectory[0][0].output == "Message flow content"

    @patch("arklex.env.agents.agent.MessageState.model_validate")
    def test_execute_with_none_response_and_message_flow(
        self, mock_validate: Mock, mock_state: MessageState
    ) -> None:
        """Test execution when both response and message_flow are None."""
        # Setup mock return value with None values
        mock_response_state = Mock(spec=MessageState)
        mock_response_state.response = None
        mock_response_state.message_flow = None
        mock_response_state.trajectory = [[Mock()]]
        mock_response_state.trajectory[0][0].output = None
        mock_validate.return_value = mock_response_state

        agent = ConcreteAgent()
        result = agent.execute(mock_state)

        assert result == mock_response_state
        assert mock_response_state.trajectory[0][0].output is None

    @patch("arklex.env.agents.agent.MessageState.model_validate")
    def test_execute_with_empty_trajectory(
        self, mock_validate: Mock, mock_state: MessageState
    ) -> None:
        """Test execution when trajectory is empty."""
        mock_response_state = Mock(spec=MessageState)
        mock_response_state.response = "Test response"
        mock_response_state.message_flow = ""
        mock_response_state.trajectory = []
        mock_validate.return_value = mock_response_state

        agent = ConcreteAgent()
        result = agent.execute(mock_state)

        assert result == mock_response_state
        # Should not raise an exception even with empty trajectory

    @patch("arklex.env.agents.agent.MessageState.model_validate")
    def test_execute_with_none_trajectory(
        self, mock_validate: Mock, mock_state: MessageState
    ) -> None:
        """Test execution when trajectory is None."""
        mock_response_state = Mock(spec=MessageState)
        mock_response_state.response = "Test response"
        mock_response_state.message_flow = ""
        mock_response_state.trajectory = None
        mock_validate.return_value = mock_response_state

        agent = ConcreteAgent()
        result = agent.execute(mock_state)

        assert result == mock_response_state

    @patch("arklex.env.agents.agent.MessageState.model_validate")
    def test_execute_with_nested_empty_trajectory(
        self, mock_validate: Mock, mock_state: MessageState
    ) -> None:
        """Test execution when trajectory has empty nested lists."""
        mock_response_state = Mock(spec=MessageState)
        mock_response_state.response = "Test response"
        mock_response_state.message_flow = ""
        mock_response_state.trajectory = [[]]
        mock_validate.return_value = mock_response_state

        agent = ConcreteAgent()
        result = agent.execute(mock_state)

        assert result == mock_response_state

    def test_register_agent_with_existing_name(self) -> None:
        """Test register_agent when class already has a name attribute."""

        @register_agent
        class TestAgent:
            name = "ExistingName"

        # Should override existing name with class name
        assert TestAgent.name == "TestAgent"

    def test_register_agent_multiple_decorations(self) -> None:
        """Test that register_agent can be applied multiple times."""

        @register_agent
        @register_agent
        class TestAgent:
            pass

        assert TestAgent.name == "TestAgent"

    def test_base_agent_name_attribute_not_set_by_default(self) -> None:
        """Test that BaseAgent doesn't have name set by default."""
        # This should raise AttributeError since name is not set
        with pytest.raises(AttributeError):
            agent = ConcreteAgent()
            _ = agent.name

    def test_agent_with_custom_name(self) -> None:
        """Test agent with manually set name."""
        agent = ConcreteAgent()
        agent.name = "CustomAgentName"

        assert agent.name == "CustomAgentName"
        assert str(agent) == "ConcreteAgent"  # str still uses class name

    @patch("arklex.env.agents.agent.MessageState.model_validate")
    def test_execute_model_validate_exception(
        self, mock_validate: Mock, mock_state: MessageState
    ) -> None:
        """Test execute when MessageState.model_validate raises exception."""
        mock_validate.side_effect = ValueError("Validation error")

        agent = ConcreteAgent()
        result = agent.execute(mock_state)

        # Should return original state when validation fails
        assert result == mock_state

    @patch("arklex.env.agents.agent.log_context.error")
    @patch("arklex.env.agents.agent.traceback.format_exc")
    def test_execute_logs_specific_exception_type(
        self, mock_format_exc: Mock, mock_log_error: Mock, mock_state: MessageState
    ) -> None:
        """Test that execute logs the specific exception traceback."""

        class CustomExceptionAgent(BaseAgent):
            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:
                raise RuntimeError("Custom runtime error")

        mock_format_exc.return_value = (
            "RuntimeError: Custom runtime error\nTraceback..."
        )

        agent = CustomExceptionAgent()
        result = agent.execute(mock_state)

        assert result == mock_state
        mock_log_error.assert_called_once_with(
            "RuntimeError: Custom runtime error\nTraceback..."
        )

    def test_multiple_agents_different_names(self) -> None:
        """Test that multiple registered agents have different names."""

        @register_agent
        class FirstAgent(BaseAgent):
            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:
                return {}

        @register_agent
        class SecondAgent(BaseAgent):
            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:
                return {}

        assert FirstAgent.name == "FirstAgent"
        assert SecondAgent.name == "SecondAgent"
        assert FirstAgent.name != SecondAgent.name

    @patch("arklex.env.agents.agent.MessageState.model_validate")
    def test_execute_with_complex_kwargs(
        self, mock_validate: Mock, mock_state: MessageState
    ) -> None:
        """Test execute with complex kwargs including nested objects."""

        class ComplexKwargsAgent(BaseAgent):
            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:
                nested_data = kwargs.get("nested", {})
                return {
                    "status": StatusEnum.COMPLETE,
                    "response": f"Processed: {nested_data.get('key', 'default')}",
                    "message_flow": "",
                    "function_calling_trajectory": [],
                    "trajectory": msg_state.trajectory,
                }

        mock_response_state = Mock(spec=MessageState)
        mock_response_state.response = "Processed: test_value"
        mock_response_state.message_flow = ""
        mock_response_state.trajectory = [[Mock()]]
        mock_response_state.trajectory[0][0].output = None
        mock_validate.return_value = mock_response_state

        agent = ComplexKwargsAgent()
        result = agent.execute(
            mock_state, nested={"key": "test_value"}, other_param=42, flag=True
        )

        assert result == mock_response_state

    def test_agent_inheritance_preserves_description(self) -> None:
        """Test that agent inheritance preserves description from parent class."""

        class ParentAgent(BaseAgent):
            description = "Parent description"

            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:
                return {}

        class ChildAgent(ParentAgent):
            pass

        parent = ParentAgent()
        child = ChildAgent()

        assert parent.description == "Parent description"
        assert child.description == "Parent description"

    def test_agent_description_override(self) -> None:
        """Test that child agent can override parent description."""

        class ParentAgent(BaseAgent):
            description = "Parent description"

            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:
                return {}

        class ChildAgent(ParentAgent):
            description = "Child description"

        parent = ParentAgent()
        child = ChildAgent()

        assert parent.description == "Parent description"
        assert child.description == "Child description"

    def test_execute_with_kwargs(self, mock_state: MessageState) -> None:
        """Test execute method with additional kwargs."""
        from unittest.mock import patch

        # Create a test agent that uses kwargs
        class KwargsTestAgent(BaseAgent):
            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:  # noqa: ANN401
                return {
                    "status": StatusEnum.COMPLETE,
                    "response": f"Response with {kwargs.get('test_param', 'default')}",
                    "message_flow": "",
                    "function_calling_trajectory": [],
                    "trajectory": msg_state.trajectory,
                }

        agent = KwargsTestAgent()

        # Mock MessageState.model_validate to return a proper MessageState
        with patch(
            "arklex.env.agents.agent.MessageState.model_validate"
        ) as mock_validate:
            mock_response_state = Mock(spec=MessageState)
            mock_response_state.response = "Response with custom_value"
            mock_response_state.message_flow = ""
            mock_response_state.trajectory = [[Mock()]]
            mock_response_state.trajectory[0][0].output = None
            mock_validate.return_value = mock_response_state

            result = agent.execute(mock_state, test_param="custom_value")

            # Verify the result includes the custom parameter
            assert "custom_value" in result.response

    def test_execute_with_exception_returns_original_state(
        self, mock_state: MessageState
    ) -> None:
        """Test execute method when _execute raises an exception - should return original state."""

        # Create an agent that raises an exception in _execute
        class ExceptionAgent(BaseAgent):
            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:  # noqa: ANN401
                raise RuntimeError("Test exception")

        agent = ExceptionAgent()
        result = agent.execute(mock_state)

        # Should return the original message state when exception occurs
        assert result == mock_state

    def test_execute_with_exception_logs_error(self, mock_state: MessageState) -> None:
        """Test execute method logs error when _execute raises an exception."""
        from unittest.mock import patch

        # Create an agent that raises an exception in _execute
        class ExceptionAgent(BaseAgent):
            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:  # noqa: ANN401
                raise ValueError("Test exception")

        agent = ExceptionAgent()

        # Patch the log_context.error to verify it's called
        with patch("arklex.env.agents.agent.log_context.error") as mock_error:
            agent.execute(mock_state)

            # Verify error was logged
            mock_error.assert_called_once()
            # Verify the call includes traceback.format_exc()
            assert "Test exception" in mock_error.call_args[0][0]

    def test_execute_with_exception_logs_traceback(
        self, mock_state: MessageState
    ) -> None:
        """Test execute method logs traceback when _execute raises an exception."""
        from unittest.mock import patch

        # Create an agent that raises an exception in _execute
        class ExceptionAgent(BaseAgent):
            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:  # noqa: ANN401
                raise RuntimeError("Test runtime exception")

        agent = ExceptionAgent()

        # Patch both log_context.error and traceback.format_exc to verify they're called
        with (
            patch("arklex.env.agents.agent.log_context.error") as mock_error,
            patch("arklex.env.agents.agent.traceback.format_exc") as mock_format_exc,
        ):
            mock_format_exc.return_value = "Test traceback"
            agent.execute(mock_state)

            # Verify traceback.format_exc was called
            mock_format_exc.assert_called_once()
            # Verify error was logged with the traceback
            mock_error.assert_called_once_with("Test traceback")

    def test_execute_with_exception_returns_original_state_different_exception(
        self, mock_state: MessageState
    ) -> None:
        """Test execute method returns original state for different exception types."""

        # Create an agent that raises a different exception in _execute
        class DifferentExceptionAgent(BaseAgent):
            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:  # noqa: ANN401
                raise TypeError("Test type error")

        agent = DifferentExceptionAgent()
        result = agent.execute(mock_state)

        # Should return the original message state when exception occurs
        assert result == mock_state

    def test_abstract_execute_method(self) -> None:
        """Test that BaseAgent cannot be instantiated due to abstract method."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseAgent()


class TestAgentIntegration:
    """Integration tests for agent functionality."""

    def test_registered_agent_with_execution(self, mock_state: MessageState) -> None:
        """Test a registered agent can be executed successfully."""

        @register_agent
        class IntegrationTestAgent(BaseAgent):
            description = "Integration test agent"

            def _execute(
                self,
                msg_state: MessageState,
                **kwargs: Any,  # noqa: ANN401
            ) -> dict[str, Any]:  # noqa: ANN401, E501
                return {
                    "status": StatusEnum.COMPLETE,
                    "response": "Integration test complete",
                    "message_flow": "",
                    "function_calling_trajectory": [],
                    "trajectory": msg_state.trajectory,
                }

        agent = IntegrationTestAgent()

        # Test registration
        assert agent.name == "IntegrationTestAgent"
        assert agent.description == "Integration test agent"

        # Test execution
        with patch(
            "arklex.env.agents.agent.MessageState.model_validate"
        ) as mock_validate:
            mock_response_state = Mock(spec=MessageState)
            mock_response_state.response = "Integration test complete"
            mock_response_state.message_flow = ""
            mock_response_state.trajectory = [[Mock()]]
            mock_response_state.trajectory[0][0].output = None
            mock_validate.return_value = mock_response_state

            agent.execute(mock_state)
            assert (
                mock_response_state.trajectory[0][0].output
                == "Integration test complete"
            )
