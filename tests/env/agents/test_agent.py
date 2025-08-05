from typing import Any
from unittest.mock import Mock, patch

import pytest

from arklex.env.agents.agent import AgentOutput, BaseAgent, register_agent
from arklex.orchestrator.entities.orchestrator_state_entities import (
    OrchestratorState,
    StatusEnum,
)


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

    def init_agent_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        pass

    def _execute(self) -> tuple[OrchestratorState, AgentOutput]:  # noqa: ANN401
        """Mock implementation of _execute method."""
        mock_state = Mock(spec=OrchestratorState)
        mock_state.trajectory = [[Mock()]]
        mock_state.trajectory[0][0].output = None
        agent_output = AgentOutput(
            response="Test response",
            status=StatusEnum.COMPLETE,
        )
        return mock_state, agent_output


class FailingAgent(BaseAgent):  # noqa: D101
    """Agent that raises exceptions for testing error handling."""

    def init_agent_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        pass

    def _execute(self) -> tuple[OrchestratorState, AgentOutput]:  # noqa: ANN401
        """Implementation that raises an exception."""
        raise ValueError("Test error")


@pytest.fixture
def mock_state() -> OrchestratorState:
    """Create a mock OrchestratorState for testing."""
    state = Mock(spec=OrchestratorState)
    state.status = StatusEnum.INCOMPLETE
    state.response = ""
    state.message_flow = ""
    state.trajectory = [[Mock()]]
    state.trajectory[0][0].output = None
    return state


@pytest.fixture
def complete_mock_state() -> OrchestratorState:
    """Create a mock OrchestratorState with COMPLETE status."""
    state = Mock(spec=OrchestratorState)
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

            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                mock_state = Mock(spec=OrchestratorState)
                agent_output = AgentOutput(response="", status=StatusEnum.INCOMPLETE)
                return mock_state, agent_output

        agent = TestAgent()
        assert agent.description == "Test agent description"

    def test_execute_success(self, mock_state: OrchestratorState) -> None:
        """Test successful execution of agent."""
        agent = ConcreteAgent()
        result_state, agent_output = agent.execute(mock_state, node_specific_data={})

        # Verify the result
        assert agent_output.status == StatusEnum.INCOMPLETE
        assert agent_output.response == "Test response"

    def test_execute_with_message_flow_fallback(
        self, mock_state: OrchestratorState
    ) -> None:  # noqa: E501
        """Test execution when response is empty but message_flow has content."""

        class EmptyResponseAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                mock_state = Mock(spec=OrchestratorState)
                mock_state.trajectory = [[Mock()]]
                mock_state.trajectory[0][0].output = None
                agent_output = AgentOutput(response="", status=StatusEnum.INCOMPLETE)
                return mock_state, agent_output

        agent = EmptyResponseAgent()
        _, agent_output = agent.execute(mock_state, node_specific_data={})

        assert agent_output.status == StatusEnum.INCOMPLETE
        assert agent_output.response == ""

    def test_execute_with_empty_trajectory(self, mock_state: OrchestratorState) -> None:
        """Test execution when trajectory is empty."""

        class EmptyTrajectoryAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                mock_state = Mock(spec=OrchestratorState)
                mock_state.trajectory = []
                agent_output = AgentOutput(
                    response="Test", status=StatusEnum.INCOMPLETE
                )
                return mock_state, agent_output

        agent = EmptyTrajectoryAgent()
        orch_state, _ = agent.execute(mock_state, node_specific_data={})

        # Should not raise an exception even with empty trajectory
        assert True  # Test passes if no exception is raised

    def test_execute_with_none_trajectory(self, mock_state: OrchestratorState) -> None:
        """Test execution when trajectory is None."""

        class NoneTrajectoryAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                mock_state = Mock(spec=OrchestratorState)
                mock_state.trajectory = None
                agent_output = AgentOutput(
                    response="Test", status=StatusEnum.INCOMPLETE
                )
                return mock_state, agent_output

        agent = NoneTrajectoryAgent()
        result_state, result_output = agent.execute(mock_state, node_specific_data={})

        # Should return a tuple, not just the state
        assert isinstance(result_output, AgentOutput)

    def test_execute_with_nested_empty_trajectory(
        self, mock_state: OrchestratorState
    ) -> None:
        """Test execution when trajectory has empty nested lists."""

        class NestedEmptyTrajectoryAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                mock_state = Mock(spec=OrchestratorState)
                mock_state.trajectory = [[Mock()]]
                mock_state.trajectory[0][0].output = None
                agent_output = AgentOutput(
                    response="Test response", status=StatusEnum.INCOMPLETE
                )
                return mock_state, agent_output

        agent = NestedEmptyTrajectoryAgent()
        _, agent_output = agent.execute(mock_state, node_specific_data={})

        assert agent_output.status == StatusEnum.INCOMPLETE
        assert agent_output.response == "Test response"

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

    def test_execute_model_validate_exception(
        self, mock_state: OrchestratorState
    ) -> None:
        """Test execute when _execute raises exception."""

        class ValidationExceptionAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                raise ValueError("Validation error")

        agent = ValidationExceptionAgent()
        result_state, result_output = agent.execute(mock_state, node_specific_data={})

        # Should return original state and error output when validation fails
        assert result_state == mock_state
        assert result_output.response == ""
        assert result_output.status == StatusEnum.INCOMPLETE

    @patch("arklex.env.agents.agent.log_context.error")
    @patch("arklex.env.agents.agent.traceback.format_exc")
    def test_execute_logs_specific_exception_type(
        self, mock_format_exc: Mock, mock_log_error: Mock, mock_state: OrchestratorState
    ) -> None:
        """Test that execute logs the specific exception traceback."""

        class CustomExceptionAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                raise RuntimeError("Custom runtime error")

        mock_format_exc.return_value = (
            "RuntimeError: Custom runtime error\nTraceback..."
        )

        agent = CustomExceptionAgent()
        result = agent.execute(mock_state, node_specific_data={})

        assert result[0] == mock_state
        mock_log_error.assert_called_once_with(
            "RuntimeError: Custom runtime error\nTraceback..."
        )

    def test_multiple_agents_different_names(self) -> None:
        """Test that multiple registered agents have different names."""

        @register_agent
        class FirstAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                mock_state = Mock(spec=OrchestratorState)
                agent_output = AgentOutput(response="", status=StatusEnum.INCOMPLETE)
                return mock_state, agent_output

        @register_agent
        class SecondAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                mock_state = Mock(spec=OrchestratorState)
                agent_output = AgentOutput(response="", status=StatusEnum.INCOMPLETE)
                return mock_state, agent_output

        assert FirstAgent.name == "FirstAgent"
        assert SecondAgent.name == "SecondAgent"
        assert FirstAgent.name != SecondAgent.name

    def test_execute_with_complex_kwargs(self, mock_state: OrchestratorState) -> None:
        """Test execute with complex kwargs including nested objects."""

        class ComplexKwargsAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                self.node_data = node_specific_data

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                nested_data = self.node_data.get("nested", {})
                mock_state = Mock(spec=OrchestratorState)
                mock_state.trajectory = [[Mock()]]
                mock_state.trajectory[0][0].output = None
                agent_output = AgentOutput(
                    response=f"Processed: {nested_data.get('key', 'default')}",
                    status=StatusEnum.COMPLETE,
                )
                return mock_state, agent_output

        agent = ComplexKwargsAgent()
        result_state, result_output = agent.execute(
            mock_state,
            node_specific_data={
                "nested": {"key": "test_value"},
                "other_param": 42,
                "flag": True,
            },
        )

        assert "test_value" in result_output.response

    def test_agent_inheritance_preserves_description(self) -> None:
        """Test that agent inheritance preserves description from parent class."""

        class ParentAgent(BaseAgent):
            description = "Parent description"

            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                mock_state = Mock(spec=OrchestratorState)
                agent_output = AgentOutput(response="", status=StatusEnum.INCOMPLETE)
                return mock_state, agent_output

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

            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                mock_state = Mock(spec=OrchestratorState)
                agent_output = AgentOutput(response="", status=StatusEnum.INCOMPLETE)
                return mock_state, agent_output

        class ChildAgent(ParentAgent):
            description = "Child description"

        parent = ParentAgent()
        child = ChildAgent()

        assert parent.description == "Parent description"
        assert child.description == "Child description"

    def test_execute_with_kwargs(self, mock_state: OrchestratorState) -> None:
        """Test execute method with additional kwargs."""

        # Create a test agent that uses kwargs
        class KwargsTestAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                self.node_data = node_specific_data

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                mock_state = Mock(spec=OrchestratorState)
                mock_state.trajectory = [[Mock()]]
                mock_state.trajectory[0][0].output = None
                agent_output = AgentOutput(
                    response=f"Response with {self.node_data.get('test_param', 'default')}",
                    status=StatusEnum.COMPLETE,
                )
                return mock_state, agent_output

        agent = KwargsTestAgent()
        result_state, result_output = agent.execute(
            mock_state, node_specific_data={"test_param": "custom_value"}
        )

        # Verify the result includes the custom parameter
        assert "custom_value" in result_output.response

    def test_execute_with_exception_returns_original_state(
        self, mock_state: OrchestratorState
    ) -> None:
        """Test execute method when _execute raises an exception - should return original state."""

        # Create an agent that raises an exception in _execute
        class ExceptionAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                raise RuntimeError("Test exception")

        agent = ExceptionAgent()
        result_state, result_output = agent.execute(mock_state, node_specific_data={})

        # Should return the original message state when exception occurs
        assert result_state == mock_state
        assert result_output.response == ""
        assert result_output.status == StatusEnum.INCOMPLETE

    def test_execute_with_exception_logs_error(
        self, mock_state: OrchestratorState
    ) -> None:
        """Test execute method logs error when _execute raises an exception."""
        from unittest.mock import patch

        # Create an agent that raises an exception in _execute
        class ExceptionAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                raise ValueError("Test exception")

        agent = ExceptionAgent()

        # Patch the log_context.error to verify it's called
        with patch("arklex.env.agents.agent.log_context.error") as mock_error:
            agent.execute(mock_state, node_specific_data={})

            # Verify error was logged
            mock_error.assert_called_once()
            # Verify the call includes traceback.format_exc()
            assert "Test exception" in mock_error.call_args[0][0]

    def test_execute_with_exception_logs_traceback(
        self, mock_state: OrchestratorState
    ) -> None:
        """Test execute method logs traceback when _execute raises an exception."""
        from unittest.mock import patch

        # Create an agent that raises an exception in _execute
        class ExceptionAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                raise RuntimeError("Test runtime exception")

        agent = ExceptionAgent()

        # Patch both log_context.error and traceback.format_exc to verify they're called
        with (
            patch("arklex.env.agents.agent.log_context.error") as mock_error,
            patch("arklex.env.agents.agent.traceback.format_exc") as mock_format_exc,
        ):
            mock_format_exc.return_value = "Test traceback"
            agent.execute(mock_state, node_specific_data={})

            # Verify traceback.format_exc was called
            mock_format_exc.assert_called_once()
            # Verify error was logged with the traceback
            mock_error.assert_called_once_with("Test traceback")

    def test_execute_with_exception_returns_original_state_different_exception(
        self, mock_state: OrchestratorState
    ) -> None:
        """Test execute method returns original state for different exception types."""

        # Create an agent that raises a different exception in _execute
        class DifferentExceptionAgent(BaseAgent):
            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                raise TypeError("Test type error")

        agent = DifferentExceptionAgent()
        result_state, result_output = agent.execute(mock_state, node_specific_data={})

        # Should return the original message state when exception occurs
        assert result_state == mock_state
        assert result_output.response == ""
        assert result_output.status == StatusEnum.INCOMPLETE


class TestAgentIntegration:
    """Integration tests for agent functionality."""

    def test_registered_agent_with_execution(
        self, mock_state: OrchestratorState
    ) -> None:
        """Test a registered agent can be executed successfully."""

        @register_agent
        class IntegrationTestAgent(BaseAgent):
            description = "Integration test agent"

            def init_agent_data(
                self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
            ) -> None:
                pass

            def _execute(self) -> tuple[OrchestratorState, AgentOutput]:
                mock_response_state = Mock(spec=OrchestratorState)
                mock_response_state.trajectory = [[Mock()]]
                mock_response_state.trajectory[0][0].output = None
                agent_output = AgentOutput(
                    response="Integration test complete",
                    status=StatusEnum.COMPLETE,
                )
                return mock_response_state, agent_output

        agent = IntegrationTestAgent()

        # Test registration
        assert agent.name == "IntegrationTestAgent"
        assert agent.description == "Integration test agent"

        # Test execution
        result_state, result_output = agent.execute(mock_state, node_specific_data={})
        assert result_output.response == "Integration test complete"
