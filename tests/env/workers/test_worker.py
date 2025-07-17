"""Comprehensive tests for the worker module.

This module provides comprehensive test coverage for the worker module,
ensuring all functionality is properly tested including error handling and edge cases.
"""

from typing import Any, NoReturn

import pytest

from arklex.env.workers.worker import BaseWorker, WorkerKwargs, register_worker
from arklex.memory.entities.memory_entities import ResourceRecord
from arklex.orchestrator.entities.msg_state_entities import MessageState, StatusEnum


class TestRegisterWorkerDecorator:
    """Test the register_worker decorator."""

    def test_register_worker_decorator(self) -> None:
        """Test that register_worker decorator sets the name attribute."""

        @register_worker
        class TestWorker(BaseWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {"status": "complete"}

        assert TestWorker.name == "TestWorker"

    def test_register_worker_preserves_class(self) -> None:
        """Test that register_worker preserves the original class."""

        @register_worker
        class TestWorker(BaseWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {"status": "complete"}

        assert issubclass(TestWorker, BaseWorker)
        assert TestWorker.__name__ == "TestWorker"


class TestBaseWorker:
    """Test the BaseWorker abstract class."""

    def test_base_worker_str_representation(self) -> None:
        """Test string representation of BaseWorker."""

        class TestWorker(BaseWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {"status": "complete"}

        worker = TestWorker()
        assert str(worker) == "TestWorker"

    def test_base_worker_repr_representation(self) -> None:
        """Test detailed string representation of BaseWorker."""

        class TestWorker(BaseWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {"status": "complete"}

        worker = TestWorker()
        assert repr(worker) == "TestWorker"

    def test_base_worker_description_default(self) -> None:
        """Test that description is None by default."""

        class TestWorker(BaseWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {"status": "complete"}

        worker = TestWorker()
        assert worker.description is None

    def test_base_worker_description_custom(self) -> None:
        """Test that description can be set."""

        class TestWorker(BaseWorker):
            description = "A test worker"

            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {"status": "complete"}

        worker = TestWorker()
        assert worker.description == "A test worker"

    def test_base_worker_cannot_instantiate_abstract(self) -> None:
        """Test that BaseWorker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseWorker()


class TestConcreteWorker:
    """Test concrete worker implementations."""

    class SimpleWorker(BaseWorker):
        """A simple concrete worker for testing."""

        def _execute(
            self, msg_state: MessageState, **kwargs: WorkerKwargs
        ) -> dict[str, Any]:
            return {
                "status": StatusEnum.COMPLETE,
                "response": "Test response",
                "message_flow": "test flow",
            }

    class ErrorWorker(BaseWorker):
        """A worker that raises an exception for testing."""

        def _execute(self, msg_state: MessageState, **kwargs: WorkerKwargs) -> NoReturn:
            raise ValueError("Test error")

    class IncompleteWorker(BaseWorker):
        """A worker that returns incomplete status."""

        def _execute(
            self, msg_state: MessageState, **kwargs: WorkerKwargs
        ) -> dict[str, Any]:
            return {
                "status": StatusEnum.INCOMPLETE,
                "response": "Incomplete response",
                "message_flow": "incomplete flow",
            }

    class EmptyResponseWorker(BaseWorker):
        """A worker that returns empty response."""

        def _execute(
            self, msg_state: MessageState, **kwargs: WorkerKwargs
        ) -> dict[str, Any]:
            return {"status": StatusEnum.COMPLETE, "message_flow": "flow only"}

    def test_worker_execute_success(self) -> None:
        """Test successful worker execution."""
        worker = self.SimpleWorker()
        resource_record = ResourceRecord(info={}, intent="test")
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE, trajectory=[[resource_record]]
        )

        result = worker.execute(msg_state)

        assert result.status == StatusEnum.COMPLETE
        assert result.response == "Test response"
        assert result.message_flow == "test flow"
        assert len(result.trajectory) > 0
        assert result.trajectory[-1][-1].output == "Test response"

    def test_worker_execute_with_kwargs(self) -> None:
        """Test worker execution with additional kwargs."""
        worker = self.SimpleWorker()
        resource_record = ResourceRecord(info={}, intent="test")
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE, trajectory=[[resource_record]]
        )

        result = worker.execute(msg_state, test_param="value")

        assert result.status == StatusEnum.COMPLETE
        assert result.response == "Test response"

    def test_worker_execute_incomplete_status(self) -> None:
        """Test worker execution that returns incomplete status."""
        worker = self.IncompleteWorker()
        resource_record = ResourceRecord(info={}, intent="test")
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE, trajectory=[[resource_record]]
        )

        result = worker.execute(msg_state)

        # Should be converted to COMPLETE
        assert result.status == StatusEnum.COMPLETE
        assert result.response == "Incomplete response"

    def test_worker_execute_empty_response(self) -> None:
        """Test worker execution with empty response."""
        worker = self.EmptyResponseWorker()
        resource_record = ResourceRecord(info={}, intent="test")
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE, trajectory=[[resource_record]]
        )

        result = worker.execute(msg_state)

        assert result.status == StatusEnum.COMPLETE
        assert result.response == ""
        assert result.trajectory[-1][-1].output == "flow only"

    def test_worker_execute_with_exception(self) -> None:
        """Test worker execution that raises an exception."""
        worker = self.ErrorWorker()
        msg_state = MessageState(status=StatusEnum.INCOMPLETE)

        result = worker.execute(msg_state)

        # Should return original state with INCOMPLETE status
        assert result.status == StatusEnum.INCOMPLETE
        assert result is msg_state  # Should be the same object

    def test_worker_execute_with_validation_error(self) -> None:
        """Test worker execution that returns invalid response."""

        class InvalidWorker(BaseWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {"invalid": "response"}  # Missing required fields

        worker = InvalidWorker()
        msg_state = MessageState(status=StatusEnum.INCOMPLETE)

        result = worker.execute(msg_state)

        # Should return a new state with COMPLETE status
        assert result.status == StatusEnum.COMPLETE
        assert result is not msg_state  # Should be a new object

    def test_worker_execute_with_none_response(self) -> None:
        """Test worker execution with None response."""

        class NoneResponseWorker(BaseWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {
                    "status": StatusEnum.COMPLETE,
                    "response": "",
                    "message_flow": "flow",
                }

        worker = NoneResponseWorker()
        resource_record = ResourceRecord(info={}, intent="test")
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE, trajectory=[[resource_record]]
        )

        result = worker.execute(msg_state)

        assert result.status == StatusEnum.COMPLETE
        assert result.response == ""
        assert result.trajectory[-1][-1].output == "flow"

    def test_worker_execute_with_empty_message_flow(self) -> None:
        """Test worker execution with empty message flow."""

        class EmptyFlowWorker(BaseWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {"status": StatusEnum.COMPLETE, "response": "response only"}

        worker = EmptyFlowWorker()
        resource_record = ResourceRecord(info={}, intent="test")
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE, trajectory=[[resource_record]]
        )

        result = worker.execute(msg_state)

        assert result.status == StatusEnum.COMPLETE
        assert result.response == "response only"
        assert result.trajectory[-1][-1].output == "response only"

    def test_worker_execute_preserves_trajectory(self) -> None:
        """Test that worker execution preserves existing trajectory."""
        worker = self.SimpleWorker()
        resource_record = ResourceRecord(info={"step": "previous"}, intent="test")
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE, trajectory=[[resource_record]]
        )

        result = worker.execute(msg_state)

        assert len(result.trajectory) > 1
        assert result.trajectory[0][0].info == {"step": "previous"}

    def test_worker_execute_with_complex_response(self) -> None:
        """Test worker execution with complex response structure."""

        class ComplexWorker(BaseWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {
                    "status": StatusEnum.COMPLETE,
                    "response": "complex response",
                    "message_flow": "step1, step2",
                }

        worker = ComplexWorker()
        resource_record = ResourceRecord(info={}, intent="test")
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE, trajectory=[[resource_record]]
        )

        result = worker.execute(msg_state)

        assert result.status == StatusEnum.COMPLETE
        assert result.response == "complex response"
        assert result.message_flow == "step1, step2"
        assert result.trajectory[-1][-1].output == "complex response"


class TestWorkerIntegration:
    """Test worker integration scenarios."""

    def test_worker_with_register_decorator(self) -> None:
        """Test worker with register_worker decorator."""

        @register_worker
        class DecoratedWorker(BaseWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {"status": StatusEnum.COMPLETE, "response": "decorated"}

        worker = DecoratedWorker()
        assert worker.name == "DecoratedWorker"

        resource_record = ResourceRecord(info={}, intent="test")
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE, trajectory=[[resource_record]]
        )
        result = worker.execute(msg_state)

        assert result.status == StatusEnum.COMPLETE
        assert result.response == "decorated"

    def test_multiple_workers_same_class(self) -> None:
        """Test multiple instances of the same worker class."""

        class MultiWorker(BaseWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {"status": StatusEnum.COMPLETE, "response": "multi"}

        worker1 = MultiWorker()
        worker2 = MultiWorker()

        resource_record = ResourceRecord(info={}, intent="test")
        msg_state1 = MessageState(
            status=StatusEnum.INCOMPLETE, trajectory=[[resource_record]]
        )
        msg_state2 = MessageState(
            status=StatusEnum.INCOMPLETE, trajectory=[[resource_record]]
        )

        result1 = worker1.execute(msg_state1)
        result2 = worker2.execute(msg_state2)

        assert result1.status == StatusEnum.COMPLETE
        assert result1.response == "multi"
        assert result2.status == StatusEnum.COMPLETE
        assert result2.response == "multi"

    def test_worker_inheritance(self) -> None:
        """Test worker inheritance behavior."""

        class ParentWorker(BaseWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {"status": StatusEnum.COMPLETE, "response": "parent"}

        class ChildWorker(ParentWorker):
            def _execute(
                self, msg_state: MessageState, **kwargs: WorkerKwargs
            ) -> dict[str, Any]:
                return {"status": StatusEnum.COMPLETE, "response": "child"}

        parent = ParentWorker()
        child = ChildWorker()

        resource_record = ResourceRecord(info={}, intent="test")
        msg_state = MessageState(
            status=StatusEnum.INCOMPLETE, trajectory=[[resource_record]]
        )
        parent_result = parent.execute(msg_state)
        child_result = child.execute(msg_state)

        assert parent_result.response == "parent"
        assert child_result.response == "child"

    def test_hitlworker_execute_final_return(self) -> None:
        from arklex.env.workers.hitl_worker import HITLWorker, MessageState

        class DummyHITLWorker(HITLWorker):
            def verify(self, state: MessageState) -> tuple[bool, str]:
                return True, ""

            def _create_action_graph(self) -> object:
                class DummyGraph:
                    def compile(self) -> object:
                        return self

                    def invoke(self, state: MessageState) -> MessageState:
                        return state

                return DummyGraph()

        worker = DummyHITLWorker(name="test", server_ip="1.1.1.1", server_port=1234)
        state = MessageState()
        result = worker._execute(state)
        assert result is state
