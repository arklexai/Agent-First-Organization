from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar
from arklex.utils.graph_state import MessageState, StatusEnum
import logging
import traceback

logger = logging.getLogger(__name__)

T = TypeVar("T")


def register_worker(cls: Type[T]) -> Type[T]:
    """Decorator to register a worker."""
    cls.name = cls.__name__  # Automatically set name to the class name
    return cls


class BaseWorker(ABC):
    description: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    @abstractmethod
    def _execute(self, msg_state: MessageState, **kwargs: Any) -> Dict[str, Any]:
        pass

    def execute(self, msg_state: MessageState, **kwargs: Any) -> MessageState:
        try:
            response_return: Dict[str, Any] = self._execute(msg_state, **kwargs)
            response_state: MessageState = MessageState.model_validate(response_return)
            response_state.trajectory[-1][-1].output = (
                response_state.response
                if response_state.response
                else response_state.message_flow
            )
            if response_state.status == StatusEnum.INCOMPLETE:
                response_state.status = StatusEnum.COMPLETE
            return response_state
        except Exception as e:
            logger.error(traceback.format_exc())
            msg_state.status = StatusEnum.INCOMPLETE
            return msg_state
