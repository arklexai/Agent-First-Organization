from abc import ABC, abstractmethod
from arklex.utils.graph_state import MessageState, StatusEnum
import logging
import traceback

logger = logging.getLogger(__name__)

def register_worker(cls):
    """Decorator to register a worker."""
    cls.name = cls.__name__  # Automatically set name to the class name
    return cls


class BaseWorker(ABC):
    
    description = None

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return f"{self.__class__.__name__}"
    
    @abstractmethod
    def _execute(self, msg_state: MessageState, **kwargs):
        pass

    def execute(self, msg_state: MessageState, **kwargs):
        try:
            response_return = self._execute(msg_state, **kwargs)
            response_state = MessageState.model_validate(response_return)
            response_state.trajectory[-1][-1].output = response_state.response if response_state.response else response_state.message_flow
            if response_state.status == StatusEnum.INCOMPLETE:
                response_state.status = StatusEnum.COMPLETE
            return response_state
        except Exception as e:
            logger.error(traceback.format_exc())
            msg_state.status = StatusEnum.INCOMPLETE
            return msg_state