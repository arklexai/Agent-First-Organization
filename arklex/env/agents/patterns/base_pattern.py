from abc import ABC, abstractmethod

from langgraph.graph import StateGraph

from arklex.orchestrator.entities.msg_state_entities import MessageState


class BasePattern(ABC):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.llm_config = config["llm_config"]

    @abstractmethod
    def step_fn(self, state: MessageState) -> MessageState | None:
        """Define core step logic (sync or async)."""
        pass

    def build(self) -> StateGraph:
        graph = StateGraph(MessageState)
        graph.add_node("step", self.step_fn)
        graph.set_entry_point("step")
        return graph
