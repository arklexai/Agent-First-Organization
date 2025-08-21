from abc import ABC, abstractmethod

from langgraph.graph import StateGraph

from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState


class BasePattern(ABC):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.llm_config = config["llm_config"]

    @abstractmethod
    def step_fn(self, state: OrchestratorState) -> OrchestratorState | None:
        """Define core step logic (sync or async)."""
        pass

    def build(self) -> StateGraph:
        graph = StateGraph(OrchestratorState)
        graph.add_node("step", self.step_fn)
        graph.set_entry_point("step")
        return graph
