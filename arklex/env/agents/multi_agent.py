import inspect
import traceback
from typing import Any

from arklex.env.agents.agent import BaseAgent, register_agent
from arklex.env.agents.patterns.registry import dispatch_pattern
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


@register_agent
class MultiAgent(BaseAgent):
    description: str = "Multi-agent system using configured sub-agents and patterns."

    def __init__(
        self,
        successors: list,
        predecessors: list,
        tools: list,
        state: OrchestratorState,
        multi_agent_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.state = state
        self.workflow = None
        self.multi_agent_config = multi_agent_config or {}
        self._load_multi_agent_system()
        log_context.info(
            f"MultiAgent initialized with {self.multi_agent_config.get('node_specific_data')} pattern."
        )

    def init_agent_data(
        self, orch_state: OrchestratorState, node_specific_data: dict[str, Any]
    ) -> None:
        pass

    def is_async(self) -> bool:
        """Check if this multi-agent instance should be run asynchronously."""
        node_data = self.multi_agent_config.get("node_specific_data") or {}
        return node_data.get("is_async", False)

    def _load_multi_agent_system(self) -> None:
        try:
            log_context.info("Preparing MultiAgent...")
            if not self.multi_agent_config:
                raise ValueError("MultiAgent config not found")
            self.multi_agent_config["llm_config"] = self.state.bot_config.llm_config
            self.workflow = dispatch_pattern(self.multi_agent_config)
        except Exception:
            log_context.error(
                f"[MultiAgent] Initialization error: {traceback.format_exc()}"
            )
            raise

    def _execute(self, msg_state: OrchestratorState, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Synchronous execution if config does not enable async."""
        try:
            log_context.info("[MultiAgent] Executing MAS workflow (sync)...")
            graph = self.workflow.compile()
            result = graph.invoke(msg_state)
            return dict(result)
        except Exception as e:
            log_context.error(
                f"[MultiAgent] Sync execution error: {traceback.format_exc()}"
            )
            msg_state.response = f"[MultiAgent Error] {e}"
            return msg_state.model_dump()

    async def _async_execute(
        self,
        msg_state: OrchestratorState,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """Asynchronous execution if enabled in config."""
        try:
            log_context.info("[MultiAgent] Executing MAS workflow (async)...")
            graph = self.workflow.compile()

            if hasattr(graph, "ainvoke") and inspect.iscoroutinefunction(graph.ainvoke):
                result = await graph.ainvoke(msg_state)
            else:
                log_context.warning(
                    "[MultiAgent] Graph does not support ainvoke, falling back to sync."
                )
                result = graph.invoke(msg_state)

            return dict(result)
        except Exception as e:
            log_context.error(
                f"[MultiAgent] Async execution error: {traceback.format_exc()}"
            )
            msg_state.response = f"[MultiAgent Error] {e}"
            return msg_state.model_dump()
