import traceback
from typing import Any

from arklex.env.agents.agent import BaseAgent, register_agent
from arklex.env.agents.patterns.registry import dispatch_pattern
from arklex.utils.graph_state import MessageState
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
        state: MessageState,
        multiagent_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.workflow = None
        self.initialized = False
        self.multiagent_config: dict[str, Any] = multiagent_config
        self._prepare(state)

    def _prepare(self, state: MessageState) -> None:
        """Prepare and compile the MAS system once."""
        try:
            log_context.info("Preparing MultiAgent...")

            if not self.multiagent_config:
                raise ValueError("MultiAgent config not found in agent.config")

            # Compile MAS system once
            self.workflow = dispatch_pattern(self.multiagent_config)
            self.initialized = True
            log_context.info("MultiAgent system compiled successfully.")

        except Exception:
            log_context.error(
                f"[MultiAgent] Initialization error: {traceback.format_exc()}"
            )

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Execute the pre-compiled MAS workflow."""
        if not self.initialized or self.workflow is None:
            log_context.error("[MultiAgent] System not initialized.")
            return msg_state.model_dump()

        try:
            log_context.info("[MultiAgent] Executing MAS workflow...")
            graph = self.workflow.compile()
            result = graph.invoke(msg_state)
            return dict(result)

        except Exception as e:
            log_context.error(f"[MultiAgent] Execution error: {traceback.format_exc()}")
            msg_state.response = f"[MultiAgent Error] {e}"
            return msg_state.model_dump()
