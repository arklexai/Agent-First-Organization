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
        multi_agent_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.state = state
        self.workflow = None
        self.multi_agent_config: dict[str, Any] = multi_agent_config
        self._load_multi_agent_system()
        log_context.info(
            f"MultiAgent initialized with {self.multi_agent_config.get('pattern')} pattern."
        )

    def _load_multi_agent_system(self) -> None:
        """Loading the Multi-Agent System based on the specified pattern in the config."""
        try:
            log_context.info("Preparing MultiAgent...")

            if not self.multi_agent_config:
                raise ValueError("MultiAgent config not found in agent.config")
            self.multi_agent_config["llm_config"] = self.state.bot_config.llm_config
            self.workflow = dispatch_pattern(self.multi_agent_config)

        except Exception:
            log_context.error(
                f"[MultiAgent] Initialization error: {traceback.format_exc()}"
            )

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        try:
            log_context.info("[MultiAgent] Executing MAS workflow...")
            graph = self.workflow.compile()
            # make this flexible invoke(synchronous) or ainvoke(asynchronous)
            # need to figure out to do with parallelization pattern
            result = graph.invoke(msg_state)
            return dict(result)
        except Exception as e:
            log_context.error(f"[MultiAgent] Execution error: {traceback.format_exc()}")
            msg_state.response = f"[MultiAgent Error] {e}"
            return msg_state.model_dump()
