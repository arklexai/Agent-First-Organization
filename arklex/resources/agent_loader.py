from typing import Any

from arklex.resources.agent_map import AGENT_MAP
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class AgentLoader:
    @staticmethod
    def init_agents(agents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Initialize agents from configuration.

        Args:
            agents: list of agent configurations

        Returns:
            dictionary mapping agent IDs to their configurations
        """
        agent_registry: dict[str, dict[str, Any]] = {}
        for agent in agents:
            agent_id: str = agent["id"]
            try:
                agent_instance = AGENT_MAP[agent_id]["item_cls"]
                agent_registry[agent_id] = {
                    "agent_instance": agent_instance,
                }
            except Exception as e:
                log_context.error(f"Agent {agent_id} is not registered, error: {e}")
                continue

        return agent_registry
