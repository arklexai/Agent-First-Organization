"""
Note: This module is currently inactive and reserved for future use.
It may contain experimental or planned features.
It will need to be updated in the future
to comply with the new multi-agent system code structure.

Status:
    - Not in use (as of 2025-07-22)
    - Intended for future feature expansion
"""

from agents import Runner, trace

from arklex.env.agents.patterns.base_pattern import BasePattern
from arklex.env.agents.utils.agent_loader import build_agents
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState


class DeterministicPattern(BasePattern):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.agents = build_agents(config["sub_agents"], self.llm_config)

    def step_fn(self, state: OrchestratorState) -> OrchestratorState:
        input_items = state.function_calling_trajectory

        with trace(f"{self.config['role']}"):
            for agent in self.agents:
                result = Runner.run_sync(agent, input_items)
                input_items = result.to_input_list()
                state.response = result.final_output

        return state
