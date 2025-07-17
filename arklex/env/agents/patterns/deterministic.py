from agents import Runner, trace

from arklex.env.agents.patterns.base_pattern import BasePattern
from arklex.env.agents.utils.agent_loader import build_agents
from arklex.orchestrator.entities.msg_state_entities import MessageState


class DeterministicPattern(BasePattern):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.agents = build_agents(config["sub_agents"], self.llm_config)

    def step_fn(self, state: MessageState) -> MessageState:
        input_items = state.function_calling_trajectory

        with trace(f"{self.config['role']}"):
            for agent in self.agents:
                result = Runner.run_sync(agent, input_items)
                input_items = result.to_input_list()
                state.response = result.final_output

        return state
