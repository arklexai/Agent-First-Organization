from agents import Agent, Runner, trace

from arklex.env.agents.patterns.base_pattern import BasePattern
from arklex.env.agents.utils.agent_loader import build_tool_wrapped_agents
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState


class AgentsAsToolsPattern(BasePattern):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.tool_agents, self.tool_wrappers = build_tool_wrapped_agents(
            config.get("sub_agents", []), self.llm_config
        )

        self.orchestrator_agent = Agent(
            name="OrchestratorAgent",
            instructions=f"You are the orchestrator. Use the tools to complete this task: {config.get('task')}. Do NOT answer on your own.",
            tools=self.tool_wrappers,
            model=self.llm_config.model_type_or_path,
        )

    async def step_fn(self, state: OrchestratorState) -> OrchestratorState:
        input_items = state.function_calling_trajectory
        with trace(f"{self.config.get('type', 'MultiAgent')}"):
            result = await Runner.run(self.orchestrator_agent, input_items)
            state.response = result.final_output
        return state
