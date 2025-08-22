"""
Note: This module is currently inactive and reserved for future use.
It may contain experimental or planned features.
It will need to be updated in the future
to comply with the new multi-agent system code structure.

Status:
    - Not in use (as of 2025-07-22)
    - Intended for future feature expansion
"""

import asyncio

from agents import Agent, ItemHelpers, Runner, trace
from rich.console import Console
from rich.panel import Panel

from arklex.env.agents.patterns.base_pattern import BasePattern
from arklex.env.agents.utils.agent_loader import build_agents
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState

console = Console()


class ParallelPattern(BasePattern):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.parallel_agents = build_agents(config["sub_agents"], self.llm_config)
        self.selector_agent = Agent(
            name="SelectorAgent",
            instructions=(
                f"You are the selector. Choose the best response for this task: {config['task']}"
            ),
            model=self.llm_config.model_type_or_path,
        )

    async def step_fn(self, state: OrchestratorState) -> OrchestratorState:
        input_items = state.function_calling_trajectory

        with trace(f"{self.config['role']}"):
            # Run both agents in parallel
            res_1, res_2 = await asyncio.gather(
                Runner.run(self.parallel_agents, input_items),
                Runner.run(self.parallel_agents, input_items),
            )

            responses = [
                ItemHelpers.text_message_outputs(res_1.new_items),
                ItemHelpers.text_message_outputs(res_2.new_items),
            ]

            # Pretty-print responses from both agents
            console.print("\n[b cyan]Parallel Agent Responses:[/b cyan]\n")
            for i, r in enumerate(responses, 1):
                console.print(
                    Panel(r, title=f"Agent #{i}", subtitle="Output", expand=False)
                )

            selector_input = f"Input: {input_items}\n\nResponses:\n{responses}"
            best_response = await Runner.run(self.selector_agent, selector_input)
            state.response = best_response.final_output

        return state
