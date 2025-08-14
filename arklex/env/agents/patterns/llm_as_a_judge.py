"""
Note: This module is currently inactive and reserved for future use.
It may contain experimental or planned features.
It will need to be updated in the future
to comply with the new multi-agent system code structure.

Status:
    - Not in use (as of 2025-07-22)
    - Intended for future feature expansion
"""

from dataclasses import dataclass
from typing import Literal

from agents import Agent, ItemHelpers, Runner, trace
from rich.console import Console
from rich.panel import Panel

from arklex.env.agents.patterns.base_pattern import BasePattern
from arklex.env.agents.utils.agent_loader import build_agents
from arklex.orchestrator.entities.orchestrator_state_entities import OrchestratorState

console = Console()


@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


class LLMAsJudgePattern(BasePattern):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.generator_agent = build_agents(config["sub_agents"], self.llm_config)
        self.evaluator_agent = Agent(
            name="EvaluatorAgent",
            instructions=(
                f"You evaluate for the following task: {config['task']} and determine if it is good enough. "
                "If it's not good enough, you provide feedback on what needs to be improved. "
                "Never give it a pass on the first try."
            ),
            model=self.llm_config.model_type_or_path,
            output_type=EvaluationFeedback,
        )
        self.max_attempts = config.get("max_attempts", 3)

    async def step_fn(self, state: OrchestratorState) -> OrchestratorState:
        input_items = state.function_calling_trajectory
        attempt = 0
        latest_output = ""
        evaluation_result = None

        with trace(f"{self.config['role']}"):
            while attempt < self.max_attempts:
                attempt += 1

                generator_result = await Runner.run(self.generator_agent, input_items)
                input_items = generator_result.to_input_list()

                latest_output = ItemHelpers.text_message_outputs(
                    generator_result.new_items
                )

                evaluation_result = await Runner.run(self.evaluator_agent, input_items)
                feedback: EvaluationFeedback = evaluation_result.final_output

                self.print_iteration(
                    attempt, latest_output, feedback.feedback, feedback.score == "pass"
                )

                if feedback.score.lower() == "pass":
                    break

                input_items.append(
                    {"content": f"Feedback: {feedback.feedback}", "role": "user"}
                )

        if attempt == self.max_attempts and feedback.score.lower() != "pass":
            state.response = (
                f"Judge did not approve output after {self.max_attempts} attempts."
            )
        else:
            state.response = latest_output

        return state

    def print_iteration(
        self, attempt: int, output: str, feedback: str, passed: bool
    ) -> None:
        console.rule(f"[bold cyan]Attempt {attempt}")
        console.print(
            Panel.fit(
                output, title="[bold green]Generator Output", border_style="green"
            )
        )
        console.print(
            Panel.fit(
                feedback, title="[bold yellow]Evaluator Feedback", border_style="yellow"
            )
        )

        result_text = "✅ PASSED" if passed else "❌ NEEDS IMPROVEMENT"
        color = "green" if passed else "red"
        console.print(f"[bold {color}]{result_text}\n")
