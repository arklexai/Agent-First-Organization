from dataclasses import dataclass
from typing import Literal

from agents import Agent, ItemHelpers, Runner, trace
from langgraph.graph import StateGraph
from rich.console import Console
from rich.panel import Panel

from arklex.env.agents.utils.agent_loader import build_agents
from arklex.utils.graph_state import LLMConfig, MessageState


@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


def build_llm_as_a_judge_flow(config: dict) -> StateGraph:
    llm_config: LLMConfig = config["llm_config"]
    generator_agent = build_agents(config["sub_agents"], llm_config)
    evaluator_agent = Agent(
        name="EvaluatorAgent",
        instructions=(
            f"You evaluate for the following task: {config['task']} and determine if it is good enough. "
            "If it's not good enough, you provide feedback on what needs to be improved. "
            "Never give it a pass on the first try."
        ),
        model=llm_config.model_type_or_path,
        output_type=EvaluationFeedback,
    )

    async def step_fn(state: MessageState) -> MessageState:
        input_items = state.function_calling_trajectory
        max_attempts = config.get("max_attempts", 3)
        attempt = 0
        latest_output = ""

        with trace(f"{config['role']}"):
            while attempt < max_attempts:
                attempt += 1
                generator_result = await Runner.run(generator_agent, input_items)
                input_items = generator_result.to_input_list()
                latest_output = ItemHelpers.text_message_outputs(
                    generator_result.new_items
                )

                evaluator_result = await Runner.run(evaluator_agent, input_items)
                result: EvaluationFeedback = evaluator_result.final_output

                print_iteration(
                    attempt, latest_output, result.feedback, result.score == "pass"
                )

                if result.score.lower() == "pass":
                    break

                input_items.append(
                    {"content": f"Feedback: {result.feedback}", "role": "user"}
                )

        if attempt == max_attempts and result.score.lower() != "pass":
            state.response = (
                f"Judge did not approve output after {max_attempts} attempts."
            )
        else:
            state.response = latest_output
        return state

    graph = StateGraph(MessageState)
    graph.add_node("step", step_fn)
    graph.set_entry_point("step")
    return graph


console = Console()


def print_iteration(
    attempt: int, generator_output: str, feedback: str, passed: bool
) -> None:
    console.rule(f"[bold cyan]Attempt {attempt}")

    console.print(
        Panel.fit(
            generator_output, title="[bold green]Generator Output", border_style="green"
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
