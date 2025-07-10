from dataclasses import dataclass
from typing import Literal

from agents import Agent, ItemHelpers, Runner, trace
from langgraph.graph import StateGraph

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
        name="evaluator",
        instructions=(
            f"You evaluate for the following task: {config['task']} and determine if it is good enough",
            "If it's not good enough, you provide feedback on what needs to be improved."
            "Never give it a pass on the first try.",
        ),
        model=llm_config.model_type_or_path,
        output=EvaluationFeedback,
    )

    async def step_fn(state: MessageState) -> MessageState:
        input_items = state.function_calling_trajectory
        with trace(f"{config['role']}"):
            while True:
                generator_result = await Runner.run(
                    generator_agent,
                    input_items,
                )
                input_items = generator_result.to_input_list()
                latest_output = ItemHelpers.text_message_outputs(
                    generator_result.new_items
                )

                evaluator_result = await Runner.run(evaluator_agent, input_items)
                result: EvaluationFeedback = evaluator_result.final_output

                if result.score == "pass":
                    break

                print("Re-running with feedback")

                input_items.append(
                    {"content": f"Feedback: {result.feedback}", "role": "user"}
                )

        state.response = latest_output
        return state

    graph = StateGraph(MessageState)
    graph.add_node("step", step_fn)
    graph.set_entry_point("step")
    return graph
