import asyncio

from agents import Agent, ItemHelpers, Runner, trace
from langgraph.graph import StateGraph

from arklex.env.agents.utils.agent_loader import build_agents
from arklex.utils.graph_state import LLMConfig, MessageState


def build_parallel_flow(config: dict) -> StateGraph:
    llm_config: LLMConfig = config["llm_config"]
    parallel_agent = build_agents(config["sub_agents"], llm_config)

    selector_agent = Agent(
        name="selector",
        instructions=f"You are the selector. Choose the best response for this task: {config['task']}",
        model=llm_config.model_type_or_path,
    )

    async def step_fn(state: MessageState) -> MessageState:
        input_items = state.function_calling_trajectory
        with trace(f"{config['role']}"):
            # start with two parallelizations,
            # figure out how to make this dynamic
            res_1, res_2 = await asyncio.gather(
                Runner.run(
                    parallel_agent,
                    input_items,
                ),
                Runner.run(
                    parallel_agent,
                    input_items,
                ),
            )

            responses = [
                ItemHelpers.text_message_outputs(res_1.new_items),
                ItemHelpers.text_message_outputs(res_2.new_items),
            ]

            combined_responses = "\n\n".join(responses)
            print(f"\n\nResponses:\n\n{combined_responses}")

            best_responsse = await Runner.run(
                selector_agent,
                f"Input: {input_items}\n\nResponses:\n{responses}",
            )

            state.response = best_responsse.final_output
        breakpoint()
        return state

    graph = StateGraph(MessageState)
    graph.add_node("step", step_fn)
    graph.set_entry_point("step")
    return graph
