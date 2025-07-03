from agents import Runner, trace
from langgraph.graph import StateGraph

from arklex.env.agents.utils.agent_loader import build_agents
from arklex.utils.graph_state import MessageState


def build_deterministic_flow(config: dict) -> StateGraph:
    agents = build_agents(config["agents"], config["llm_config"])

    def step_fn(state: MessageState) -> MessageState:
        input_items = state.function_calling_trajectory
        with trace(f"{config['role']}"):
            for agent in agents:
                result = Runner.run_sync(agent, input_items)
                input_items = result.to_input_list()
                state.response = result.final_output
        return state

    graph = StateGraph(MessageState)
    graph.add_node("step", step_fn)
    graph.set_entry_point("step")
    return graph
