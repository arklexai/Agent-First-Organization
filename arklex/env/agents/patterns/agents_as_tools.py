from agents import Agent, Runner, trace
from langgraph.graph import StateGraph

from arklex.env.agents.utils.agent_loader import build_tool_wrapped_agents
from arklex.utils.graph_state import LLMConfig, MessageState


def build_agents_as_tools_flow(config: dict) -> StateGraph:
    llm_config: LLMConfig = config["llm_config"]
    tool_agents, tool_wrappers = build_tool_wrapped_agents(
        config["sub_agents"], llm_config
    )

    orchestrator_agent = Agent(
        name="OrchestratorAgent",
        instructions=f"You are the orchestrator. Use the tools to complete this task: {config['task']}. Do NOT answer on your own.",
        tools=tool_wrappers,
        model=llm_config.model_type_or_path,
    )

    def step_fn(state: MessageState) -> MessageState:
        input_items = state.function_calling_trajectory
        with trace(f"{config['role']}"):
            orchestrator_result = Runner.run_sync(orchestrator_agent, input_items)
            state.response = orchestrator_result.final_output
        return state

    graph = StateGraph(MessageState)
    graph.add_node("step", step_fn)
    graph.set_entry_point("step")
    return graph
