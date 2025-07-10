from langgraph.graph import StateGraph

from arklex.env.agents.patterns.agents_as_tools import build_agents_as_tools_flow
from arklex.env.agents.patterns.deterministic import build_deterministic_flow
from arklex.env.agents.patterns.llm_as_a_judge import build_llm_as_a_judge_flow
from arklex.env.agents.patterns.parallelization import build_parallel_flow

PATTERN_DISPATCHER = {
    "deterministic": build_deterministic_flow,
    "agents_as_tools": build_agents_as_tools_flow,
    "parallel": build_parallel_flow,
    "llm_as_a_judge": build_llm_as_a_judge_flow,
}


def dispatch_pattern(config: dict) -> StateGraph:
    pattern = config["pattern"]
    run_func = PATTERN_DISPATCHER.get(pattern)

    if not run_func:
        raise ValueError(f"Unsupported pattern: {pattern}")

    graph = run_func(config)
    return graph
