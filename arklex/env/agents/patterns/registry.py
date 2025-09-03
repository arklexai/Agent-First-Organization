from langgraph.graph import StateGraph

from arklex.env.agents.patterns.agents_as_tools import AgentsAsToolsPattern
from arklex.env.agents.patterns.base_pattern import BasePattern
from arklex.env.agents.patterns.deterministic import DeterministicPattern
from arklex.env.agents.patterns.llm_as_a_judge import LLMAsJudgePattern
from arklex.env.agents.patterns.parallelization import ParallelPattern

PATTERN_DISPATCHER: dict[str, type[BasePattern]] = {
    "deterministic": DeterministicPattern,
    "agents_as_tools": AgentsAsToolsPattern,
    "parallel": ParallelPattern,
    "llm_as_a_judge": LLMAsJudgePattern,
}


def dispatch_pattern(config: dict) -> StateGraph:
    node_specific_config = config.get("node_specific_data")
    pattern_name = node_specific_config.get("type")
    pattern_cls = PATTERN_DISPATCHER.get(pattern_name)

    if not pattern_cls:
        raise ValueError(f"Unsupported pattern: {pattern_name}")

    return pattern_cls(config).build()
