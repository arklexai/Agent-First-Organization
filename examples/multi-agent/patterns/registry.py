# from patterns.judge import run_judge_flow
# from patterns.routing import run_routing_flow
# from patterns.parallel import run_parallel_flow
from patterns.agent_as_tools import run_agent_as_tools_flow
from patterns.deterministic import run_deterministic_flow

PATTERN_DISPATCHER = {
    "deterministic": run_deterministic_flow,
    # "judge": run_judge_flow,
    # "routing": run_routing_flow,
    # "parallel": run_parallel_flow,
    "agent_as_tools": run_agent_as_tools_flow,
}


async def dispatch_pattern(config: dict) -> None:
    pattern = config["pattern"]
    run_func = PATTERN_DISPATCHER.get(pattern)

    if not run_func:
        raise ValueError(f"Unsupported pattern: {pattern}")

    await run_func(config)
