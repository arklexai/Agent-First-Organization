from agents import Agent, Tool
from common.tool_resolver import resolve_tool


def build_agents(agent_configs: list[dict], llm_config: dict) -> list[Agent]:
    agents = []
    for cfg in agent_configs:
        tools = [resolve_tool(t) for t in cfg.get("tools", []) if resolve_tool(t)]
        agent = Agent(
            name=cfg["name"],
            instructions=cfg["instructions"],
            tools=tools,
            model=llm_config.get("model", "gpt-4o-mini"),
        )
        agents.append(agent)
    return agents


def build_tool_wrapped_agents(
    agent_configs: list[dict], llm_config: dict
) -> tuple[list[Agent], list[Tool]]:
    """
    Builds agents using `build_agents`, then wraps each as a tool.

    Returns:
        (agents, tool_wrappers)
    """
    agents = build_agents(agent_configs, llm_config)
    tool_wrappers = [
        agent.as_tool(
            tool_name=agent.name,
            tool_description=cfg.get("description", f"Tool for {cfg['name']}"),
        )
        for agent, cfg in zip(agents, agent_configs, strict=False)
    ]
    return agents, tool_wrappers
