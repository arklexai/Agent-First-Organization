from agents import Agent, Tool, WebSearchTool

TOOL_REGISTRY = {
    "web_search": WebSearchTool,
}


def resolve_tool(name: str) -> Tool:
    tool_cls = TOOL_REGISTRY.get(name)
    return tool_cls() if tool_cls else None


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
