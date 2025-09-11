from agents import Agent, Tool

from arklex.env.agents.utils.tool_resolver import resolve_tools_for_agent
from arklex.utils.llm_config import LLMConfig


def build_agents(agent_configs: list[dict], llm_config: LLMConfig) -> list[Agent]:
    agents = []
    for config in agent_configs:
        tools = resolve_tools_for_agent(config.get("tools", []))
        agent = Agent(
            name=config["name"],
            instructions=config["task"],
            tools=tools,
            model=llm_config.model_type_or_path,
        )
        agents.append(agent)
    return agents


def build_tool_wrapped_agents(
    agent_configs: list[dict], llm_config: LLMConfig
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
