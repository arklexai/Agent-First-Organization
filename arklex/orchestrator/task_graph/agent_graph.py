import collections
import re
import threading
from typing import Any

from agents import (
    Agent,
    handoff,
)
from agents.realtime import RealtimeAgent, realtime_handoff
from jinja2 import Template

from arklex.orchestrator.task_graph.base import GraphBase
from arklex.resources.agents.base.entities import PromptVariable
from arklex.resources.agents.llm_based_agent.guardrail_agent import (
    GuardrailAgentData,
    GuardrailAgentResult,
)
from arklex.resources.agents.llm_based_agent.openai_agent import (
    OpenAIAgentData,
)
from arklex.resources.agents.realtime_voice_agent.openai_realtime_agent import (
    OpenAIRealtimeAgent,
    OpenAIRealtimeAgentData,
)
from arklex.resources.resource_loader import ResourceLoader
from arklex.resources.resource_types import (
    AgentItem,
    ResourceType,
    ToolItem,
)
from arklex.resources.tools.tools import Tool
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class AgentGraph(GraphBase):
    """
    AgentGraph is a task graph that contains agents and tools that the agent can use.

    Methods:
        create_graph(): Creates the graph structure
    """

    def __init__(self, name: str, graph_config: dict[str, Any]) -> None:
        self.agents: dict[str, Agent] = {}
        self.resources = {}
        self.prompt_variables: list[PromptVariable] = []
        self.start_message: str = ""
        self.start_agent_name: str = ""
        self.enabled = False
        self.agents_safety_response: dict[str, str] = {}
        self.agents_input_guardrails: dict[str, list[Agent]] = collections.defaultdict(
            list
        )
        self.agents_output_guardrails: dict[str, list[Agent]] = collections.defaultdict(
            list
        )
        super().__init__(name, graph_config)

    def create_graph(self) -> None:
        nodes: list[dict[str, Any]] = self.graph_config["nodes"]
        edges: list[tuple[str, str, dict[str, Any]]] = self.graph_config["edges"]
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

        all_resources: list[dict[str, Any]] = self.graph_config["tools"]
        resource_map: dict[str, dict[str, Any]] = {}
        for resource in all_resources:
            resource_map[resource["id"]] = resource

        resource_loader = ResourceLoader()
        agent_handovers: dict[str, list[str]] = collections.defaultdict(list)
        input_guardrail_map: dict[str, list[str]] = collections.defaultdict(list)
        output_guardrail_map: dict[str, list[str]] = collections.defaultdict(list)
        agent_data_map: dict[str, OpenAIAgentData] | None = {}
        for node in self.graph.nodes.data():
            resource = node[1].get("resource", {})
            if resource["id"] == AgentItem.OPENAI_AGENT:
                node_specific_data = node[1].get("data", {})
                if node[1].get("attribute", {}).get("start", False):
                    self.start_agent_name = node_specific_data["name"]
                    self.start_message = node_specific_data.get(
                        "agent_start_message", ""
                    )
                # process successors and predecessors to get resources
                available_tools = []
                available_nodes = []
                for node_id in self.graph.successors(node[0]):
                    successor_node = self.graph.nodes[node_id]
                    if successor_node["resource"]["id"] == AgentItem.OPENAI_AGENT:
                        agent_handovers[node_specific_data["name"]].append(
                            successor_node["data"]["name"]
                        )
                    elif (
                        successor_node.get("attribute", {}).get("type", "")
                        == ResourceType.TOOL
                    ):
                        available_tools.append(
                            resource_map[successor_node["resource"]["id"]]
                        )
                        available_nodes.append([node_id, successor_node])
                    elif (
                        successor_node.get("attribute", {}).get("type", "")
                        == ResourceType.AGENT
                    ):
                        if (
                            successor_node["resource"]["id"]
                            == AgentItem.INPUT_GUARDRAIL
                        ):
                            input_guardrail_map[node_specific_data["name"]].append(
                                successor_node["data"]["name"]
                            )
                        elif (
                            successor_node["resource"]["id"]
                            == AgentItem.OUTPUT_GUARDRAIL
                        ):
                            output_guardrail_map[node_specific_data["name"]].append(
                                successor_node["data"]["name"]
                            )
                for node_id in self.graph.predecessors(node[0]):
                    predecessor_node = self.graph.nodes[node_id]
                    if (
                        predecessor_node.get("attribute", {}).get("type", "")
                        == ResourceType.TOOL
                    ):
                        available_tools.append(
                            resource_map[predecessor_node["resource"]["id"]]
                        )
                        available_nodes.append([node_id, predecessor_node])
                    elif (
                        predecessor_node.get("attribute", {}).get("type", "")
                        == ResourceType.AGENT
                    ):
                        if (
                            predecessor_node["resource"]["id"]
                            == AgentItem.INPUT_GUARDRAIL
                        ):
                            input_guardrail_map[node_specific_data["name"]].append(
                                predecessor_node["data"]["name"]
                            )
                        elif (
                            predecessor_node["resource"]["id"]
                            == AgentItem.OUTPUT_GUARDRAIL
                        ):
                            output_guardrail_map[node_specific_data["name"]].append(
                                predecessor_node["data"]["name"]
                            )
                tool_registry = resource_loader.init_tools(
                    available_tools, available_nodes
                )
                # load resource instances
                agents_tools = []
                for tool_id in tool_registry:
                    tool_registry[tool_id]["tool_instance"].name = re.sub(
                        r"[^a-zA-Z0-9]", "_", tool_id
                    )
                    agents_tools.append(
                        tool_registry[tool_id][
                            "tool_instance"
                        ].to_openai_agents_function_tool()
                    )
                self.resources.update(tool_registry)
                agent_data = OpenAIAgentData(**node_specific_data)
                prompt = agent_data.prompt
                if agent_data.prompt_variables:
                    self.prompt_variables.extend(agent_data.prompt_variables)
                    template = Template(prompt)
                    # convert prompt_variables to a dict
                    prompt_variables_dict = {
                        pv.name: pv.value for pv in self.prompt_variables
                    }
                    prompt = template.render(prompt_variables_dict)
                if agent_data.safety_response:
                    self.agents_safety_response[agent_data.name] = (
                        agent_data.safety_response
                    )
                else:
                    self.agents_safety_response[agent_data.name] = ""
                agent_data_map[agent_data.name] = agent_data
                self.agents[agent_data.name] = Agent(
                    name=agent_data.name,
                    instructions=prompt,
                    tools=agents_tools,
                    handoff_description=agent_data.handoff_description,
                )
            elif (
                resource["id"] == AgentItem.INPUT_GUARDRAIL
                or resource["id"] == AgentItem.OUTPUT_GUARDRAIL
            ):
                node_specific_data = node[1].get("data", {})
                agent_data = GuardrailAgentData(**node_specific_data)
                self.agents[agent_data.name] = Agent(
                    name=agent_data.name,
                    instructions=agent_data.instructions,
                    output_type=GuardrailAgentResult,
                )

        if len(self.agents) == 0:
            log_context.info("No agents found in the graph")
            return
        self.enabled = True
        if not self.start_agent_name:
            self.start_agent_name = list(self.agents.keys())[0]

        for agent_name, handover_agents in agent_handovers.items():
            # Set handoffs for OpenAIAgent
            self.agents[agent_name].handoffs = [
                handoff(self.agents[handover_agent])
                for handover_agent in handover_agents
            ]
        for agent_name, input_guardrails in input_guardrail_map.items():
            self.agents_input_guardrails[agent_name].extend(
                [self.agents[input_guardrail] for input_guardrail in input_guardrails]
            )

        for agent_name, output_guardrails in output_guardrail_map.items():
            self.agents_output_guardrails[agent_name].extend(
                [
                    self.agents[output_guardrail]
                    for output_guardrail in output_guardrails
                ]
            )

    def configure_params(self) -> dict[str, Any]:
        return {
            "agents": self.agents,
            "start_agent_name": self.start_agent_name,
            "start_message": self.start_message,
            "agents_input_guardrails": self.agents_input_guardrails,
            "agents_output_guardrails": self.agents_output_guardrails,
            "agents_safety_response": self.agents_safety_response,
        }


class RealtimeAgentGraph(GraphBase):
    """
    RealtimeAgentGraph is a task graph that contains realtime agents and tools that the agent can use.

    Methods:
        create_graph(): Creates the graph structure
    """

    def __init__(self, name: str, graph_config: dict[str, Any]) -> None:
        self.agents: dict[str, RealtimeAgent] = {}
        self.resources = {}
        self.start_agent: OpenAIRealtimeAgent | None = None
        self.prompt_variables: list[PromptVariable] = []
        super().__init__(name, graph_config)

    def create_graph(self) -> None:
        nodes: list[dict[str, Any]] = self.graph_config["nodes"]
        edges: list[tuple[str, str, dict[str, Any]]] = self.graph_config["edges"]
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

        all_resources: list[dict[str, Any]] = self.graph_config["tools"]
        resource_map: dict[str, dict[str, Any]] = {}
        for resource in all_resources:
            resource_map[resource["id"]] = resource

        resource_loader = ResourceLoader()
        agent_handovers: dict[str, list[str]] = collections.defaultdict(list)
        start_agent_data: dict[str, Any] | None = None
        start_agent_name: str | None = None
        voicemail_tool: Tool | None = None
        response_played_event = threading.Event()
        agent_data_map: dict[str, OpenAIRealtimeAgentData] | None = {}
        for node in self.graph.nodes.data():
            resource = node[1].get("resource", {})
            if resource["id"] == AgentItem.OPENAI_REALTIME_VOICE_AGENT:
                node_specific_data = node[1].get("data", {})
                if node_specific_data.get("start_agent", False):
                    start_agent_name = node_specific_data["name"]
                # process successors and predecessors to get resources
                available_tools = []
                available_nodes = []
                for node_id in self.graph.successors(node[0]):
                    successor_node = self.graph.nodes[node_id]
                    if (
                        successor_node["resource"]["id"]
                        == AgentItem.OPENAI_REALTIME_VOICE_AGENT
                    ):
                        agent_handovers[node_specific_data["name"]].append(
                            successor_node["data"]["name"]
                        )
                    elif (
                        successor_node.get("attribute", {}).get("type", "")
                        == ResourceType.TOOL
                    ):
                        available_tools.append(
                            resource_map[successor_node["resource"]["id"]]
                        )
                        available_nodes.append([node_id, successor_node])
                for node_id in self.graph.predecessors(node[0]):
                    predecessor_node = self.graph.nodes[node_id]
                    if (
                        predecessor_node.get("attribute", {}).get("type", "")
                        == ResourceType.TOOL
                    ):
                        available_tools.append(
                            resource_map[predecessor_node["resource"]["id"]]
                        )
                        available_nodes.append([node_id, predecessor_node])
                tool_registry = resource_loader.init_tools(
                    available_tools, available_nodes
                )
                # load resource instances
                agents_tools = []
                for tool_id in tool_registry:
                    tool_registry[tool_id]["tool_instance"].name = re.sub(
                        r"[^a-zA-Z0-9]", "_", tool_id
                    )
                    tool_registry[tool_id]["tool_instance"].runtime_args.update(
                        {"response_played_event": response_played_event}
                    )
                    if tool_id == ToolItem.TWILIO_CALL_VOICEMAIL:
                        voicemail_tool = tool_registry[tool_id]["tool_instance"]
                    else:
                        agents_tools.append(
                            tool_registry[tool_id][
                                "tool_instance"
                            ].to_openai_agents_function_tool()
                        )
                self.resources.update(tool_registry)
                agent_data = OpenAIRealtimeAgentData(**node_specific_data)
                prompt = agent_data.prompt
                if agent_data.prompt_variables:
                    self.prompt_variables.extend(agent_data.prompt_variables)
                    template = Template(prompt)
                    # convert prompt_variables to a dict
                    prompt_variables_dict = {
                        pv.name: pv.value for pv in self.prompt_variables
                    }
                    prompt = template.render(prompt_variables_dict)
                agent_data_map[agent_data.name] = agent_data
                self.agents[agent_data.name] = RealtimeAgent(
                    name=agent_data.name,
                    instructions=prompt,
                    tools=agents_tools,
                    handoff_description=agent_data.handoff_description,
                )

        if len(self.agents) == 0:
            log_context.info("No realtimeagents found in the graph")
            return
        for agent_name, handover_agents in agent_handovers.items():
            # Set handoffs for RealtimeAgent
            self.agents[agent_name].handoffs = [
                realtime_handoff(self.agents[handover_agent])
                for handover_agent in handover_agents
            ]
        if start_agent_name is None:
            start_agent_name = list(self.agents.keys())[0]
        start_agent_data: OpenAIRealtimeAgentData = agent_data_map[start_agent_name]
        log_context.info(f"agent handovers: {agent_handovers}")
        log_context.info(
            f"start agent: {start_agent_name}, handovers: {self.agents[start_agent_name].handoffs}"
        )
        self.start_agent = OpenAIRealtimeAgent(
            realtime_agent=self.agents[start_agent_name],
            voice=start_agent_data.voice,
            transcription_language=start_agent_data.transcription_language,
            speed=start_agent_data.speed,
            turn_detection=start_agent_data.turn_detection,
            voicemail_tool=voicemail_tool,
        )
        self.start_agent.response_played = response_played_event
