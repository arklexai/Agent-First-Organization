import json
from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph

from arklex.env.agents.agent import BaseAgent, register_agent
from arklex.env.prompts import load_prompts
from arklex.env.tools.tools import register_tool
from arklex.env.tools.utils import trace
from arklex.utils.graph_state import MessageState, StatusEnum
from arklex.utils.logging_utils import LogContext
from arklex.utils.model_provider_config import PROVIDER_MAP

log_context = LogContext(__name__)


@register_tool("Ends the conversation with a thank you message.", isResponse=True)
def end_conversation(state: MessageState) -> str:
    """Ends the conversation with a thank you message. This is a default tool that is used to end the conversation and added to the tool map of the OpenAIAgent.

    Args:
        state (MessageState): The state of the conversation.

    Returns:
        MessageState: The updated state of the conversation.
    """
    log_context.info("Ending the conversation.")
    state.status = StatusEnum.COMPLETE
    llm = PROVIDER_MAP.get(state.bot_config.llm_config.llm_provider, ChatOpenAI)(
        model=state.bot_config.llm_config.model_type_or_path
    )
    try:
        return llm.invoke(
            [
                SystemMessage(
                    content="Ends the conversation with a thank you and goodbye message."
                ).model_dump(),
            ]
        ).content
    except Exception as e:
        log_context.error(f"Error when ending conversation: {e}")
        return "I hope I was able to help you today. Goodbye!"


@register_agent
class OpenAIAgent(BaseAgent):
    description: str = "General-purpose Arklex agent for chat or voice."

    def __init__(
        self, successors: list, predecessors: list, tools: list, state: MessageState
    ) -> None:
        super().__init__()
        self.action_graph: StateGraph = self._create_action_graph()
        self.llm: BaseChatModel | None = None
        self.available_tools: dict[str, dict[str, Any]] = {}
        self.tool_map = {}
        self.tool_defs = []
        self.tool_args: dict[str, Any] = {}

        self._load_tools(successors=successors, predecessors=predecessors, tools=tools)
        self._configure_tools()

        log_context.info(f"OpenAIAgent initialized with {len(self.tool_defs)} tools.")

    def generate(self, state: MessageState) -> MessageState:
        log_context.info("\nGenerating response using the agent.")

        if state.status == StatusEnum.INCOMPLETE:
            if not self.prompt:
                orchestrator_message = state.orchestrator_message
                orch_msg_content: str = (
                    "None"
                    if not orchestrator_message.message
                    else orchestrator_message.message
                )
                prompts: dict[str, str] = load_prompts(state.bot_config)
                prompt: PromptTemplate = PromptTemplate.from_template(
                    prompts["function_calling_agent_prompt"]
                )
                input_prompt = prompt.invoke(
                    {
                        "sys_instruct": state.sys_instruct,
                        "message": orch_msg_content,
                    }
                )
                input_prompt = input_prompt.text
            else:
                input_prompt = self.prompt

            if not any(
                message.get("content") == input_prompt
                for message in state.function_calling_trajectory
            ):
                log_context.info(
                    "Adding input prompt to the function calling trajectory."
                )
                state.function_calling_trajectory.append(
                    SystemMessage(content=input_prompt).model_dump()
                )

        log_context.info(f"\nagent messages: {state.function_calling_trajectory}")

        final_chain = self.llm
        ai_message: AIMessage = final_chain.invoke(state.function_calling_trajectory)

        log_context.info(f"Generated answer: {ai_message}")

        if ai_message.tool_calls:
            log_context.info("Processing tool calls.")
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call.get("name")
                if tool_name in self.tool_map:
                    state.function_calling_trajectory.append(
                        AIMessage(
                            content=f"Calling tool: {tool_name}",
                            tool_calls=[tool_call],
                        ).model_dump()
                    )
                    tool_response = self.tool_map[tool_name](
                        state=state,
                        **tool_call.get("args"),
                        **self.tool_args.get(tool_name, {}),
                    )
                    state.function_calling_trajectory.append(
                        ToolMessage(
                            name=tool_name,
                            content=json.dumps(tool_response),
                            tool_call_id=tool_call.get("id"),
                        ).model_dump()
                    )
                    ai_message = final_chain.invoke(state.function_calling_trajectory)
                else:
                    log_context.warning(f"Tool {tool_name} not found in tool map.")
        state.message_flow = ""
        state.response = ai_message.content
        state = trace(input=ai_message.content, state=state)
        return state

    def _create_action_graph(self) -> StateGraph:
        workflow = StateGraph(MessageState)
        workflow.add_node("generate", self.generate)
        workflow.add_edge(START, "generate")

        return workflow

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        self.llm = PROVIDER_MAP.get(
            msg_state.bot_config.llm_config.llm_provider, ChatOpenAI
        )(model=msg_state.bot_config.llm_config.model_type_or_path)
        self.llm = self.llm.bind_tools(self.tool_defs)
        self.prompt: str = kwargs.get("prompt", "")
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result

    def _load_tools(self, successors: list, predecessors: list, tools: list) -> None:
        """
        Load tools for the agent.
        This method is called during the initialization of the agent.
        """
        for node in successors + predecessors:
            if node.resource_id in tools:
                self.available_tools[node.resource_id] = tools[node.resource_id]

    def _configure_tools(self) -> None:
        """
        Configure tools for the agent.
        This method is called during the initialization of the agent.
        """
        for _tool_id, tool in self.available_tools.items():
            tool_object = tool["execute"]()
            tool_def = tool_object.to_openai_tool_def_v2()
            self.tool_defs.append(tool_object.to_openai_tool_def_v2())
            self.tool_map[tool_def["function"]["name"]] = tool_object.func
            self.tool_args[tool_def["function"]["name"]] = tool["fixed_args"]

        end_conversation_tool = end_conversation()
        end_conversation_tool_def = end_conversation_tool.to_openai_tool_def_v2()
        end_conversation_tool_def["function"]["name"] = "end_conversation"
        self.tool_defs.append(end_conversation_tool_def)
        self.tool_map["end_conversation"] = end_conversation_tool.func
