import contextlib
import json
from typing import Any

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import START, StateGraph

from arklex.env.agents.agent import BaseAgent, register_agent
from arklex.env.prompts import load_prompts
from arklex.env.tools.tools import TYPE_CONVERTERS, register_tool
from arklex.env.tools.utils import trace
from arklex.orchestrator.entities.msg_state_entities import MessageState, StatusEnum
from arklex.types import EventType, StreamType
from arklex.utils.logging_utils import LogContext
from arklex.utils.provider_utils import validate_and_get_model_class

log_context = LogContext(__name__)


@register_tool("Ends the conversation with a thank you message.", isResponse=True)
def end_conversation(state: MessageState) -> str:
    """
    Ends the conversation with a thank you message. This is a default tool that is used to end the conversation and added to the tool map of the OpenAIAgent.

    Args:
        state (MessageState): The state of the conversation.

    Returns:
        str: The thank you message or a fallback.
    """
    log_context.info("Ending the conversation.")
    state.status = StatusEnum.COMPLETE
    model_class = validate_and_get_model_class(state.bot_config.llm_config)

    llm = model_class(model=state.bot_config.llm_config.model_type_or_path)
    try:
        result = llm.invoke([
            SystemMessage(
                content="Ends the conversation with a thank you and goodbye message."
            ).model_dump(),
        ])
        content = None
        if isinstance(result, dict):
            content = result.get("content") or result.get("result")
            if not content and "choices" in result:
                with contextlib.suppress(Exception):
                    content = result["choices"][0]["message"]["content"]
        elif hasattr(result, "content"):
            content = result.content
        elif isinstance(result, str):
            import json
            try:
                parsed = json.loads(result)
                content = parsed.get("content") or parsed.get("result")
                if not content and "choices" in parsed:
                    content = parsed["choices"][0]["message"]["content"]
            except Exception:
                content = result  # fallback to the string itself

        if not content or "dummy response" in str(content):
            raise ValueError("LLM returned no content or dummy response")
        return content
    except Exception as e:
        log_context.error(f"Error when ending conversation: {e}")
        return "I hope I was able to help you today. Goodbye!"


@register_agent
class OpenAIAgent(BaseAgent):
    description: str = "General-purpose Arklex agent for chat or voice."

    def __init__(
        self, successors: list, predecessors: list, tools: dict, state: MessageState
    ) -> None:
        super().__init__()
        self.action_graph: StateGraph = self._create_action_graph()
        self.llm: BaseChatModel | None = None
        self.available_tools: dict[str, tuple[dict[str, Any], Any]] = {}
        self.tool_map = {}
        self.tool_defs = []
        self.tool_args: dict[str, Any] = {}
        self.tool_slots: dict[str, Any] = {}

        self._load_tools(successors=successors, predecessors=predecessors, tools=tools)
        self._configure_tools()

        log_context.info(f"OpenAIAgent initialized with {len(self.tool_defs)} tools.")

    def generate(self, state: MessageState) -> MessageState:
        """Generate response without streaming."""
        return self._generate_response(state, stream=False)

    def text_stream_generate(self, state: MessageState) -> MessageState:
        """Generate response with text streaming capability."""
        return self._generate_response(state, stream=True, is_speech=False)

    def speech_stream_generate(self, state: MessageState) -> MessageState:
        """Generate response with speech streaming capability."""
        return self._generate_response(state, stream=True, is_speech=True)

    def _prepare_prompt(self, state: MessageState, is_speech: bool = False) -> str:
        """Prepare the input prompt for generation."""
        if self.prompt:
            return self.prompt

        orchestrator_message = state.orchestrator_message
        orch_msg_content: str = (
            "None" if not orchestrator_message.message else orchestrator_message.message
        )

        prompts: dict[str, str] = load_prompts(state.bot_config)

        # Choose prompt based on speech flag
        if is_speech:
            prompt_key = (
                "function_calling_agent_prompt_speech"
                if "function_calling_agent_prompt_speech" in prompts
                else "function_calling_agent_prompt"
            )
        else:
            prompt_key = "function_calling_agent_prompt"

        prompt: PromptTemplate = PromptTemplate.from_template(prompts[prompt_key])
        input_prompt = prompt.invoke(
            {
                "sys_instruct": state.sys_instruct,
                "message": orch_msg_content,
            }
        )
        return input_prompt.text

    def _add_prompt_to_trajectory(self, state: MessageState, input_prompt: str) -> None:
        """Add the input prompt to the function calling trajectory if not already present."""
        if not any(
            message.get("content") == input_prompt
            for message in state.function_calling_trajectory
        ):
            log_context.info("Adding input prompt to the function calling trajectory.")
            state.function_calling_trajectory.append(
                SystemMessage(content=input_prompt).model_dump()
            )

    def _process_tool_calls(self, state: MessageState, ai_message: AIMessage) -> None:
        """Process tool calls and update the function calling trajectory."""
        if not ai_message.tool_calls:
            return

        log_context.info("Processing tool calls.")
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call.get("name")
            if tool_name in self.tool_map:
                # Ensure tool_call has proper structure
                tool_call_id = tool_call.get("id", f"call_{tool_name}")
                tool_call_args = tool_call.get("args", {})

                # Create properly structured tool call
                structured_tool_call = {
                    "name": tool_name,
                    "args": tool_call_args,
                    "id": tool_call_id,
                }

                state.function_calling_trajectory.append(
                    AIMessage(
                        content=f"Calling tool: {tool_name}",
                        tool_calls=[structured_tool_call],
                    ).model_dump()
                )

                # Prepare arguments for tool execution
                tool_args = {
                    **tool_call_args,
                    **self.tool_args.get(tool_name, {}),
                }

                # Call tool with unified interface
                tool_response = self._execute_tool(tool_name, state, tool_args)

                state.function_calling_trajectory.append(
                    ToolMessage(
                        name=tool_name,
                        content=json.dumps(tool_response),
                        tool_call_id=tool_call_id,
                    ).model_dump()
                )
            else:
                log_context.warning(f"Tool {tool_name} not found in tool map.")

    def _stream_response(self, state: MessageState, final_chain: BaseChatModel) -> str:
        """Stream the response and put chunks in the message queue."""
        answer = ""
        for chunk in final_chain.stream(state.function_calling_trajectory):
            if hasattr(chunk, "content") and chunk.content:
                answer += chunk.content
                state.message_queue.put(
                    {"event": EventType.CHUNK.value, "message_chunk": chunk.content}
                )
        return answer

    def _generate_response(
        self, state: MessageState, stream: bool = False, is_speech: bool = False
    ) -> MessageState:
        """Unified response generation method with optional streaming."""
        generation_type = (
            "speech streaming"
            if is_speech and stream
            else "streaming"
            if stream
            else "standard"
        )
        log_context.info(f"\nGenerating {generation_type} response using the agent.")

        if state.status == StatusEnum.INCOMPLETE:
            input_prompt = self._prepare_prompt(state, is_speech)
            self._add_prompt_to_trajectory(state, input_prompt)

        log_context.info(f"\nagent messages: {state.function_calling_trajectory}")

        final_chain = self.llm
        ai_message: AIMessage = final_chain.invoke(state.function_calling_trajectory)

        log_context.info(f"Generated answer: {ai_message}")

        # Process tool calls first
        self._process_tool_calls(state, ai_message)

        # Generate final response
        if ai_message.tool_calls:
            # After tool execution, get final response
            if stream:
                answer = self._stream_response(state, final_chain)
            else:
                ai_message = final_chain.invoke(state.function_calling_trajectory)
                answer = ai_message.content
        else:
            # No tool calls
            if stream:
                answer = self._stream_response(state, final_chain)
            else:
                answer = ai_message.content

        state.message_flow = ""
        state.response = answer

        if not stream:
            state = trace(input=answer, state=state)

        return state

    def choose_generator(self, state: MessageState) -> str:
        """Choose the appropriate generator based on stream type and language."""
        if state.bot_config.language == "CN" and state.stream_type == StreamType.SPEECH:
            return "text_stream_generate"
        if (
            state.stream_type == StreamType.TEXT
            or state.stream_type == StreamType.AUDIO
        ):
            return "text_stream_generate"
        elif state.stream_type == StreamType.SPEECH:
            return "speech_stream_generate"
        return "generate"

    def _create_action_graph(self) -> StateGraph:
        workflow = StateGraph(MessageState)
        workflow.add_node("generate", self.generate)
        workflow.add_node("text_stream_generate", self.text_stream_generate)
        workflow.add_node("speech_stream_generate", self.speech_stream_generate)

        # Add conditional edges based on stream type
        workflow.add_conditional_edges(START, self.choose_generator)

        return workflow

    def _execute(self, msg_state: MessageState, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        model_class = validate_and_get_model_class(msg_state.bot_config.llm_config)

        self.llm = model_class(model=msg_state.bot_config.llm_config.model_type_or_path)
        self.llm = self.llm.bind_tools(self.tool_defs)
        self.prompt: str = kwargs.get("prompt", "")
        graph = self.action_graph.compile()
        result = graph.invoke(msg_state)
        return result

    def _load_tools(self, successors: list, predecessors: list, tools: dict) -> None:
        """
        Load tools for the agent.
        This method is called during the initialization of the agent.
        """
        for node in predecessors:
            if node.type == "tool":
                tool_id = (
                    f"{node.resource_id}_{node.attributes['task']}"
                    if node.attributes.get("task")
                    else node.resource_id
                )
                tool_id = tool_id.replace(" ", "_").replace("/", "_")
                self.available_tools[tool_id] = (tools[node.resource_id], node)

    def _configure_tools(self) -> None:
        """
        Configure tools for the agent.
        This method is called during the initialization of the agent.
        """
        for tool_id, (tool, node_info) in self.available_tools.items():
            tool_object = tool["tool_instance"]
 
            log_context.info(
                f"Configuring tool: {tool_object.func.__name__} with slots: {tool_object.slots}"
            )
            tool_def = tool_object.to_openai_tool_def_v2()
            tool_def["function"]["name"] = tool_id
            self.tool_defs.append(tool_def)
            self.tool_slots[tool_id] = tool_object.slots.copy()
            self.tool_map[tool_id] = tool_object.func
            combined_args: dict[str, Any] = {
                **tool["fixed_args"],
                **(node_info.additional_args or {}),
            }
            self.tool_args[tool_id] = combined_args
        log_context.info(f"Tool Definitions: {self.tool_defs}")

        end_conversation_tool = end_conversation()
        end_conversation_tool_def = end_conversation_tool.to_openai_tool_def_v2()
        end_conversation_tool_def["function"]["name"] = "end_conversation"
        self.tool_defs.append(end_conversation_tool_def)
        self.tool_map["end_conversation"] = end_conversation_tool.func

    def _execute_tool(
        self, tool_name: str, state: MessageState, tool_args: dict[str, Any]
    ) -> Any:  # noqa: ANN401
        """Execute a tool with unified interface.

        This method handles the different calling patterns for different types of tools.
        For http_tool, it prepares the slots parameter. For other tools, it passes state directly.

        Args:
            tool_name: Name of the tool to execute
            state: Current message state
            tool_args: Arguments for the tool

        Returns:
            Tool execution result
        """
        def build_slot_values(schema: list[dict[str, Any]], tool_args: dict[str, Any]) -> list[dict[str, Any]]:
            def type_convert(value: object, slot_type: str) -> object:
                if value is None:
                    return value
                try:
                    converter = TYPE_CONVERTERS.get(slot_type)
                    if converter:
                        return converter(value)
                    return value
                except Exception:
                    return value

            def flatten_group_items(group_items: list[Any]) -> list[dict[str, Any]]:
                result: list[dict[str, Any]] = []
                for item in group_items:
                    if isinstance(item, list):
                        flat = {slot["name"]: slot["value"] for slot in item}
                        result.append(flat)
                    else:
                        result.append(item)
                return result

            result = []
            for slot in schema:
                name = slot["name"]
                slot_type = slot["type"]
                value_source = slot.get("valueSource", "prompt")
                slot_value = None

                if slot_type == "group":
                    if slot.get("repeatable", False):
                        group_values = tool_args.get(name, [])
                        if not group_values and value_source == "default" or not group_values and value_source == "fixed":
                            group_values = [slot.get("value", "")]
                        slot_value = [build_slot_values(slot["schema"], item) for item in group_values]
                        slot_value = flatten_group_items(slot_value)
                    else:
                        group_value = tool_args.get(name, {})
                        if not group_value and value_source == "default" or not group_value and value_source == "fixed":
                            group_value = slot.get("value", "")
                        slot_value = build_slot_values(slot["schema"], group_value)
                else:
                    if value_source == "fixed":
                        slot_value = slot.get("value", "")
                    elif value_source == "default":
                        slot_value = tool_args.get(name, slot.get("value", ""))
                    else:  # prompt or anything else
                        slot_value = tool_args.get(name, "")
                    slot_value = type_convert(slot_value, slot_type)

                slot_dict = slot.copy()
                slot_dict["value"] = slot_value
                result.append(slot_dict)
            return result

        if "http_tool" in tool_name:
            all_slots = self.tool_slots.get(tool_name, [])
            slots = build_slot_values([slot.model_dump() if hasattr(slot, "model_dump") else slot for slot in all_slots], tool_args)
            # Call http_tool with slots parameter, excluding slots from tool_args
            filtered_args = {k: v for k, v in tool_args.items() if k != "slots"}
            return self.tool_map[tool_name](slots=slots, **filtered_args)
        else:
            # Call other tools with state parameter
            return self.tool_map[tool_name](state=state, **tool_args)
