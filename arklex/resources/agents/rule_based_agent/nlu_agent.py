import concurrent.futures
import copy
import time
from typing import Any

import janus
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from arklex.memory.entities.memory_entities import ResourceRecord
from arklex.models.llm_config import LLMConfig
from arklex.models.model_service import ModelService
from arklex.orchestrator.entities.orchestrator_param_entities import OrchestratorParams
from arklex.orchestrator.entities.orchestrator_state_entities import (
    BotConfig,
    ConvoMessage,
    OrchestratorState,
    StatusEnum,
)
from arklex.orchestrator.entities.taskgraph_entities import NodeInfo
from arklex.orchestrator.executor.entities import NodeResponse
from arklex.orchestrator.executor.executor import Executor
from arklex.orchestrator.task_graph.nlu_graph import NLUGraph
from arklex.orchestrator.types.stream_types import StreamType
from arklex.resources.agents.base.agent import BaseAgent, register_agent
from arklex.resources.agents.rule_based_agent.post_process import post_process_response
from arklex.resources.tools.utils import ToolGenerator
from arklex.utils.logging.logging_utils import LogContext
from arklex.utils.prompts import load_prompts
from arklex.utils.utils import format_chat_history

log_context = LogContext(__name__)

_GUARDRAIL_SCHEMA: dict = {
    "title": "GuardrailOutput",
    "description": "Structured output for guardrail check",
    "type": "object",
    "properties": {
        "violated": {
            "type": "boolean",
            "description": "True if the user message violates the policy",
        },
    },
    "required": ["violated"],
    "additionalProperties": False,
}


class NLUAgentData(BaseModel):
    prompt: str
    response_length: int
    language: str
    input_guardrail_description: str | None = None
    input_guardrail_fixed_message: str | None = None


@register_agent
class NLUAgent(BaseAgent):
    def __init__(
        self,
        llm_config: LLMConfig,
        nlu_graph: NLUGraph,
        executor: Executor,
        guardrail_llm_config: LLMConfig,
    ) -> None:
        self.user_prefix = "user"
        self.llm_config = llm_config
        self.guardrail_llm_config = guardrail_llm_config
        self.executor = executor
        self.nlu_graph = nlu_graph
        self.agent_data: NLUAgentData = NLUAgentData.model_validate(
            self.nlu_graph.agent_node.data
        )

    def format_system_prompt(self) -> str:
        if self.agent_data.language == "EN":
            return (
                self.agent_data.prompt
                + f"\nLimit the response within {self.agent_data.response_length} words"
            )
        else:
            return (
                self.agent_data.prompt
                + f"限制回复长度在{self.agent_data.response_length}字以内"
            )

    def check_input_guardrail(self, text: str, chat_history_str: str = "") -> bool:
        """Returns True if the input violates the guardrail and should be blocked."""
        history_section = ""
        if chat_history_str:
            history_section = f"Conversation history:\n{chat_history_str}\n"
        prompt = (
            "You are a safety guardrail. Determine if the following user message violates the policy.\n"
            f"Policy: {self.agent_data.input_guardrail_description}\n"
            f"{history_section}"
            f"User message: {text}\n"
        )
        log_context.info(
            f"Input guardrail using model: {self.guardrail_llm_config.model_type_or_path}"
        )
        model_service: ModelService = ModelService(self.guardrail_llm_config)
        try:
            response = model_service.get_response_with_structured_output(
                prompt, _GUARDRAIL_SCHEMA
            )
            return bool(response.get("violated", False))
        except Exception as e:
            log_context.error(f"Input guardrail check failed: {e}")
            return False  # Fail open

    def _guardrail_triggered(
        self, guardrail_future: concurrent.futures.Future[bool] | None
    ) -> bool:
        """Non-blocking check if the parallel guardrail has flagged the input."""
        if guardrail_future is None or not guardrail_future.done():
            return False
        try:
            return guardrail_future.result(timeout=0)
        except Exception as e:
            log_context.error(f"Input guardrail future error: {e}")
            return False

    def init_params(
        self, inputs: dict[str, Any]
    ) -> tuple[str, str, OrchestratorParams, OrchestratorState]:
        text: str = inputs["text"]
        chat_history: list[dict[str, str]] = inputs["chat_history"]
        input_params: dict[str, Any] | None = inputs["parameters"]

        params: OrchestratorParams = OrchestratorParams()

        if input_params:
            params = OrchestratorParams.model_validate(input_params)

        chat_history_copy: list[dict[str, str]] = copy.deepcopy(chat_history)
        chat_history_copy.append({"role": self.user_prefix, "content": text})
        chat_history_str: str = format_chat_history(chat_history_copy)
        params.metadata.turn_id += 1
        if not params.memory.function_calling_trajectory:
            params.memory.function_calling_trajectory = copy.deepcopy(chat_history_copy)
        else:
            params.memory.function_calling_trajectory.extend(chat_history_copy[-2:])

        params.memory.trajectory.append([])

        orch_state: OrchestratorState = OrchestratorState(
            sys_instruct=self.agent_data.prompt,
            bot_config=BotConfig(
                language=self.agent_data.language,
                llm_config=self.llm_config,
            ),
        )
        return text, chat_history_str, params, orch_state

    def check_skip_node(self, node_info: NodeInfo, chat_history_str: str) -> bool:
        if not node_info.attribute.get("can_skipped", False):
            return False

        task = node_info.attribute.get("task", "")
        if not task:
            return False

        prompts = load_prompts(self.agent_data.language)
        prompt = prompts["check_skip_node_prompt"].format(
            chat_history_str=chat_history_str, task=task
        )
        log_context.info(f"prompt for check skip node: {prompt}")

        model_service: ModelService = ModelService(self.llm_config)
        try:
            response_text = model_service.get_response(prompt)
            log_context.info(f"LLM response for task verification: {response_text}")
            response_text = str(response_text).lower().strip()
            return response_text == "yes"
        except Exception as e:
            log_context.error(f"Error in LLM task verification: {str(e)}")
            return False

    def perform_node(
        self,
        orch_state: OrchestratorState,
        node_info: NodeInfo,
        params: OrchestratorParams,
        text: str,
        chat_history_str: str,
        stream_type: StreamType | None,
        message_queue: janus.Queue | None,
    ) -> tuple[NodeResponse, OrchestratorState, OrchestratorParams]:
        # Create initial resource record with common info and output from trajectory
        resource_record: ResourceRecord = ResourceRecord(
            info={
                "resource": node_info.resource,
                "attribute": node_info.attribute,
                "node_id": params.nlugraph.curr_node,
            },
            intent=params.nlugraph.intent,
        )

        # Add resource record to current turn's list
        params.memory.trajectory[-1].append(resource_record)

        # Update orchestrator state
        orch_state.user_message = ConvoMessage(history=chat_history_str, message=text)
        orch_state.function_calling_trajectory = (
            params.memory.function_calling_trajectory
        )
        orch_state.trajectory = params.memory.trajectory
        orch_state.metadata = params.metadata
        orch_state.stream_type = stream_type
        orch_state.message_queue = message_queue
        # Execute the node
        response_state: OrchestratorState
        response_state, node_response = self.executor.step(
            node_info.resource["id"],
            orch_state,
            node_info,
            params.nlugraph.dialog_states,
        )
        # Update params
        params.nlugraph.node_status[node_info.node_id] = node_response.status
        if node_response.slots:
            params.nlugraph.dialog_states = node_response.slots
        return node_info, response_state, params, node_response

    def execute(
        self,
        inputs: dict[str, Any],
        stream_type: StreamType | None,
        message_queue: janus.Queue | None,
    ) -> tuple[str, str, OrchestratorParams, OrchestratorState]:
        text, chat_history_str, params, orch_state = self.init_params(inputs)

        # --- Parallel guardrail: runs alongside execution ---
        guardrail_executor = None
        guardrail_future = None
        if self.agent_data.input_guardrail_description and text != "<start>":
            guardrail_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            guardrail_future = guardrail_executor.submit(
                self.check_input_guardrail, text, chat_history_str
            )

        # NLU Graph Chain
        nlugraph_inputs: dict[str, Any] = {
            "text": text,
            "chat_history_str": chat_history_str,
            "nlu_params": params.nlugraph,
            "allow_global_intent_switch": True,
        }

        orch_state.trajectory = params.memory.trajectory
        nlugraph_chain = RunnableLambda(self.nlu_graph.get_node)

        n_node_performed = 0
        max_n_node_performed = 5
        while n_node_performed < max_n_node_performed:
            if self._guardrail_triggered(guardrail_future):
                guardrail_executor.shutdown(wait=False)
                log_context.info("[Input guardrail] Early abort")
                return NodeResponse(
                    status=StatusEnum.COMPLETE,
                    response=self.agent_data.input_guardrail_fixed_message,
                ), params

            nlugraph_start_time = time.time()
            node_info, nlu_params = nlugraph_chain.invoke(nlugraph_inputs)
            nlugraph_inputs["allow_global_intent_switch"] = False
            params.nlugraph = nlu_params
            params.metadata.timing.nlugraph = time.time() - nlugraph_start_time
            # Check if current node can be skipped
            can_skip = self.check_skip_node(node_info, chat_history_str)
            if can_skip:
                continue
            log_context.info(f"The current node info is : {node_info}")

            # perform node
            node_info, orch_state, params, node_response = self.perform_node(
                orch_state,
                node_info,
                params,
                text,
                chat_history_str,
                stream_type,
                message_queue,
            )

            n_node_performed += 1
            # If the current node is not complete, then no need to continue to the next node
            if node_response.status == StatusEnum.INCOMPLETE:
                break
            # If the current node has a response, break the loop
            if node_response.response:
                break
            # If the current node is a leaf node, break the loop
            if node_info.is_leaf is True:
                break

        if not node_response.response and self._guardrail_triggered(guardrail_future):
            guardrail_executor.shutdown(wait=False)
            log_context.info("[Input guardrail] Early abort before context generation")
            return NodeResponse(
                status=StatusEnum.COMPLETE,
                response=self.agent_data.input_guardrail_fixed_message,
            ), params

        if not node_response.response:
            log_context.info("No response, do context generation")
            if stream_type == StreamType.NON_STREAM:
                answer = ToolGenerator.context_generate(orch_state)
                node_response.response = answer
            else:
                answer = ToolGenerator.stream_context_generate(orch_state)
                node_response.response = answer
        node_response = post_process_response(
            orch_state,
            node_response,
        )

        # Check guardrail result (ran in parallel with the execution above)
        if guardrail_future is not None:
            try:
                triggered = guardrail_future.result(timeout=30)
            except Exception as e:
                log_context.error(f"[Input guardrail] future error: {e}")
                triggered = False
            finally:
                guardrail_executor.shutdown(wait=False)
            if triggered:
                log_context.info(
                    "[Input guardrail] triggered, returning the fixed block message"
                )
                node_response.response = self.agent_data.input_guardrail_fixed_message

        return node_response, params
