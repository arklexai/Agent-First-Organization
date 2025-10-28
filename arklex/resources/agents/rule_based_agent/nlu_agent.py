import copy
import time
from typing import Any

import janus
from langchain_core.runnables import RunnableLambda

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
from arklex.utils.logging.logging_utils import LogContext
from arklex.utils.prompts import load_prompts
from arklex.utils.utils import format_chat_history

log_context = LogContext(__name__)


@register_agent
class NLUAgent(BaseAgent):
    def __init__(
        self,
        product_kwargs: dict[str, Any],
        nlu_graph: NLUGraph,
        executor: Executor,
    ) -> None:
        self.user_prefix = "user"
        self.product_kwargs = product_kwargs
        self.executor = executor
        self.llm_config: LLMConfig = LLMConfig.model_validate(
            self.product_kwargs.get("model")
        )
        self.nlu_graph = nlu_graph
        self.model_service: ModelService = ModelService(self.llm_config)
        bot_config = BotConfig.model_validate(self.product_kwargs.get("bot_config", {}))
        self.prompts = load_prompts(bot_config)

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
            sys_instruct=self.product_kwargs.get("sys_instruct", ""),
            bot_config=BotConfig.model_validate(
                self.product_kwargs.get("bot_config", {})
            ),
        )
        return text, chat_history_str, params, orch_state

    def check_skip_node(self, node_info: NodeInfo, chat_history_str: str) -> bool:
        if not node_info.attribute.get("can_skipped", False):
            return False

        task = node_info.attribute.get("task", "")
        if not task:
            return False

        prompt = self.prompts["check_skip_node_prompt"].format(
            chat_history_str=chat_history_str, task=task
        )
        log_context.info(f"prompt for check skip node: {prompt}")

        try:
            response_text = self.model_service.get_response(prompt)
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
        return node_response, orch_state, params
