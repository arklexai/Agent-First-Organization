import json
import time
from typing import Any, Dict
import logging
import uuid
import os
from typing import List, Dict, Any, Tuple
import ast
import copy
import janus
from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda

from arklex.env.env import Env
from arklex.orchestrator.task_graph import TaskGraph
from arklex.env.tools.utils import ToolGenerator
from arklex.types import StreamType
from arklex.utils.graph_state import (ConvoMessage, NodeInfo, OrchestratorMessage,
                                      MessageState, PathNode,StatusEnum,
                                      BotConfig, Params, ResourceRecord,
                                      OrchestratorResp)
from arklex.utils.utils import format_chat_history


load_dotenv()
logger = logging.getLogger(__name__)

INFO_WORKERS = ["planner", "MessageWorker", "RagMsgWorker"]

class AgentOrg:
    def __init__(self, config, env: Env, **kwargs):
        if isinstance(config, dict):
            self.product_kwargs = config
        else:
            self.product_kwargs = json.load(open(config))
        self.user_prefix = "user"
        self.worker_prefix = "assistant"
        self.environment_prefix = "tool"
        self.__eos_token = "\n"
        self.task_graph = TaskGraph("taskgraph", self.product_kwargs)
        self.env = env

    
    def init_params(self, inputs) -> Tuple[str, str, Params]:
        text = inputs["text"]
        chat_history = inputs["chat_history"]
        input_params = inputs["parameters"]

        # Create base params with defaults
        params = Params()
        
        # Update with any provided values
        if input_params:
            params = Params.model_validate(input_params)
        
        # Update specific fields
        chat_history_copy = copy.deepcopy(chat_history)
        chat_history_copy.append({"role": self.user_prefix, "content": text})
        chat_history_str = format_chat_history(chat_history_copy)        
        # Update turn_id and function_calling_trajectory
        params.metadata.turn_id += 1
        if not params.memory.function_calling_trajectory:
            params.memory.function_calling_trajectory = copy.deepcopy(chat_history_copy)
        else:
            params.memory.function_calling_trajectory.extend(chat_history_copy[-2:])
        
        return text, chat_history_str, params

    def check_skip_node(self, node_info: NodeInfo, params: Params):
        return False
        if not node_info.can_skipped:
            return False
        cur_node_id = params.taskgraph.curr_node
        if cur_node_id in params.taskgraph.node_limit:
            if params.taskgraph.node_limit[cur_node_id] <= 0:
                return True
        return False

    def post_process_node(self, node_info: NodeInfo, params: Params, update_info:dict={}):
        '''
        update_info is a dict of
            skipped = Optional[bool]
        '''
        curr_node = params.taskgraph.curr_node
        node = PathNode(
            node_id = curr_node,
            is_skipped = update_info.get("is_skipped", False),
            in_flow_stack=node_info.add_flow_stack,
            nested_graph_node_value = None,
            nested_graph_leaf_jump = None,
        )
        
        params.taskgraph.path.append(node)

        if curr_node in params.taskgraph.node_limit:
            params.taskgraph.node_limit[curr_node] -= 1
        return params
    
    def handl_direct_node(self, node_info: NodeInfo, params: Params):
        # Direct response
        node_attribute = node_info.attributes
        if node_attribute.get("value", "").strip() and node_attribute.get("direct"):
            params = self.post_process_node(node_info, params)
            return_response = OrchestratorResp(
                answer=node_attribute["value"],
                parameters=params.model_dump()
            )
            # TODO: multiple choice list
            # if node_attribute.get("type", "") == "multiple-choice" and node_attribute.get("choice_list", []):
            #     return_response["choice_list"] = node_attribute["choice_list"]
            return True, return_response, params
        return False, None, params
    
    def perform_node(self, node_info: NodeInfo, params: Params, text: str, chat_history_str: str, same_turn: bool, stream_type: StreamType, message_queue: janus.SyncQueue):
        # Tool/Worker
        user_message = ConvoMessage(history=chat_history_str, message=text)
        orchestrator_message = OrchestratorMessage(message=node_info.attributes["value"], attribute=node_info.attributes)
        sys_instruct = "You are a " + self.product_kwargs["role"] + ". " + \
                            self.product_kwargs["user_objective"] + \
                            self.product_kwargs["builder_objective"] + \
                            self.product_kwargs["intro"] + \
                            self.product_kwargs.get("opt_instruct", "")
        bot_config = BotConfig(
            bot_id=self.product_kwargs.get("bot_id", "default"),
            version=self.product_kwargs.get("version", "default"),
            language=self.product_kwargs.get("language", "EN"),
            bot_type=self.product_kwargs.get("bot_type", "presalebot"),
        )
    
        # Create initial resource record with common info and output from trajectory
        resource_record = ResourceRecord(
            info={
                "id": node_info.resource_id,
                "name": node_info.resource_name,
                "attribute": node_info.attributes,
                "node_id": params.taskgraph.curr_node
            }
        )

        # If this is a new turn, create a new list
        if not same_turn:
            params.memory.trajectory.append([])
        
        # Add resource record to current turn's list
        params.memory.trajectory[-1].append(resource_record)

        message_state = MessageState(
            sys_instruct=sys_instruct, 
            bot_config=bot_config,
            user_message=user_message, 
            orchestrator_message=orchestrator_message, 
            function_calling_trajectory=params.memory.function_calling_trajectory, 
            trajectory=params.memory.trajectory,
            message_flow=params.memory.trajectory[-1][-1].output if params.memory.trajectory[-1] else "", 
            slots=params.taskgraph.dialog_states,
            metadata=params.metadata,
            is_stream=True if stream_type is not None else False,
            message_queue=message_queue
        )
        
        response_state, params = self.env.step(node_info.resource_id, message_state, params)
        params.memory.trajectory = response_state.trajectory
        return response_state, params
    
    def _get_response(self, 
                     inputs: dict, 
                     stream_type: StreamType = None, 
                     message_queue: janus.SyncQueue = None) -> OrchestratorResp:
        text, chat_history_str, params = self.init_params(inputs)
        ##### TaskGraph Chain
        same_turn = False
        taskgraph_inputs = {
            "text": text,
            "chat_history_str": chat_history_str,
            "parameters": params,
            "allow_global_intent_switch": True,
        }
        taskgraph_chain = RunnableLambda(self.task_graph.get_node) | RunnableLambda(self.task_graph.postprocess_node)

        # TODO: when planner is re-implemented, execute/break the loop based on whether the planner should be used (bot config).
        msg_counter = 0
        
        n_node_performed = 0
        max_n_node_performed = 5
        while n_node_performed < max_n_node_performed:
            taskgraph_start_time = time.time()
            node_info, params = taskgraph_chain.invoke(taskgraph_inputs)
            taskgraph_inputs["allow_global_intent_switch"] = False
            params.metadata.timing.taskgraph = time.time() - taskgraph_start_time
            # Check if current node can be skipped
            can_skip = self.check_skip_node(node_info, params)
            if can_skip:
                params = self.post_process_node(node_info, params, {"is_skipped": True})
                continue
            logger.info(f"The current node info is : {node_info}")
            # Check current node attributes
            if node_info.resource_name in INFO_WORKERS:
                msg_counter += 1
            # handle direct node
            is_direct_node, direct_response, params = self.handl_direct_node(node_info, params)
            if is_direct_node:
                return direct_response
            # perform node
            response_state, params = self.perform_node(node_info,
                                                       params,
                                                       text,
                                                       chat_history_str,
                                                       same_turn,
                                                       stream_type,
                                                       message_queue)
            params = self.post_process_node(node_info, params)
            n_node_performed += 1
            same_turn = True
            # If the current node is not complete, then no need to continue to the next node
            node_status = params.taskgraph.node_status
            cur_node_id = params.taskgraph.curr_node
            status = node_status.get(cur_node_id, StatusEnum.COMPLETE)
            if status == StatusEnum.INCOMPLETE:
                break
            # If the counter of message worker or counter of planner or counter of ragmsg worker == 1, break the loop
            if msg_counter == 1:
                break
            if node_info.is_leaf is True:
                break

        if not response_state.response:
            logger.info("No response, do context generation")
            if not stream_type:
                response_state = ToolGenerator.context_generate(response_state)
            else:
                response_state = ToolGenerator.stream_context_generate(response_state)
        
        # TODO: Need to reformat the RAG response from trajectory
        # params["memory"]["tool_response"] = {}
        return OrchestratorResp(
            answer=response_state.response,
            parameters=params.model_dump(),
            human_in_the_loop=params.metadata.hitl,
        )
    
    def get_response(self, 
                     inputs: dict, 
                     stream_type: StreamType = None, 
                     message_queue: janus.SyncQueue = None) -> Dict[str, Any]:
        orchestrator_response = self._get_response(inputs, stream_type, message_queue)
        return orchestrator_response.model_dump()
