import json
import time
from typing import Any, Dict
import logging
import uuid
from typing import List, Dict, Any, Tuple
import copy
from arklex.env.env import Env
import janus
from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda

from arklex.orchestrator.task_graph import TaskGraph
from arklex.env.tools.utils import ToolGenerator
from arklex.types import EventType, StreamType
from arklex.utils.graph_state import (ConvoMessage, NodeInfo, OrchestratorMessage,
                                      MessageState, PathNode, StatusEnum,
                                      BotConfig, Slot, Timing, Metadata,
                                      Taskgraph, Memory, Params)
from arklex.utils.utils import format_chat_history


load_dotenv()
logger = logging.getLogger(__name__)

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
        params = inputs["parameters"]

        timing = Timing(
            taskgraph=params.get("metadata", {}).get("timing", {}).get("taskgraph", None)
        )
        metadata = Metadata(
            chat_id=params.get("metadata", {}).get("chat_id", str(uuid.uuid4())),
            turn_id=params.get("metadata", {}).get("turn_id", 0),
            hitl=params.get("metadata", {}).get("hitl", None),
            timing=timing,
        )
        taskgraph = Taskgraph(
            dialog_states=params.get("taskgraph", {}).get("dialog_states", {}),
            path=params.get("taskgraph", {}).get("path", []),
            curr_node=params.get("taskgraph", {}).get("curr_node", None),
            curr_pred_intent=params.get("taskgraph", {}).get("curr_pred_intent", None),
            node_limit=params.get("taskgraph", {}).get("node_limit", {}),
            nlu_records=params.get("taskgraph", {}).get("nlu_records", []),
            node_status=params.get("taskgraph", {}).get("node_status", {}),
            available_global_intents=params.get("taskgraph", {}).get("available_global_intents", [])
        )
        memory = Memory(
            history=params.get("memory", {}).get("history", []),
            tool_response=params.get("memory", {}).get("tool_response", {}),
        )
        params = Params(
            metadata=metadata,
            taskgraph=taskgraph,
            memory=memory
        )

        chat_history_copy = copy.deepcopy(chat_history)
        chat_history_copy.append({"role": self.user_prefix, "content": text})
        chat_history_str = format_chat_history(chat_history_copy)
        if not params["taskgraph"]["dialog_states"]:
            params["taskgraph"]["dialog_states"] = {}
        params["metadata"]["turn_id"] += 1
        params["metadata"]["tool_response"] = {}
        if not params["memory"]["history"]:
            params["memory"]["history"] = copy.deepcopy(chat_history_copy)
        else:
            params["memory"]["history"].append(chat_history_copy[-2])
            params["memory"]["history"].append(chat_history_copy[-1])
        return text, chat_history_str, params

    def check_skip_node(self, node_info: NodeInfo, params: Params):
        if not node_info["can_skipped"]:
            return False
        cur_node_id = params["taskgraph"]["curr_node"]
        if cur_node_id in params["taskgraph"]["node_limit"]:
            if params["taskgraph"]["node_limit"][cur_node_id] <= 0:
                return True
        return False

    def post_process_node(self, node_info: NodeInfo, params: Params, update_info:dict={}):
        '''
        update_info is a dict of
            skipped = Optional[bool]
        '''
        curr_node = params["taskgraph"]["curr_node"]
        node = PathNode(
            node_id = curr_node,
            is_skipped = update_info.get("is_skipped", False),
            in_flow_stack=node_info.get("add_flow_stack", False),
            nested_graph_node_value = None,
            nested_graph_leaf_jump = None,
        )
        
        params["taskgraph"]['path'].append(node)

        if curr_node in params["taskgraph"]["node_limit"]:
            params["taskgraph"]["node_limit"][curr_node] -= 1
        return params
    
    def handl_direct_node(self, node_info: NodeInfo, params: Params):
        # Direct response
        node_attribute = node_info["attributes"]
        if node_attribute.get("value", "").strip() and node_attribute.get("direct"):
            return_response = {
                "answer": node_attribute["value"],
                "parameters": params
            }   
            if node_attribute.get("type", "") == "multiple-choice" and node_attribute.get("choice_list", []):
                return_response["choice_list"] = node_attribute["choice_list"]
            return True, return_response, params
        return False, None, params
    
    def perform_node(self, node_info: NodeInfo, params: Params, text: str, message_flow: str, chat_history_str: str, stream_type: StreamType, message_queue: janus.SyncQueue):
        # Tool/Worker
        user_message = ConvoMessage(history=chat_history_str, message=text)
        orchestrator_message = OrchestratorMessage(message=node_info["attributes"]["value"], attribute=node_info["attributes"])
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
        message_state = MessageState(
            sys_instruct=sys_instruct, 
            bot_config=bot_config,
            user_message=user_message, 
            orchestrator_message=orchestrator_message, 
            trajectory=params["memory"]["history"], 
            message_flow=message_flow, 
            slots=params["taskgraph"]["dialog_states"],
            metadata=params["metadata"],
            is_stream=True if stream_type is not None else False,
            message_queue=message_queue
        )
        
        response_state, params = self.env.step(node_info["resource_id"], message_state, params)
        return response_state, params
    
    def get_response(self, inputs: dict, stream_type: StreamType = None, message_queue: janus.SyncQueue = None) -> Dict[str, Any]:
        text, chat_history_str, params = self.init_params(inputs)
        ##### TaskGraph Chain
        taskgraph_inputs = {
            "text": text,
            "chat_history_str": chat_history_str,
            "parameters": params,  ## TODO: different params for different components
            "allow_global_intent_switch": True, # ensure global intent changing happends only in the first get_node.
        }
        taskgraph_chain = RunnableLambda(self.task_graph.get_node) | RunnableLambda(self.task_graph.postprocess_node)

        
        counter_message_worker = 0
        counter_ragmsg_worker = 0
        counter_planner = 0 # TODO: when planner is re-implemented, execute/break the loop based on whether the planner should be used (bot config) .
        
        message_flow = "" # Store the message flow between different nodes

        n_node_performed = 0
        max_n_node_performed = 5
        while n_node_performed < max_n_node_performed:
            taskgraph_start_time = time.time()
            node_info, params = taskgraph_chain.invoke(taskgraph_inputs)
            taskgraph_inputs["allow_global_intent_switch"] = False
            params["metadata"]["timing"]["taskgraph"] = time.time() - taskgraph_start_time
            # Check if current node can be skipped
            can_skip = self.check_skip_node(node_info, params)
            if can_skip:
                params = self.post_process_node(node_info, params, {"is_skipped": True})
                continue
            logger.info(f"The current node info is : {node_info}")
            # Check current node attributes
            if node_info["resource_id"] == "planner":
                counter_planner += 1
            elif node_info["resource_id"] == self.env.name2id["MessageWorker"]:
                counter_message_worker += 1
            elif node_info["resource_id"] == self.env.name2id["RagMsgWorker"]:
                counter_ragmsg_worker += 1
            # handle direct node
            is_direct_node, direct_response, params = self.handl_direct_node(node_info, params)
            if is_direct_node:
                params = self.post_process_node(node_info, params)
                return direct_response
            # perform node
            response_state, params = self.perform_node(node_info,
                                                       params,
                                                       text,
                                                       message_flow,
                                                       chat_history_str,
                                                       stream_type,
                                                       message_queue)
            params = self.post_process_node(node_info, params)
            message_flow = response_state.get("message_flow", "")
            n_node_performed += 1

            # If the current node is not complete, then no need to continue to the next node
            node_status = params["taskgraph"]["node_status"]
            cur_node_id = params["taskgraph"]["curr_node"]
            status = node_status.get(cur_node_id, StatusEnum.COMPLETE.value)
            if status == StatusEnum.INCOMPLETE.value:
                break
            # If the counter of message worker or counter of default worker or counter of ragmsg worker == 1, break the loop
            if counter_message_worker == 1 or counter_planner == 1 or counter_ragmsg_worker == 1:
                break
            if node_info["is_leaf"] is True:
                break

        if not response_state.get("response", ""):
            logger.info("No response, do context generation")
            if stream_type is None:
                response_state = ToolGenerator.context_generate(response_state)
            else:
                response_state = ToolGenerator.stream_context_generate(response_state)
            params["memory"]["tool_response"] = {}
        
        response = response_state.get("response", "")
        return {
            "answer": response,
            "parameters": params,
            "human-in-the-loop": params['metadata'].get('hitl', None),
        }
