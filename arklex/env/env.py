import os
import logging
import uuid
import importlib
from typing import Optional
from functools import partial

from arklex.env.tools.tools import Tool
from arklex.env.workers.worker import BaseWorker
from arklex.env.planner.function_calling import FunctionCallingPlanner
from arklex.utils.graph_state import Params, MessageState, NodeInfo
from arklex.orchestrator.NLU.nlu import SlotFilling


logger = logging.getLogger(__name__)


class BaseResourceInitializer:
    @staticmethod
    def init_tools(tools):
        raise NotImplementedError

    @staticmethod
    def init_workers(workers):
        raise NotImplementedError
    
class DefaulResourceInitializer(BaseResourceInitializer):
    @staticmethod
    def init_tools(tools):
        # return dict of valid tools with name and description
        tool_registry = {}
        for tool in tools:
            tool_id = tool["id"]
            name = tool["name"]
            path = tool["path"]
            try: # try to import the tool to check its existance
                filepath = os.path.join("arklex.env.tools", path)
                module_name = filepath.replace(os.sep, ".").rstrip(".py")
                module = importlib.import_module(module_name)
                func = getattr(module, name)
            except Exception as e:
                logger.error(f"Tool {name} is not registered, error: {e}")
                continue
            tool_registry[tool_id] = {
                "name": func().name,
                "description": func().description,
                "execute": func,
                "fixed_args": tool.get("fixed_args", {}),
            }
        return tool_registry
    
    @staticmethod
    def init_workers(workers):
        worker_registry = {}
        for worker in workers:
            worker_id = worker["id"]
            name = worker["name"]
            path = worker["path"]
            try: # try to import the worker to check its existance
                filepath = os.path.join("arklex.env.workers", path)
                module_name = filepath.replace(os.sep, ".").rstrip(".py")
                module = importlib.import_module(module_name)
                func = getattr(module, name)
            except Exception as e:
                logger.error(f"Worker {name} is not registered, error: {e}")
                continue
            worker_registry[worker_id] = {
                "name": name,
                "description": func.description,
                "execute": partial(func, **worker.get("fixed_args", {}))
            }
        return worker_registry

class Env():
    def __init__(self, tools, workers, slotsfillapi = "", resource_inizializer: Optional[BaseResourceInitializer] = None):
        if resource_inizializer is None:
            resource_inizializer = DefaulResourceInitializer()
        self.tools = resource_inizializer.init_tools(tools)
        self.workers = resource_inizializer.init_workers(workers)
        self.name2id = {resource["name"]: id for id, resource in {**self.tools, **self.workers}.items()}
        self.id2name = {id: resource["name"] for id, resource in {**self.tools, **self.workers}.items()}
        self.slotfillapi = self.initialize_slotfillapi(slotsfillapi)
        self.planner = FunctionCallingPlanner(
            tools_map=self.tools,
            name2id=self.name2id
        )

    def initialize_slotfillapi(self, slotsfillapi):
        return SlotFilling(slotsfillapi)

    def step(self, 
             id: str, 
             message_state: MessageState, 
             params: Params,
             node_info: NodeInfo):
        if id in self.tools:
            logger.info(f"{self.tools[id]['name']} tool selected")
            tool: Tool = self.tools[id]["execute"]()
            # slotfilling is in the basetoool class
            tool.init_slotfilling(self.slotfillapi)
            response_state = tool.execute(message_state, **self.tools[id]["fixed_args"])
            params.memory.function_calling_trajectory = response_state.function_calling_trajectory
            params.taskgraph.dialog_states = response_state.slots
            params.taskgraph.node_status[params.taskgraph.curr_node] = response_state.status

                
        elif id in self.workers:
            logger.info(f"{self.workers[id]['name']} worker selected")
            worker: BaseWorker = self.workers[id]["execute"]()
            # If the worker need to do the slotfilling, then it should have this method
            if hasattr(worker, "init_slotfilling"):
                worker.init_slotfilling(self.slotfillapi)
            response_state = worker.execute(message_state, **node_info.additional_args)
            call_id = str(uuid.uuid4())
            params.memory.function_calling_trajectory.append({
                'content': None, 
                'role': 'assistant', 
                'tool_calls': [{'function': {'arguments': "{}", 'name': self.id2name[id]}, 'id': call_id, 'type': 'function'}], 
                'function_call': None
            })
            params.memory.function_calling_trajectory.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": self.id2name[id],
                        "content": response_state.response if response_state.response else response_state.message_flow,
            })
            params.taskgraph.node_status[params.taskgraph.curr_node] = response_state.status
        else:
            logger.info("planner selected")
            action, response_state, msg_history = self.planner.execute(message_state, params.memory.function_calling_trajectory)
        
        
        logger.info(f"Response state from {id}: {response_state}")
        return response_state, params
