import os
import logging
import json
import uuid
import inspect
import traceback

from arklex.utils.graph_state import MessageState, StatusEnum
from arklex.utils.slot import Slot
from arklex.orchestrator.NLU.nlu import SlotFilling
from arklex.utils.utils import format_chat_history
from arklex.exceptions import ToolExecutionError, AuthenticationError

logger = logging.getLogger(__name__)

    
def register_tool(desc, slots=[], outputs=[], isResponse=False):
    current_file_dir = os.path.dirname(__file__)
    def inner(func):
        file_path = inspect.getfile(func)
        relative_path = os.path.relpath(file_path, current_file_dir)
        # reformat the relative path to replace / and \\ with -, and remove .py, because the function calling in openai only allow the function name match the patter the pattern '^[a-zA-Z0-9_-]+$'
        # different file paths format in Windows and linux systems
        relative_path = relative_path.replace("/", "-").replace("\\", "-").replace(".py", "")
        key = f"{relative_path}-{func.__name__}"
        tool = lambda : Tool(func, key, desc, slots, outputs, isResponse)
        return tool
    return inner

class Tool:
    def __init__(self, func, name, description, slots, outputs, isResponse):
        self.func = func
        self.name = name
        self.description = description
        self.output = outputs
        self.slotfillapi: SlotFilling = None
        self.info = self.get_info(slots)
        self.slots = [Slot.model_validate(slot) for slot in slots]
        self.isResponse = isResponse

    def get_info(self, slots):
        self.properties = {}
        for slot in slots:
            self.properties[slot["name"]] = {k: v for k, v in slot.items() if k in ["type", "description", "prompt", "items"]}
        required = [slot["name"] for slot in slots if slot.get("required", False)]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": required,
                },
            },
        }
        
    def init_slotfilling(self, slotfillapi: SlotFilling):
        self.slotfillapi = slotfillapi

    def _init_slots(self, state: MessageState):
        default_slots = state.slots.get("default_slots", [])
        logger.info(f'Default slots are: {default_slots}')
        if not default_slots:
            return
        response = {}
        for default_slot in default_slots:
            response[default_slot.name] = default_slot.value
            for slot in self.slots:
                if slot.name == default_slot.name and default_slot.value:
                    slot.value = default_slot.value
                    slot.verified = True
        state.function_calling_trajectory.append({
            "role": "tool",
            "tool_call_id": str(uuid.uuid4()),
            "name": "default_slots",
            "content": json.dumps(response)
        })
        
        logger.info(f'Slots after initialization are: {self.slots}')
        
    def _execute(self, state: MessageState, **fixed_args):
        # if this tool has been called before, then load the previous slots status
        if state.slots.get(self.name):
            self.slots = state.slots[self.name]
        else:
            state.slots[self.name] = self.slots
        # init slot values saved in default slots
        self._init_slots(state)
        # do slotfilling
        chat_history_str = format_chat_history(state.function_calling_trajectory)
        slots : list[Slot] = self.slotfillapi.execute(self.slots, chat_history_str, self.llm_config)
        logger.info(f'{slots=}')
        if not all([slot.value and slot.verified for slot in slots if slot.required]):
            for slot in slots:
                # if there is extracted slots values but haven't been verified
                if slot.value and not slot.verified:
                    # check whether it verified or not
                    verification_needed, thought = self.slotfillapi.verify_needed(slot, chat_history_str)
                    if verification_needed:
                        response = slot.prompt + "The reason is: " + thought
                        break
                    else:
                        slot.verified = True
                # if there is no extracted slots values, then should prompt the user to fill the slot
                if not slot.value:
                    response = slot.prompt
                    break
            
            state.status = StatusEnum.INCOMPLETE

        # if slot.value is not empty for all slots, and all the slots has been verified, then execute the function
        tool_success = False
        if all([slot.value and slot.verified for slot in slots if slot.required]):
            logger.info("all slots filled")
            kwargs = {slot.name: slot.value for slot in slots}
            combined_kwargs = {**kwargs, **fixed_args, **self.llm_config}
            try:
                response = self.func(**combined_kwargs)
                tool_success = True
            except ToolExecutionError as tee:
                logger.error(traceback.format_exc())
                response = tee.extra_message
            except AuthenticationError as ae:
                logger.error(traceback.format_exc())
                response = str(ae)
            except Exception as e:
                logger.error(traceback.format_exc())
                response = str(e)
            logger.info(f"Tool {self.name} response: {response}")
            call_id = str(uuid.uuid4())
            state.function_calling_trajectory.append({
                'content': None, 
                'role': 'assistant', 
                'tool_calls': [
                    {
                        'function': {
                            'arguments': json.dumps(kwargs), 
                            'name': self.name
                        }, 
                        'id': call_id, 
                        'type': 'function'
                    }
                ], 
                'function_call': None
            })
            state.function_calling_trajectory.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": self.name,
                "content": response
            })
            state.status = StatusEnum.COMPLETE if tool_success else StatusEnum.INCOMPLETE

        state.trajectory[-1][-1].input = slots
        state.trajectory[-1][-1].output = response

        if self.isResponse and tool_success:
            logger.info("Tool output is stored in response instead of message flow")
            state.response = response
        else:
            state.message_flow = state.message_flow + f"Context from {self.name} tool execution: {response}\n"
        state.slots[self.name] = slots
        return state

    def execute(self, state: MessageState, **fixed_args):
        self.llm_config = state.bot_config.llm_config.model_dump()
        state = self._execute(state, **fixed_args)
        return state
    
    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return f"{self.__class__.__name__}"
    