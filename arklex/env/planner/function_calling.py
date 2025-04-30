import logging
import os
import json
from typing import Any, Dict, List
from pydantic import BaseModel
import traceback

from langchain.schema import AIMessage
from langchain_openai import ChatOpenAI

from litellm import completion
import litellm

from arklex.utils.graph_state import MessageState
from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.orchestrator.prompts import RESPOND_ACTION_NAME


logger = logging.getLogger(__name__)

class Action(BaseModel):
    name: str
    kwargs: Dict[str, Any]

class EnvResponse(BaseModel):
    observation: Any

logger = logging.getLogger(__name__)

class FunctionCallingPlanner:
    
    description = "Default worker decided by chat records if there is no specific worker for the user's query"

    def __init__(self,
        tools_map: Dict[str, Any],
        name2id: Dict[str, int]):
        super().__init__()
        self.tools_map = tools_map
        self.tools_info = [tool["execute"]().info for tool in self.tools_map.values()]
        self.name2id = name2id

    def message_to_actions(
        self,
        message: Dict[str, Any],
    ) -> List[Action]:
        actions = []
        if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
            for tool_call in message["tool_calls"]:
                actions.append(Action(
                    name=tool_call["function"]["name"],
                    kwargs=json.loads(tool_call["function"]["arguments"]),
                ))
            return actions
        else:
            return [Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})]

    def plan(self, state: MessageState, msg_history, max_num_steps=3):
        user_message = state.user_message.message
        task = state.orchestrator_message.attribute.get("task", "")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": state.sys_instruct + "Your current task is: " + task},
        ]
        messages.extend(msg_history)
        llm_config = state.bot_config.llm_config
        llm = PROVIDER_MAP.get(llm_config.llm_provider, ChatOpenAI)(
            model=llm_config.model_type_or_path,
            temperature = 0.0
        )

        for _ in range(max_num_steps):
            logger.info(f"messages in function calling: {json.dumps(messages)}")
            logger.info(f"tools_info in function calling: {self.tools_info}")
            litellm.modify_params = True
            if not self.tools_info:
                res = llm.invoke(messages)
                next_message = aimessage_to_dict(res)             
            else:
                res = completion(
                    messages=messages,
                    model=llm_config.model_type_or_path,
                    custom_llm_provider=llm_config.llm_provider,
                    tools= convert_to_gemini_tools(self.tools_info) if llm_config.llm_provider == 'gemini' else self.tools_info,
                    temperature=0.0
                )
                next_message = res.choices[0].message.model_dump()
            actions = self.message_to_actions(next_message)
            messages.append(next_message)
            msg_history.append(next_message)
            logger.info("===============Function calling actions=====================")
            logger.info(actions)
            for idx, action in enumerate(actions):
                env_response = self.step(action)
                if action.name != RESPOND_ACTION_NAME:
                    messages.extend(
                        [
                            {
                                "role": "tool",
                                "tool_call_id": next_message["tool_calls"][idx]["id"],
                                "name": next_message["tool_calls"][idx]["function"]["name"],
                                "content": env_response.observation,
                            }
                        ]
                    )
                    msg_history.extend(
                        [
                            {
                                "role": "tool",
                                "tool_call_id": next_message["tool_calls"][idx]["id"],
                                "name": next_message["tool_calls"][idx]["function"]["name"],
                                "content": env_response.observation,
                            }
                        ]
                    )
                else:
                    return msg_history, action.name, env_response.observation
        return msg_history, action.name, env_response.observation
        
    
    def step(self, action: Action) -> EnvResponse:
        if action.name == RESPOND_ACTION_NAME:
            response = action.kwargs["content"]
            observation = response
        elif self.name2id[action.name] in self.tools_map:
            try:
                calling_tool = self.tools_map[self.name2id[action.name]]
                kwargs = action.kwargs
                combined_kwargs = {**kwargs, **calling_tool["fixed_args"]}
                observation = calling_tool["execute"]().func(**combined_kwargs)
                if not isinstance(observation, str):
                    # Convert to string if not already
                    observation = str(observation)
            except Exception as e:
                logger.error(traceback.format_exc())
                observation = f"Error: {e}"
        else:
            observation = f"Unknown action {action.name}"

        return EnvResponse(observation=observation)
    
    def execute(self, msg_state: MessageState, msg_history):
        # TODO: fix the logic for the planner
        # msg_history, action, response = self.plan(msg_state, msg_history)
        # msg_state.response = response
        # return action, msg_state, msg_history
        return None, msg_state, msg_history


def convert_to_gemini_tools(tools):
    converted_tools = {
        "tools": [
            {
                "function_declarations": []
            }
        ]
    }
    for tool in tools:
        if 'function' in tool:
            converted_tool = {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "parameters": tool["function"]["parameters"]
            }
            converted_tools["tools"][0]["function_declarations"].append(converted_tool)
        else:
            converted_tools["tools"].append(tool)

    return converted_tools

def aimessage_to_dict(ai_message):
    message_dict = {
        "content": ai_message.content,
        "role": "assistant" if isinstance(ai_message, AIMessage) else "user", 
        "function_call": None, 
        "tool_calls": None, 
    }
    return message_dict