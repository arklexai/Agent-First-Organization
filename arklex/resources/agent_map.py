import importlib
import logging
from collections.abc import Mapping

from arklex.resources.resource_types import (
    AgentCategory,
    AgentItem,
    Item,
    ResourceType,
)

log_context = logging.getLogger(__name__)

agent_map: Mapping[type[Item], Mapping[str, ResourceType | AgentCategory | type]] = {
    AgentItem.OPENAI_AGENT.value: {
        "type": ResourceType.AGENT,
        "category": AgentCategory.OPENAI,
        "module": "arklex.resources.agents.llm_based_agent.openai_agent",
        "item_cls": "OpenAIAgent",
    },
    AgentItem.OPENAI_REALTIME_VOICE_AGENT.value: {
        "type": ResourceType.AGENT,
        "category": AgentCategory.OPENAI,
        "module": "arklex.resources.agents.realtime_voice_agent.openai_realtime_agent",
        "item_cls": "OpenAIRealtimeAgent",
    },
    AgentItem.NLU_AGENT.value: {
        "type": ResourceType.AGENT,
        "category": AgentCategory.NLU,
        "module": "arklex.resources.agents.rule_based_agent.nlu_agent",
        "item_cls": "NLUAgent",
    },
}

AGENT_MAP = {}
for item, details in agent_map.items():
    function_name = details["item_cls"]
    module_path = details["module"]
    try:
        module = importlib.import_module(module_path)
        function = getattr(module, function_name)
        details["item_cls"] = function
        AGENT_MAP[item] = details
        log_context.info(f"Successfully imported {function_name} from {module_path}")
    except Exception as e:
        log_context.error(f"Failed to import {function_name} from {module_path}: {e}")
