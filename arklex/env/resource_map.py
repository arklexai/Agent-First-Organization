import importlib
import logging
from collections.abc import Mapping

from arklex.types.resource_types import (
    AgentCategory,
    AgentItem,
    Item,
    ResourceType,
    ToolCategory,
    ToolItem,
    WorkerCategory,
    WorkerItem,
)

log_context = logging.getLogger(__name__)

resource_map: Mapping[type[Item], Mapping[str, ResourceType | ToolCategory | type]] = {
    ToolItem.GOOGLE_CREATE_EVENT: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.GOOGLE_CALENDAR,
        "module": "arklex.env.tools.google.calendar.create_event",
        "item_cls": "create_event",
    },
    ToolItem.GOOGLE_FREE_BUSY: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.GOOGLE_CALENDAR,
        "module": "arklex.env.tools.google.calendar.free_busy",
        "item_cls": "free_busy",
    },
    ToolItem.SHOPIFY_FIND_USER_ID_BY_EMAIL: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.SHOPIFY,
        "module": "arklex.env.tools.shopify.find_user_id_by_email",
        "item_cls": "find_user_id_by_email",
    },
    ToolItem.SHOPIFY_GET_ORDER_DETAILS: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.SHOPIFY,
        "module": "arklex.env.tools.shopify.get_order_details",
        "item_cls": "get_order_details",
    },
    ToolItem.SHOPIFY_SEARCH_PRODUCTS: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.SHOPIFY,
        "module": "arklex.env.tools.shopify.search_products",
        "item_cls": "search_products",
    },
    ToolItem.SHOPIFY_CANCEL_ORDER: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.SHOPIFY,
        "module": "arklex.env.tools.shopify.cancel_order",
        "item_cls": "cancel_order",
    },
    ToolItem.SHOPIFY_CART_ADD_ITEMS: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.SHOPIFY,
        "module": "arklex.env.tools.shopify.cart_add_items",
        "item_cls": "cart_add_items",
    },
    ToolItem.SHOPIFY_GET_CART: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.SHOPIFY,
        "module": "arklex.env.tools.shopify.get_cart",
        "item_cls": "get_cart",
    },
    ToolItem.SHOPIFY_GET_USER_DETAILS_ADMIN: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.SHOPIFY,
        "module": "arklex.env.tools.shopify.get_user_details_admin",
        "item_cls": "get_user_details_admin",
    },
    ToolItem.SHOPIFY_GET_PRODUCTS: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.SHOPIFY,
        "module": "arklex.env.tools.shopify.get_products",
        "item_cls": "get_products",
    },
    ToolItem.SHOPIFY_GET_WEB_PRODUCT: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.SHOPIFY,
        "module": "arklex.env.tools.shopify.get_web_product",
        "item_cls": "get_web_product",
    },
    ToolItem.SHOPIFY_RETURN_PRODUCTS: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.SHOPIFY,
        "module": "arklex.env.tools.shopify.return_products",
        "item_cls": "return_products",
    },
    ToolItem.HUBSPOT_CHECK_AVAILABILITY: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.HUBSPOT,
        "module": "arklex.env.tools.hubspot.check_availability",
        "item_cls": "check_availability",
    },
    ToolItem.HUBSPOT_BOOK_MEETING: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.HUBSPOT,
        "module": "arklex.env.tools.hubspot.book_meeting",
        "item_cls": "book_meeting",
    },
    ToolItem.HUBSPOT_CHECK_AVAILABLE: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.HUBSPOT,
        "module": "arklex.env.tools.hubspot.check_available",
        "item_cls": "check_available",
    },
    ToolItem.HUBSPOT_CREATE_MEETING: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.HUBSPOT,
        "module": "arklex.env.tools.hubspot.create_meeting",
        "item_cls": "create_meeting",
    },
    ToolItem.HUBSPOT_CREATE_TICKET: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.HUBSPOT,
        "module": "arklex.env.tools.hubspot.create_ticket",
        "item_cls": "create_ticket",
    },
    ToolItem.HUBSPOT_FIND_CONTACT_BY_EMAIL: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.HUBSPOT,
        "module": "arklex.env.tools.hubspot.find_contact_by_email",
        "item_cls": "find_contact_by_email",
    },
    ToolItem.HUBSPOT_FIND_OWNER_ID_BY_CONTACT_ID: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.HUBSPOT,
        "module": "arklex.env.tools.hubspot.find_owner_id_by_contact_id",
        "item_cls": "find_owner_id_by_contact_id",
    },
    ToolItem.TWILIO_SMS_SEND_SMS: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.TWILIO,
        "module": "arklex.env.tools.twilio.sms.send_sms",
        "item_cls": "send_sms",
    },
    ToolItem.TWILIO_SMS_SEND_PREDEFINED_SMS: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.TWILIO,
        "module": "arklex.env.tools.twilio.sms.send_predefined_sms",
        "item_cls": "send_predefined_sms",
    },
    ToolItem.TWILIO_CALL_END_CALL: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.TWILIO,
        "module": "arklex.env.tools.twilio.calls.end_call",
        "item_cls": "end_call",
    },
    ToolItem.TWILIO_CALL_VOICEMAIL: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.TWILIO,
        "module": "arklex.env.tools.twilio.calls.voicemail",
        "item_cls": "voicemail",
    },
    ToolItem.TWILIO_CALL_TRANSFER: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.TWILIO,
        "module": "arklex.env.tools.twilio.calls.transfer",
        "item_cls": "transfer",
    },
    ToolItem.HTTP_TOOL: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.CUSTOM,
        "module": "arklex.env.tools.custom_tools.http_tool",
        "item_cls": "http_tool",
    },
    ToolItem.RETRIEVER: {
        "type": ResourceType.TOOL,
        "category": ToolCategory.CUSTOM,
        "module": "arklex.env.tools.custom_tools.retriever",
        "item_cls": "retriever",
    },
    WorkerItem.MESSAGE_WORKER: {
        "type": ResourceType.WORKER,
        "category": WorkerCategory.WORKER,
        "module": "arklex.env.workers.message.message_worker",
        "item_cls": "MessageWorker",
    },
    WorkerItem.FAISS_RAG_WORKER: {
        "type": ResourceType.WORKER,
        "category": WorkerCategory.WORKER,
        "module": "arklex.env.workers.faiss_rag.faiss_rag_worker",
        "item_cls": "FaissRAGWorker",
    },
    WorkerItem.MILVUS_RAG_WORKER: {
        "type": ResourceType.WORKER,
        "category": WorkerCategory.WORKER,
        "module": "arklex.env.workers.milvus_rag.milvus_rag_worker",
        "item_cls": "MilvusRAGWorker",
    },
    WorkerItem.RAG_MESSAGE_WORKER: {
        "type": ResourceType.WORKER,
        "category": WorkerCategory.WORKER,
        "module": "arklex.env.workers.rag_message.rag_message_worker",
        "item_cls": "RagMsgWorker",
    },
    WorkerItem.SEARCH_WORKER: {
        "type": ResourceType.WORKER,
        "category": WorkerCategory.WORKER,
        "module": "arklex.env.workers.search.search_worker",
        "item_cls": "SearchWorker",
    },
    WorkerItem.HUMAN_IN_THE_LOOP_WORKER: {
        "type": ResourceType.WORKER,
        "category": WorkerCategory.WORKER,
        "module": "arklex.env.workers.hitl.hitl_worker",
        "item_cls": "HITLWorker",
    },
    WorkerItem.MULTIPLE_CHOICE_WORKER: {
        "type": ResourceType.WORKER,
        "category": WorkerCategory.WORKER,
        "module": "arklex.env.workers.multiple_choice.multiple_choice_worker",
        "item_cls": "MultipleChoiceWorker",
    },
    WorkerItem.OUTPUT_PROCESS_WORKER: {
        "type": ResourceType.WORKER,
        "category": WorkerCategory.WORKER,
        "module": "arklex.env.workers.output_process.output_process_worker",
        "item_cls": "OutputProcessWorker",
    },
    AgentItem.OPENAI_AGENT: {
        "type": ResourceType.AGENT,
        "category": AgentCategory.OPENAI,
        "module": "arklex.env.agents.openai_agent",
        "item_cls": "OpenAIAgent",
    },
    AgentItem.OPENAI_REALTIME_VOICE_AGENT: {
        "type": ResourceType.AGENT,
        "category": AgentCategory.OPENAI,
        "module": "arklex.env.agents.openai_realtime_agent",
        "item_cls": "OpenAIRealtimeAgent",
    },
}

RESOURCE_MAP = {}
for item, details in resource_map.items():
    function_name = details["item_cls"]
    module_path = details["module"]
    try:
        module = importlib.import_module(module_path)
        function = getattr(module, function_name)
        details["item_cls"] = function
        RESOURCE_MAP[item] = details
        log_context.info(f"Successfully imported {function_name} from {module_path}")
    except Exception as e:
        log_context.error(f"Failed to import {function_name} from {module_path}: {e}")
