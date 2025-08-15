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

# Initialize flags for successful imports
openai_agent_imported = False
custom_tools_imported = False
google_tools_imported = False
hubspot_tools_imported = False
shopify_tools_imported = False
twilio_tools_imported = False
workers_imported = False

# Attempt to import and set flags
try:
    from arklex.env.agents.agent_collection import OpenAIAgent, OpenAIRealtimeAgent

    openai_agent_imported = True
except ImportError:
    print("Failed to import OpenAIAgent or OpenAIRealtimeAgent")

try:
    from arklex.env.tools.custom_tools import http_tool, retriever

    custom_tools_imported = True
except ImportError:
    print("Failed to import http_tool or retriever")

try:
    from arklex.env.tools.google.calendar.tool_collection import create_event, free_busy

    google_tools_imported = True
except ImportError:
    print("Failed to import create_event or free_busy")

try:
    from arklex.env.tools.hubspot.tool_collection import (
        book_meeting,
        check_availability,
        check_available,
        create_meeting,
        create_ticket,
        find_contact_by_email,
        find_owner_id_by_contact_id,
    )

    hubspot_tools_imported = True
except ImportError:
    print("Failed to import HubSpot tools")

try:
    from arklex.env.tools.shopify.tool_collection import (
        cancel_order,
        cart_add_items,
        find_user_id_by_email,
        get_cart,
        get_order_details,
        get_user_details_admin,
        get_web_product,
        return_products,
        search_products,
    )

    shopify_tools_imported = True
except ImportError:
    print("Failed to import Shopify tools")

try:
    from arklex.env.tools.twilio.tool_collection import (
        end_call,
        send_predefined_sms,
        send_sms,
        voicemail,
    )

    twilio_tools_imported = True
except ImportError:
    print("Failed to import Twilio tools")

try:
    from arklex.env.workers.worker_collection import (
        FaissRAGWorker,
        HITLWorker,
        MessageWorker,
        MilvusRAGWorker,
        MultipleChoiceWorker,
        RagMsgWorker,
        SearchWorker,
    )

    workers_imported = True
except ImportError:
    print("Failed to import workers")


# Conditionally build RESOURCE_MAP
RESOURCE_MAP: Mapping[type[Item], Mapping[str, ResourceType | ToolCategory | type]] = {}

if google_tools_imported:
    RESOURCE_MAP.update(
        {
            ToolItem.GOOGLE_CREATE_EVENT: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.GOOGLE_CALENDAR,
                "item_cls": create_event,
            },
            ToolItem.GOOGLE_FREE_BUSY: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.GOOGLE_CALENDAR,
                "item_cls": free_busy,
            },
        }
    )

if shopify_tools_imported:
    RESOURCE_MAP.update(
        {
            ToolItem.SHOPIFY_FIND_USER_ID_BY_EMAIL: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.SHOPIFY,
                "item_cls": find_user_id_by_email,
            },
            ToolItem.SHOPIFY_GET_ORDER_DETAILS: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.SHOPIFY,
                "item_cls": get_order_details,
            },
            ToolItem.SHOPIFY_SEARCH_PRODUCTS: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.SHOPIFY,
                "item_cls": search_products,
            },
            ToolItem.SHOPIFY_CANCEL_ORDER: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.SHOPIFY,
                "item_cls": cancel_order,
            },
            ToolItem.SHOPIFY_CART_ADD_ITEMS: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.SHOPIFY,
                "item_cls": cart_add_items,
            },
            ToolItem.SHOPIFY_GET_CART: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.SHOPIFY,
                "item_cls": get_cart,
            },
            ToolItem.SHOPIFY_GET_USER_DETAILS_ADMIN: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.SHOPIFY,
                "item_cls": get_user_details_admin,
            },
            ToolItem.SHOPIFY_GET_WEB_PRODUCT: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.SHOPIFY,
                "item_cls": get_web_product,
            },
            ToolItem.SHOPIFY_RETURN_PRODUCTS: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.SHOPIFY,
                "item_cls": return_products,
            },
        }
    )

if hubspot_tools_imported:
    RESOURCE_MAP.update(
        {
            ToolItem.HUBSPOT_CHECK_AVAILABILITY: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.HUBSPOT,
                "item_cls": check_availability,
            },
            ToolItem.HUBSPOT_BOOK_MEETING: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.HUBSPOT,
                "item_cls": book_meeting,
            },
            ToolItem.HUBSPOT_CHECK_AVAILABLE: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.HUBSPOT,
                "item_cls": check_available,
            },
            ToolItem.HUBSPOT_CREATE_MEETING: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.HUBSPOT,
                "item_cls": create_meeting,
            },
            ToolItem.HUBSPOT_CREATE_TICKET: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.HUBSPOT,
                "item_cls": create_ticket,
            },
            ToolItem.HUBSPOT_FIND_CONTACT_BY_EMAIL: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.HUBSPOT,
                "item_cls": find_contact_by_email,
            },
            ToolItem.HUBSPOT_FIND_OWNER_ID_BY_CONTACT_ID: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.HUBSPOT,
                "item_cls": find_owner_id_by_contact_id,
            },
        }
    )

if twilio_tools_imported:
    RESOURCE_MAP.update(
        {
            ToolItem.TWILIO_SMS_SEND_SMS: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.TWILIO,
                "item_cls": send_sms,
            },
            ToolItem.TWILIO_SMS_SEND_PREDEFINED_SMS: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.TWILIO,
                "item_cls": send_predefined_sms,
            },
            ToolItem.TWILIO_CALL_END_CALL: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.TWILIO,
                "item_cls": end_call,
            },
            ToolItem.TWILIO_CALL_VOICEMAIL: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.TWILIO,
                "item_cls": voicemail,
            },
        }
    )

if custom_tools_imported:
    RESOURCE_MAP.update(
        {
            ToolItem.HTTP_TOOL: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.CUSTOM,
                "item_cls": http_tool,
            },
            ToolItem.RETRIEVER: {
                "type": ResourceType.TOOL,
                "category": ToolCategory.CUSTOM,
                "item_cls": retriever,
            },
        }
    )

if workers_imported:
    RESOURCE_MAP.update(
        {
            WorkerItem.MESSAGE_WORKER: {
                "type": ResourceType.WORKER,
                "category": WorkerCategory.WORKER,
                "item_cls": MessageWorker,
            },
            WorkerItem.FAISS_RAG_WORKER: {
                "type": ResourceType.WORKER,
                "category": WorkerCategory.WORKER,
                "item_cls": FaissRAGWorker,
            },
            WorkerItem.MILVUS_RAG_WORKER: {
                "type": ResourceType.WORKER,
                "category": WorkerCategory.WORKER,
                "item_cls": MilvusRAGWorker,
            },
            WorkerItem.RAG_MESSAGE_WORKER: {
                "type": ResourceType.WORKER,
                "category": WorkerCategory.WORKER,
                "item_cls": RagMsgWorker,
            },
            WorkerItem.SEARCH_WORKER: {
                "type": ResourceType.WORKER,
                "category": WorkerCategory.WORKER,
                "item_cls": SearchWorker,
            },
            WorkerItem.HUMAN_IN_THE_LOOP_WORKER: {
                "type": ResourceType.WORKER,
                "category": WorkerCategory.WORKER,
                "item_cls": HITLWorker,
            },
            WorkerItem.MULTIPLE_CHOICE_WORKER: {
                "type": ResourceType.WORKER,
                "category": WorkerCategory.WORKER,
                "item_cls": MultipleChoiceWorker,
            },
        }
    )

if openai_agent_imported:
    RESOURCE_MAP.update(
        {
            AgentItem.OPENAI_AGENT: {
                "type": ResourceType.AGENT,
                "category": AgentCategory.OPENAI,
                "item_cls": OpenAIAgent,
            },
            AgentItem.OPENAI_REALTIME_VOICE_AGENT: {
                "type": ResourceType.AGENT,
                "category": AgentCategory.OPENAI,
                "item_cls": OpenAIRealtimeAgent,
            },
        }
    )
