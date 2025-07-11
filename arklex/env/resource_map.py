from collections.abc import Mapping

from arklex.env.tools.google.calendar import create_event
from arklex.env.tools.hubspot import (
    check_availability,
    check_available,
    create_meeting,
    create_ticket,
    find_contact_by_email,
    find_owner_id_by_contact_id,
)
from arklex.env.tools.shopify import (
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
from arklex.env.workers import (
    FaissRAGWorker,
    HITLWorkerChatFlag,
    MessageWorker,
    MilvusRAGWorker,
    RagMsgWorker,
    SearchWorker,
)
from arklex.types.resource_types import (
    Item,
    Resource,
    ResourceType,
    ToolCategory,
    ToolItem,
    WorkerCategory,
    WorkerItem,
)

RESOURCE_MAP: Mapping[type[Item], Resource] = {
    ToolItem.GOOGLE_CREATE_EVENT: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.GOOGLE_CALENDAR,
        item_cls=create_event,
    ),
    ToolItem.SHOPIFY_FIND_USER_ID_BY_EMAIL: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.SHOPIFY,
        item_cls=find_user_id_by_email,
    ),
    ToolItem.SHOPIFY_GET_ORDER_DETAILS: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.SHOPIFY,
        item_cls=get_order_details,
    ),
    ToolItem.SHOPIFY_SEARCH_PRODUCTS: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.SHOPIFY,
        item_cls=search_products,
    ),
    ToolItem.SHOPIFY_CANCEL_ORDER: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.SHOPIFY,
        item_cls=cancel_order,
    ),
    ToolItem.SHOPIFY_CART_ADD_ITEMS: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.SHOPIFY,
        item_cls=cart_add_items,
    ),
    ToolItem.SHOPIFY_GET_CART: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.SHOPIFY,
        item_cls=get_cart,
    ),
    ToolItem.SHOPIFY_GET_USER_DETAILS_ADMIN: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.SHOPIFY,
        item_cls=get_user_details_admin,
    ),
    ToolItem.SHOPIFY_GET_WEB_PRODUCT: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.SHOPIFY,
        item_cls=get_web_product,
    ),
    ToolItem.SHOPIFY_RETURN_PRODUCTS: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.SHOPIFY,
        item_cls=return_products,
    ),
    ToolItem.HUBSPOT_CHECK_AVAILABLE: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.HUBSPOT,
        item_cls=check_available,
    ),
    ToolItem.HUBSPOT_CHECK_AVAILABILITY: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.HUBSPOT,
        item_cls=check_availability,
    ),
    ToolItem.HUBSPOT_CREATE_MEETING: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.HUBSPOT,
        item_cls=create_meeting,
    ),
    ToolItem.HUBSPOT_CREATE_TICKET: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.HUBSPOT,
        item_cls=create_ticket,
    ),
    ToolItem.HUBSPOT_FIND_CONTACT_BY_EMAIL: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.HUBSPOT,
        item_cls=find_contact_by_email,
    ),
    ToolItem.HUBSPOT_FIND_OWNER_ID_BY_CONTACT_ID: Resource(
        type=ResourceType.TOOL,
        category=ToolCategory.HUBSPOT,
        item_cls=find_owner_id_by_contact_id,
    ),
    WorkerItem.MESSAGE_WORKER: Resource(
        type=ResourceType.WORKER,
        category=WorkerCategory.MESSAGE_WORKER,
        item_cls=MessageWorker,
    ),
    WorkerItem.FAISS_RAG_WORKER: Resource(
        type=ResourceType.WORKER,
        category=WorkerCategory.FAISS_RAG_WORKER,
        item_cls=FaissRAGWorker,
    ),
    WorkerItem.MILVUS_RAG_WORKER: Resource(
        type=ResourceType.WORKER,
        category=WorkerCategory.MILVUS_RAG_WORKER,
        item_cls=MilvusRAGWorker,
    ),
    WorkerItem.RAG_MESSAGE_WORKER: Resource(
        type=ResourceType.WORKER,
        category=WorkerCategory.RAG_MESSAGE_WORKER,
        item_cls=RagMsgWorker,
    ),
    WorkerItem.SEARCH_WORKER: Resource(
        type=ResourceType.WORKER,
        category=WorkerCategory.SEARCH_WORKER,
        item_cls=SearchWorker,
    ),
    WorkerItem.HUMAN_IN_THE_LOOP_WORKER: Resource(
        type=ResourceType.WORKER,
        category=WorkerCategory.HUMAN_IN_THE_LOOP_WORKER,
        item_cls=HITLWorkerChatFlag,
    ),
}
