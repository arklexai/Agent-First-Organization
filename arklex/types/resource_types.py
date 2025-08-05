from enum import Enum


class ResourceType(str, Enum):
    """Resource type enum."""

    TOOL = "tool"
    WORKER = "worker"
    AGENT = "agent"


class ToolCategory(str, Enum):
    """Tool category enum for organizing different types of tools."""

    GOOGLE_CALENDAR = "google-calendar"
    SHOPIFY = "shopify"
    HUBSPOT = "hubspot"
    CUSTOM = "custom"


class WorkerCategory(str, Enum):
    """Worker category enum for organizing different types of workers."""

    WORKER = "worker"


class AgentCategory(str, Enum):
    """Agent category enum for organizing different types of agents."""

    OPENAI = "openai"


class Item(str, Enum):
    """Item enum for organizing different types of items."""


class ToolItem(Item):
    """Specific tool items organized by category."""

    # Google Calendar Tools
    GOOGLE_CREATE_EVENT = "google/calendar/create-event"

    # Shopify Tools
    SHOPIFY_FIND_USER_ID_BY_EMAIL = "shopify/find-user-id-by-email"
    SHOPIFY_GET_ORDER_DETAILS = "shopify/get-order-details"
    SHOPIFY_GET_PRODUCTS = "shopify/get-products"
    SHOPIFY_SEARCH_PRODUCTS = "shopify/search-products"
    SHOPIFY_CANCEL_ORDER = "shopify/cancel-order"
    SHOPIFY_CART_ADD_ITEMS = "shopify/cart-add-items"
    SHOPIFY_GET_CART = "shopify/get-cart"
    SHOPIFY_GET_USER_DETAILS_ADMIN = "shopify/get-user-details-admin"
    SHOPIFY_GET_WEB_PRODUCT = "shopify/get-web-product"
    SHOPIFY_RETURN_PRODUCTS = "shopify/return-products"

    # HubSpot Tools
    HUBSPOT_CHECK_AVAILABLE = "hubspot/check-available"
    HUBSPOT_CHECK_AVAILABILITY = "hubspot/check-availability"
    HUBSPOT_CREATE_MEETING = "hubspot/create-meeting"
    HUBSPOT_CREATE_TICKET = "hubspot/create-ticket"
    HUBSPOT_FIND_CONTACT_BY_EMAIL = "hubspot/find-contact-by-email"
    HUBSPOT_FIND_OWNER_ID_BY_CONTACT_ID = "hubspot/find-owner-id-by-contact-id"

    # Custom Tools
    HTTP_TOOL = "custom-tools/http-tool"


class WorkerItem(Item):
    """Specific worker items organized by category."""

    # Workers
    MESSAGE_WORKER = "message-worker"
    FAISS_RAG_WORKER = "faiss-rag-worker"
    MILVUS_RAG_WORKER = "milvus-rag-worker"
    MULTIPLE_CHOICE_WORKER = "multiple-choice-worker"
    RAG_MESSAGE_WORKER = "rag-message-worker"
    SEARCH_WORKER = "search-worker"
    HUMAN_IN_THE_LOOP_WORKER = "human-in-the-loop-worker"


class AgentItem(Item):
    """Specific agent items organized by category."""

    OPENAI_AGENT = "openai-agent"
    OPENAI_REALTIME_AGENT = "openai-realtime-agent"
