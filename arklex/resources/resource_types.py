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
    TWILIO = "twilio"
    CUSTOM = "custom"


class WorkerCategory(str, Enum):
    """Worker category enum for organizing different types of workers."""

    WORKER = "worker"


class AgentCategory(str, Enum):
    """Agent category enum for organizing different types of agents."""

    OPENAI = "openai"
    NLU = "nlu"


class Item(str, Enum):
    """Item enum for organizing different types of items."""


class ToolItem(Item):
    """Specific tool items organized by category."""

    # Google Calendar Tools
    GOOGLE_CREATE_EVENT = "google/calendar/create-event"
    GOOGLE_FREE_BUSY = "google/calendar/free-busy"

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
    HUBSPOT_BOOK_MEETING = "hubspot/meeting/book-meeting"
    HUBSPOT_CHECK_AVAILABILITY = "hubspot/meeting/check-availability"
    HUBSPOT_CHECK_AVAILABLE = "hubspot/meeting/check-available"
    HUBSPOT_CREATE_MEETING = "hubspot/meeting/create-meeting"
    HUBSPOT_CREATE_TICKET = "hubspot/ticket/create-ticket"
    HUBSPOT_FIND_CONTACT_BY_EMAIL = "hubspot/contact/find-contact-by-email"
    HUBSPOT_FIND_OWNER_ID_BY_CONTACT_ID = "hubspot/contact/find-owner-id-by-contact-id"

    # Twilio Tools
    TWILIO_SMS_SEND_SMS = "twilio/sms/send-sms"
    TWILIO_SMS_SEND_PREDEFINED_SMS = "twilio/sms/send-predefined-sms"
    TWILIO_CALL_END_CALL = "twilio/call/end-call"
    TWILIO_CALL_VOICEMAIL = "twilio/call/voicemail"
    TWILIO_CALL_TRANSFER = "twilio/call/transfer"

    # Custom Tools
    HTTP_TOOL = "custom/http-tool"
    RETRIEVER = "custom/retriever"


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
    OUTPUT_PROCESS_WORKER = "output-process-worker"


class AgentItem(Item):
    """Specific agent items organized by category."""

    OPENAI_AGENT = "openai-agent"
    OPENAI_REALTIME_VOICE_AGENT = "openai-realtime-voice-agent"
    NLU_AGENT = "nlu-agent"
    INPUT_GUARDRAIL = "input-guardrail"
    OUTPUT_GUARDRAIL = "output-guardrail"
