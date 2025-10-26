from datetime import datetime
from enum import Enum


class Transcript:
    def __init__(self, id: str, text: str, origin: str, created_at: datetime) -> None:
        self.id = id
        self.text = text
        self.origin = origin
        self.created_at = created_at


# Enum for chat roles
class ChatRole(str, Enum):
    BOT = "bot"
    ASSISTANT = "assistant"
    USER = "user"
    BOT_FOLLOWUP = "bot_followup"
    HUMAN_AGENT = "human_agent"
    TOOL = "tool"

    def from_str(role: str) -> "ChatRole":
        if role == "bot":
            return ChatRole.BOT
        elif role == "assistant":
            return ChatRole.ASSISTANT
        elif role == "user":
            return ChatRole.USER
        elif role == "bot_followup":
            return ChatRole.BOT_FOLLOWUP
        elif role == "human_agent":
            return ChatRole.HUMAN_AGENT
        elif role == "tool":
            return ChatRole.TOOL
        else:
            raise ValueError(f"Invalid role: {role}")


class ResourceAuthGroup(int, Enum):
    """when adding new auth group, add it also in the backend sql script"""

    PUBLIC = -1
    GOOGLE_CALENDAR = 0
    SHOPIFY = 1
    HUBSPOT = 2
    TWILIO = 3
    SALESFORCE = 4
