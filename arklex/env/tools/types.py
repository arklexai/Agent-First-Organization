from datetime import datetime
from enum import Enum


class Transcript:
    def __init__(self, id: str, text: str, origin: str, created_at: datetime) -> None:
        self.id = id
        self.text = text
        self.origin = origin
        self.created_at = created_at


class ResourceAuthGroup(int, Enum):
    """when adding new auth group, add it also in the backend sql script"""

    PUBLIC = -1
    GOOGLE_CALENDAR = 0
    SHOPIFY = 1
    HUBSPOT = 2
    TWILIO = 3
    SALESFORCE = 4
