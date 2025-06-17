from datetime import datetime


class Transcript:
    def __init__(self, id: str, text: str, origin: str, created_at: datetime) -> None:
        self.id = id
        self.text = text
        self.origin = origin
        self.created_at = created_at
