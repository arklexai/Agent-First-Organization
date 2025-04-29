from pydantic import BaseModel
from typing import  Optional, Dict
from enum import Enum

class ResourceType(str, Enum):
    WORKER = "worker"
    TOOL = "tool"


class Resource(BaseModel):
    id: str
    name: Optional[str] = None
    path: Optional[str] = None
    type: Optional[ResourceType] = None
    fixed_args: Optional[Dict] = dict()

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "type": self.type,
            "fixed_args": dict(self.fixed_args) if self.fixed_args else {},
        }