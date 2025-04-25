from pydantic import BaseModel
from typing import  Optional, Dict


class Resource(BaseModel):
    id: str
    name: Optional[str] = None
    path: Optional[str] = None
    fixed_args: Optional[Dict] = dict()

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "fixed_args": dict(self.fixed_args) if self.fixed_args else {},
        }