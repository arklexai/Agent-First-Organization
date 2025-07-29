from abc import ABC

from pydantic import BaseModel


class WorkerOutput(ABC, BaseModel):
    """Base class for worker response."""
