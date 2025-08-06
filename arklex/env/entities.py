from pydantic import BaseModel


class NodeResponse(BaseModel):
    """Response for a node."""

    response: str
    choice_list: list[str] | None = None
