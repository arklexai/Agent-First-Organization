from pydantic import BaseModel


class PromptVariable(BaseModel):
    """
    Prompt variable is a variable that is used in the prompt. e.g. Say hello {{name}}
    """

    name: str
    value: str = ""
