from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, TypeVar, Union

T = TypeVar("T")


class ModelError(Exception):
    def __init__(
        self,
        short_message: str,
        prompt: Optional[Union[str, List[Dict[str, str]]]] = None,
        response: Optional[str] = None,
    ) -> None:
        super().__init__(short_message)
        self.short_message: str = short_message
        self.prompt: Optional[Union[str, List[Dict[str, str]]]] = prompt
        self.response: Optional[str] = response


@dataclass
class Result(Generic[T]):
    value: Optional[T]
    error: Optional[ModelError]
