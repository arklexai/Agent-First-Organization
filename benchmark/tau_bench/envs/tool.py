import abc
from typing import Any, NoReturn


class Tool(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def invoke(*args: object, **kwargs: object) -> NoReturn: ...

    @staticmethod
    @abc.abstractmethod
    def get_info() -> dict[str, Any]: ...
