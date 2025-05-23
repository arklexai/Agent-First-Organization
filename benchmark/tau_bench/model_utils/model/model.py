import abc
import enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from benchmark.tau_bench.model_utils.api.datapoint import (
    BinaryClassifyDatapoint,
    ClassifyDatapoint,
    Datapoint,
    GenerateDatapoint,
    ParseDatapoint,
    ParseForceDatapoint,
    ScoreDatapoint,
)
from benchmark.tau_bench.model_utils.api.types import PartialObj

T = TypeVar("T", bound=BaseModel)


class Platform(enum.Enum):
    OPENAI = "openai"
    MISTRAL = "mistral"
    ANTHROPIC = "anthropic"
    ANYSCALE = "anyscale"
    OUTLINES = "outlines"
    VLLM_CHAT = "vllm-chat"
    VLLM_COMPLETION = "vllm-completion"


# @runtime_checkable
# class Model(Protocol):
class Model(abc.ABC):
    @abc.abstractmethod
    def get_capability(self) -> float:
        """Return the capability of the model, a float between 0.0 and 1.0."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_approx_cost(self, dp: Datapoint) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def get_latency(self, dp: Datapoint) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def supports_dp(self, dp: Datapoint) -> bool:
        raise NotImplementedError


class ClassifyModel(Model):
    @abc.abstractmethod
    def classify(
        self,
        instruction: str,
        text: str,
        options: List[str],
        examples: Optional[List[ClassifyDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> int:
        raise NotImplementedError


class BinaryClassifyModel(Model):
    @abc.abstractmethod
    def binary_classify(
        self,
        instruction: str,
        text: str,
        examples: Optional[List[BinaryClassifyDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> bool:
        raise NotImplementedError


class ParseModel(Model):
    @abc.abstractmethod
    def parse(
        self,
        text: str,
        typ: Union[Type[T], Dict[str, Any]],
        examples: Optional[List[ParseDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> Union[T, PartialObj, Dict[str, Any]]:
        raise NotImplementedError


class GenerateModel(Model):
    @abc.abstractmethod
    def generate(
        self,
        instruction: str,
        text: str,
        examples: Optional[List[GenerateDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        raise NotImplementedError


class ParseForceModel(Model):
    @abc.abstractmethod
    def parse_force(
        self,
        instruction: str,
        typ: Union[Type[T], Dict[str, Any]],
        text: Optional[str] = None,
        examples: Optional[List[ParseForceDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> Union[T, Dict[str, Any]]:
        raise NotImplementedError


class ScoreModel(Model):
    @abc.abstractmethod
    def score(
        self,
        instruction: str,
        text: str,
        min: int,
        max: int,
        examples: Optional[List[ScoreDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> int:
        raise NotImplementedError


AnyModel = Union[
    BinaryClassifyModel,
    ClassifyModel,
    ParseForceModel,
    GenerateModel,
    ParseModel,
    ScoreModel,
]
