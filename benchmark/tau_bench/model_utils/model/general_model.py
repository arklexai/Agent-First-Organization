import abc
from typing import Any, TypeVar, List, Union, Optional, Dict, Type

from pydantic import BaseModel

from benchmark.tau_bench.model_utils.api.datapoint import (
    BinaryClassifyDatapoint,
    ClassifyDatapoint,
    GenerateDatapoint,
    ParseDatapoint,
    ParseForceDatapoint,
    ScoreDatapoint,
)
from benchmark.tau_bench.model_utils.api.types import PartialObj
from benchmark.tau_bench.model_utils.model.model import (
    BinaryClassifyModel,
    ClassifyModel,
    GenerateModel,
    ParseForceModel,
    ParseModel,
    Platform,
    ScoreModel,
)

T = TypeVar("T", bound=BaseModel)

LLM_SAMPLING_TEMPERATURE_EPS: float = 1e-5


def wrap_temperature(temperature: float) -> float:
    return max(temperature, LLM_SAMPLING_TEMPERATURE_EPS)


class GeneralModel(
    ClassifyModel,
    BinaryClassifyModel,
    ParseModel,
    GenerateModel,
    ParseForceModel,
    ScoreModel,
):
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

    def binary_classify(
        self,
        instruction: str,
        text: str,
        examples: Optional[List[BinaryClassifyDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> bool:
        return (
            self.classify(
                instruction,
                text,
                ["true", "false"],
                examples=(
                    None
                    if examples is None
                    else [
                        ClassifyDatapoint(
                            instruction=example.instruction,
                            text=example.text,
                            options=["true", "false"],
                            response=0 if example.response else 1,
                        )
                        for example in examples
                    ]
                ),
                temperature=temperature,
            )
            == 0
        )

    @abc.abstractmethod
    def parse(
        self,
        text: str,
        typ: Union[Type[T], Dict[str, Any]],
        examples: Optional[List[ParseDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> Union[T, PartialObj, Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def generate(
        self,
        instruction: str,
        text: str,
        examples: Optional[List[GenerateDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        raise NotImplementedError

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


def default_model() -> GeneralModel:
    from benchmark.tau_bench.model_utils.model.openai import OpenAIModel

    return OpenAIModel()


def default_quick_model() -> GeneralModel:
    from benchmark.tau_bench.model_utils.model.openai import OpenAIModel

    return OpenAIModel(model="gpt-4o-mini")


def model_factory(
    model_id: str,
    platform: Union[str, Platform],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> GeneralModel:
    if isinstance(platform, str):
        platform = Platform(platform)
    if platform == Platform.OPENAI:
        from benchmark.tau_bench.model_utils.model.openai import OpenAIModel

        return OpenAIModel(model=model_id, api_key=api_key, temperature=temperature)
    elif platform == Platform.MISTRAL:
        from benchmark.tau_bench.model_utils.model.mistral import MistralModel

        return MistralModel(model=model_id, api_key=api_key, temperature=temperature)
    elif platform == Platform.ANTHROPIC:
        from benchmark.tau_bench.model_utils.model.claude import ClaudeModel

        return ClaudeModel(model=model_id, api_key=api_key, temperature=temperature)

    elif platform == Platform.ANYSCALE:
        from benchmark.tau_bench.model_utils.model.anyscale import AnyscaleModel

        return AnyscaleModel(model=model_id, api_key=api_key, temperature=temperature)
    elif platform == Platform.OUTLINES:
        if base_url is None:
            raise ValueError("base_url must be provided for custom models")
        from benchmark.tau_bench.model_utils.model.outlines_completion import (
            OutlinesCompletionModel,
        )

        return OutlinesCompletionModel(
            model=model_id, base_url=base_url, temperature=temperature
        )
    elif platform == Platform.VLLM_CHAT:
        if base_url is None:
            raise ValueError("base_url must be provided for custom models")
        from benchmark.tau_bench.model_utils.model.vllm_chat import VLLMChatModel

        return VLLMChatModel(
            model=model_id,
            base_url=base_url,
            api_key="sk-no-api-key-required" if api_key is None else api_key,
            temperature=temperature,
        )
    else:
        if base_url is None:
            raise ValueError("base_url must be provided for custom models")
        from benchmark.tau_bench.model_utils.model.vllm_completion import (
            VLLMCompletionModel,
        )

        return VLLMCompletionModel(
            model=model_id, base_url=base_url, temperature=temperature
        )
