from __future__ import annotations

import argparse
from typing import Any, TypeVar, List, Optional, Union, Dict, Type, Callable

from pydantic import BaseModel

from benchmark.tau_bench.model_utils.api._model_methods import MODEL_METHODS
from benchmark.tau_bench.model_utils.api.cache import cache_call_w_dedup
from benchmark.tau_bench.model_utils.api.datapoint import (
    BinaryClassifyDatapoint,
    ClassifyDatapoint,
    Datapoint,
    GenerateDatapoint,
    ParseDatapoint,
    ParseForceDatapoint,
    ScoreDatapoint,
)
from benchmark.tau_bench.model_utils.api.logging import log_call
from benchmark.tau_bench.model_utils.api.router import (
    RequestRouter,
    default_request_router,
)
from benchmark.tau_bench.model_utils.api.sample import (
    EnsembleSamplingStrategy,
    MajoritySamplingStrategy,
    SamplingStrategy,
    get_default_sampling_strategy,
)
from benchmark.tau_bench.model_utils.api.types import PartialObj
from benchmark.tau_bench.model_utils.model.general_model import GeneralModel
from benchmark.tau_bench.model_utils.model.model import (
    AnyModel,
    BinaryClassifyModel,
    ClassifyModel,
    GenerateModel,
    ParseForceModel,
    ParseModel,
    ScoreModel,
)

T = TypeVar("T", bound=BaseModel)


class API(object):
    wrappers_for_main_methods: List[Callable] = [log_call, cache_call_w_dedup]

    def __init__(
        self,
        parse_models: List[ParseModel],
        generate_models: List[GenerateModel],
        parse_force_models: List[ParseForceModel],
        score_models: List[ScoreModel],
        classify_models: List[ClassifyModel],
        binary_classify_models: Optional[List[BinaryClassifyModel]] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        request_router: Optional[RequestRouter] = None,
        log_file: Optional[str] = None,
    ) -> None:
        if sampling_strategy is None:
            sampling_strategy = get_default_sampling_strategy()
        if request_router is None:
            request_router = default_request_router()
        self.sampling_strategy: SamplingStrategy = sampling_strategy
        self.request_router: RequestRouter = request_router
        self._log_file: Optional[str] = log_file
        self.binary_classify_models: Optional[List[BinaryClassifyModel]] = (
            binary_classify_models
        )
        self.classify_models: List[ClassifyModel] = classify_models
        self.parse_models: List[ParseModel] = parse_models
        self.generate_models: List[GenerateModel] = generate_models
        self.parse_force_models: List[ParseForceModel] = parse_force_models
        self.score_models: List[ScoreModel] = score_models

        self.__init_subclass__()

    def __init_subclass__(cls) -> None:
        for method_name in MODEL_METHODS:
            if hasattr(cls, method_name):
                method = getattr(cls, method_name)
                for wrapper in cls.wrappers_for_main_methods:
                    method = wrapper(method)
                setattr(cls, method_name, method)

    @classmethod
    def from_general_model(
        cls,
        model: GeneralModel,
        sampling_strategy: Optional[SamplingStrategy] = None,
        request_router: Optional[RequestRouter] = None,
        log_file: Optional[str] = None,
    ) -> "API":
        return cls(
            binary_classify_models=[model],
            classify_models=[model],
            parse_models=[model],
            generate_models=[model],
            parse_force_models=[model],
            score_models=[model],
            log_file=log_file,
            sampling_strategy=sampling_strategy,
            request_router=request_router,
        )

    @classmethod
    def from_general_models(
        cls,
        models: List[GeneralModel],
        sampling_strategy: Optional[SamplingStrategy] = None,
        request_router: Optional[RequestRouter] = None,
        log_file: Optional[str] = None,
    ) -> "API":
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        return cls(
            binary_classify_models=models,
            classify_models=models,
            parse_models=models,
            generate_models=models,
            parse_force_models=models,
            score_models=models,
            log_file=log_file,
            sampling_strategy=sampling_strategy,
            request_router=request_router,
        )

    def set_default_binary_classify_models(
        self, models: List[BinaryClassifyModel]
    ) -> None:
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        self.binary_classify_models = models

    def set_default_classify_models(self, models: List[ClassifyModel]) -> None:
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        self.classify_models = models

    def set_default_parse_models(self, models: List[ParseModel]) -> None:
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        self.parse_models = models

    def set_default_generate_models(self, models: List[GenerateModel]) -> None:
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        self.generate_models = models

    def set_default_parse_force_models(self, models: List[ParseForceModel]) -> None:
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        self.parse_force_models = models

    def set_default_score_models(self, models: List[ScoreModel]) -> None:
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        self.score_models = models

    def set_default_sampling_strategy(
        self, sampling_strategy: SamplingStrategy
    ) -> None:
        self.sampling_strategy = sampling_strategy

    def set_default_request_router(self, request_router: RequestRouter) -> None:
        self.request_router = request_router

    def _run_with_sampling_strategy(
        self,
        models: List[AnyModel],
        datapoint: Datapoint,
        sampling_strategy: SamplingStrategy,
    ) -> T:
        assert len(models) > 0

        def _run_datapoint(model: AnyModel, temp: Optional[float] = None) -> T:
            if isinstance(datapoint, ClassifyDatapoint):
                return model.classify(
                    instruction=datapoint.instruction,
                    text=datapoint.text,
                    options=datapoint.options,
                    examples=datapoint.examples,
                    temperature=temp,
                )
            elif isinstance(datapoint, BinaryClassifyDatapoint):
                return model.binary_classify(
                    instruction=datapoint.instruction,
                    text=datapoint.text,
                    examples=datapoint.examples,
                    temperature=temp,
                )
            elif isinstance(datapoint, ParseForceDatapoint):
                return model.parse_force(
                    instruction=datapoint.instruction,
                    typ=datapoint.typ,
                    text=datapoint.text,
                    examples=datapoint.examples,
                    temperature=temp,
                )
            elif isinstance(datapoint, GenerateDatapoint):
                return model.generate(
                    instruction=datapoint.instruction,
                    text=datapoint.text,
                    examples=datapoint.examples,
                    temperature=temp,
                )
            elif isinstance(datapoint, ParseDatapoint):
                return model.parse(
                    text=datapoint.text,
                    typ=datapoint.typ,
                    examples=datapoint.examples,
                    temperature=temp,
                )
            elif isinstance(datapoint, ScoreDatapoint):
                return model.score(
                    instruction=datapoint.instruction,
                    text=datapoint.text,
                    min=datapoint.min,
                    max=datapoint.max,
                    examples=datapoint.examples,
                    temperature=temp,
                )
            else:
                raise ValueError(f"Unknown datapoint type: {type(datapoint)}")

        if isinstance(sampling_strategy, EnsembleSamplingStrategy):
            return sampling_strategy.execute(
                [lambda x=model: _run_datapoint(x, 0.0) for model in models]
            )
        return sampling_strategy.execute(
            lambda: _run_datapoint(
                models[0],
                0.2
                if isinstance(sampling_strategy, MajoritySamplingStrategy)
                else None,
            )
        )

    def _api_call(
        self,
        models: List[AnyModel],
        datapoint: Datapoint,
        sampling_strategy: SamplingStrategy,
    ) -> T:
        if isinstance(sampling_strategy, EnsembleSamplingStrategy):
            return self._run_with_sampling_strategy(
                models, datapoint, sampling_strategy
            )
        model = self.request_router.route(dp=datapoint, available_models=models)
        return self._run_with_sampling_strategy([model], datapoint, sampling_strategy)

    def classify(
        self,
        instruction: str,
        text: str,
        options: List[str],
        examples: Optional[List[ClassifyDatapoint]] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        request_router: Optional[RequestRouter] = None,
        models: Optional[List[ClassifyModel]] = None,
    ) -> int:
        if models is None:
            models = self.classify_models
        if sampling_strategy is None:
            sampling_strategy = self.sampling_strategy
        if request_router is None:
            request_router = self.request_router
        return self._api_call(
            models,
            ClassifyDatapoint(
                instruction=instruction,
                text=text,
                options=options,
                examples=examples,
            ),
            sampling_strategy,
        )

    def binary_classify(
        self,
        instruction: str,
        text: str,
        examples: Optional[List[BinaryClassifyDatapoint]] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        request_router: Optional[RequestRouter] = None,
        models: Optional[List[BinaryClassifyModel]] = None,
    ) -> bool:
        if models is None:
            models = self.binary_classify_models
        if sampling_strategy is None:
            sampling_strategy = self.sampling_strategy
        if request_router is None:
            request_router = self.request_router
        return self._api_call(
            models,
            BinaryClassifyDatapoint(
                instruction=instruction,
                text=text,
                examples=examples,
            ),
            sampling_strategy,
        )

    def parse(
        self,
        text: str,
        typ: Union[Type[T], Dict[str, Any]],
        examples: Optional[List[ParseDatapoint]] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        request_router: Optional[RequestRouter] = None,
        models: Optional[List[ParseModel]] = None,
    ) -> Union[T, PartialObj, Dict[str, Any]]:
        if models is None:
            models = self.parse_models
        if sampling_strategy is None:
            sampling_strategy = self.sampling_strategy
        if request_router is None:
            request_router = self.request_router
        return self._api_call(
            models,
            ParseDatapoint(
                text=text,
                typ=typ,
                examples=examples,
            ),
            sampling_strategy,
        )

    def generate(
        self,
        instruction: str,
        text: str,
        examples: Optional[List[GenerateDatapoint]] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        request_router: Optional[RequestRouter] = None,
        models: Optional[List[GenerateModel]] = None,
    ) -> str:
        if models is None:
            models = self.generate_models
        if sampling_strategy is None:
            sampling_strategy = self.sampling_strategy
        if request_router is None:
            request_router = self.request_router
        return self._api_call(
            models,
            GenerateDatapoint(
                instruction=instruction,
                text=text,
                examples=examples,
            ),
            sampling_strategy,
        )

    def parse_force(
        self,
        instruction: str,
        typ: Union[Type[T], Dict[str, Any]],
        text: Optional[str] = None,
        examples: Optional[List[ParseForceDatapoint]] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        request_router: Optional[RequestRouter] = None,
        models: Optional[List[ParseForceModel]] = None,
    ) -> Union[T, Dict[str, Any]]:
        if models is None:
            models = self.parse_force_models
        if sampling_strategy is None:
            sampling_strategy = self.sampling_strategy
        if request_router is None:
            request_router = self.request_router
        return self._api_call(
            models,
            ParseForceDatapoint(
                instruction=instruction,
                typ=typ,
                text=text,
                examples=examples,
            ),
            sampling_strategy,
        )

    def score(
        self,
        instruction: str,
        text: str,
        min: int,
        max: int,
        examples: Optional[List[ScoreDatapoint]] = None,
        sampling_strategy: Optional[SamplingStrategy] = None,
        request_router: Optional[RequestRouter] = None,
        models: Optional[List[ScoreModel]] = None,
    ) -> int:
        if models is None:
            models = self.score_models
        if sampling_strategy is None:
            sampling_strategy = self.sampling_strategy
        if request_router is None:
            request_router = self.request_router
        return self._api_call(
            models,
            ScoreDatapoint(
                instruction=instruction,
                text=text,
                min=min,
                max=max,
                examples=examples,
            ),
            sampling_strategy,
        )


def default_api(
    log_file: Optional[str] = None,
    sampling_strategy: Optional[SamplingStrategy] = None,
    request_router: Optional[RequestRouter] = None,
) -> API:
    from benchmark.tau_bench.model_utils.model.claude import ClaudeModel
    from benchmark.tau_bench.model_utils.model.gpt import GPTModel

    return API.from_general_models(
        [ClaudeModel(), GPTModel()],
        log_file=log_file,
        sampling_strategy=sampling_strategy,
        request_router=request_router,
    )


def default_api_from_args(args: argparse.Namespace) -> API:
    return default_api(
        log_file=args.log_file,
        sampling_strategy=args.sampling_strategy,
        request_router=args.request_router,
    )


def default_quick_api(
    log_file: Optional[str] = None,
    sampling_strategy: Optional[SamplingStrategy] = None,
    request_router: Optional[RequestRouter] = None,
) -> API:
    from benchmark.tau_bench.model_utils.model.claude import ClaudeModel

    return API.from_general_model(
        ClaudeModel(),
        log_file=log_file,
        sampling_strategy=sampling_strategy,
        request_router=request_router,
    )
