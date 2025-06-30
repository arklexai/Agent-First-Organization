"""API interface for model interactions in the TAU benchmark.

This module provides a unified API interface for interacting with various language models
in the TAU benchmark. It supports different types of model operations (classification,
generation, parsing, scoring) and includes features for model routing, sampling strategies,
and caching.

The module includes:
- API class for model interaction
- Model routing and sampling strategies
- Caching and logging utilities
- Factory functions for creating API instances
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, TypeVar

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


class API:
    """API interface for model interactions.

    This class provides a unified interface for interacting with various language models,
    supporting different types of operations (classification, generation, parsing, scoring)
    and including features for model routing, sampling strategies, and caching.

    Attributes:
        wrappers_for_main_methods (List[Callable]): List of wrapper functions to apply
            to main API methods.
        sampling_strategy (SamplingStrategy): Strategy for sampling model outputs.
        request_router (RequestRouter): Router for directing requests to appropriate models.
        _log_file (Optional[str]): Path to the log file.
        binary_classify_models (Optional[List[BinaryClassifyModel]]): Models for binary classification.
        classify_models (List[ClassifyModel]): Models for classification.
        parse_models (List[ParseModel]): Models for parsing.
        generate_models (List[GenerateModel]): Models for generation.
        parse_force_models (List[ParseForceModel]): Models for forced parsing.
        score_models (List[ScoreModel]): Models for scoring.
    """

    wrappers_for_main_methods: list[Callable] = [log_call, cache_call_w_dedup]

    def __init__(
        self,
        parse_models: list[ParseModel],
        generate_models: list[GenerateModel],
        parse_force_models: list[ParseForceModel],
        score_models: list[ScoreModel],
        classify_models: list[ClassifyModel],
        binary_classify_models: list[BinaryClassifyModel] | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        request_router: RequestRouter | None = None,
        log_file: str | None = None,
    ) -> None:
        """Initialize the API interface.

        Args:
            parse_models (List[ParseModel]): Models for parsing.
            generate_models (List[GenerateModel]): Models for generation.
            parse_force_models (List[ParseForceModel]): Models for forced parsing.
            score_models (List[ScoreModel]): Models for scoring.
            classify_models (List[ClassifyModel]): Models for classification.
            binary_classify_models (Optional[List[BinaryClassifyModel]]): Models for binary classification.
            sampling_strategy (Optional[SamplingStrategy]): Strategy for sampling model outputs.
            request_router (Optional[RequestRouter]): Router for directing requests.
            log_file (Optional[str]): Path to the log file.
        """
        if sampling_strategy is None:
            sampling_strategy = get_default_sampling_strategy()
        if request_router is None:
            request_router = default_request_router()
        self.sampling_strategy: SamplingStrategy = sampling_strategy
        self.request_router: RequestRouter = request_router
        self._log_file: str | None = log_file
        self.binary_classify_models: list[BinaryClassifyModel] | None = (
            binary_classify_models
        )
        self.classify_models: list[ClassifyModel] = classify_models
        self.parse_models: list[ParseModel] = parse_models
        self.generate_models: list[GenerateModel] = generate_models
        self.parse_force_models: list[ParseForceModel] = parse_force_models
        self.score_models: list[ScoreModel] = score_models

        self.__init_subclass__()

    def __init_subclass__(cls) -> None:
        """Initialize subclasses by applying wrappers to main methods."""
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
        sampling_strategy: SamplingStrategy | None = None,
        request_router: RequestRouter | None = None,
        log_file: str | None = None,
    ) -> API:
        """Create an API instance from a general model.

        Args:
            model (GeneralModel): The general model to use.
            sampling_strategy (Optional[SamplingStrategy]): Strategy for sampling model outputs.
            request_router (Optional[RequestRouter]): Router for directing requests.
            log_file (Optional[str]): Path to the log file.

        Returns:
            API: A new API instance configured with the general model.
        """
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
        models: list[GeneralModel],
        sampling_strategy: SamplingStrategy | None = None,
        request_router: RequestRouter | None = None,
        log_file: str | None = None,
    ) -> API:
        """Create an API instance from multiple general models.

        Args:
            models (List[GeneralModel]): The general models to use.
            sampling_strategy (Optional[SamplingStrategy]): Strategy for sampling model outputs.
            request_router (Optional[RequestRouter]): Router for directing requests.
            log_file (Optional[str]): Path to the log file.

        Returns:
            API: A new API instance configured with the general models.

        Raises:
            ValueError: If no models are provided.
        """
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
        self, models: list[BinaryClassifyModel]
    ) -> None:
        """Set the default binary classification models.

        Args:
            models (List[BinaryClassifyModel]): The models to use.

        Raises:
            ValueError: If no models are provided.
        """
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        self.binary_classify_models = models

    def set_default_classify_models(self, models: list[ClassifyModel]) -> None:
        """Set the default classification models.

        Args:
            models (List[ClassifyModel]): The models to use.

        Raises:
            ValueError: If no models are provided.
        """
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        self.classify_models = models

    def set_default_parse_models(self, models: list[ParseModel]) -> None:
        """Set the default parsing models.

        Args:
            models (List[ParseModel]): The models to use.

        Raises:
            ValueError: If no models are provided.
        """
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        self.parse_models = models

    def set_default_generate_models(self, models: list[GenerateModel]) -> None:
        """Set the default generation models.

        Args:
            models (List[GenerateModel]): The models to use.

        Raises:
            ValueError: If no models are provided.
        """
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        self.generate_models = models

    def set_default_parse_force_models(self, models: list[ParseForceModel]) -> None:
        """Set the default forced parsing models.

        Args:
            models (List[ParseForceModel]): The models to use.

        Raises:
            ValueError: If no models are provided.
        """
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        self.parse_force_models = models

    def set_default_score_models(self, models: list[ScoreModel]) -> None:
        """Set the default scoring models.

        Args:
            models (List[ScoreModel]): The models to use.

        Raises:
            ValueError: If no models are provided.
        """
        if len(models) == 0:
            raise ValueError("Must provide at least one model")
        self.score_models = models

    def set_default_sampling_strategy(
        self, sampling_strategy: SamplingStrategy
    ) -> None:
        """Set the default sampling strategy.

        Args:
            sampling_strategy (SamplingStrategy): The strategy to use.
        """
        self.sampling_strategy = sampling_strategy

    def set_default_request_router(self, request_router: RequestRouter) -> None:
        """Set the default request router.

        Args:
            request_router (RequestRouter): The router to use.
        """
        self.request_router = request_router

    def _run_with_sampling_strategy(
        self,
        models: list[AnyModel],
        datapoint: Datapoint,
        sampling_strategy: SamplingStrategy,
    ) -> T:
        """Run a datapoint through models using a sampling strategy.

        Args:
            models (List[AnyModel]): The models to use.
            datapoint (Datapoint): The datapoint to process.
            sampling_strategy (SamplingStrategy): The strategy to use.

        Returns:
            T: The result of processing the datapoint.

        Raises:
            ValueError: If no models are provided or if the datapoint type is unknown.
        """
        assert len(models) > 0

        def _run_datapoint(model: AnyModel, temp: float | None = None) -> T:
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
        models: list[AnyModel],
        datapoint: Datapoint,
        sampling_strategy: SamplingStrategy,
    ) -> T:
        """Make an API call using the specified models and sampling strategy.

        Args:
            models (List[AnyModel]): The models to use.
            datapoint (Datapoint): The datapoint to process.
            sampling_strategy (SamplingStrategy): The strategy to use.

        Returns:
            T: The result of processing the datapoint.
        """
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
        options: list[str],
        examples: list[ClassifyDatapoint] | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        request_router: RequestRouter | None = None,
        models: list[ClassifyModel] | None = None,
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
        examples: list[BinaryClassifyDatapoint] | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        request_router: RequestRouter | None = None,
        models: list[BinaryClassifyModel] | None = None,
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
        typ: type[T] | dict[str, Any],
        examples: list[ParseDatapoint] | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        request_router: RequestRouter | None = None,
        models: list[ParseModel] | None = None,
    ) -> T | PartialObj | dict[str, Any]:
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
        examples: list[GenerateDatapoint] | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        request_router: RequestRouter | None = None,
        models: list[GenerateModel] | None = None,
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
        typ: type[T] | dict[str, Any],
        text: str | None = None,
        examples: list[ParseForceDatapoint] | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        request_router: RequestRouter | None = None,
        models: list[ParseForceModel] | None = None,
    ) -> T | dict[str, Any]:
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
        examples: list[ScoreDatapoint] | None = None,
        sampling_strategy: SamplingStrategy | None = None,
        request_router: RequestRouter | None = None,
        models: list[ScoreModel] | None = None,
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
    log_file: str | None = None,
    sampling_strategy: SamplingStrategy | None = None,
    request_router: RequestRouter | None = None,
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
    log_file: str | None = None,
    sampling_strategy: SamplingStrategy | None = None,
    request_router: RequestRouter | None = None,
) -> API:
    from benchmark.tau_bench.model_utils.model.claude import ClaudeModel

    return API.from_general_model(
        ClaudeModel(),
        log_file=log_file,
        sampling_strategy=sampling_strategy,
        request_router=request_router,
    )
