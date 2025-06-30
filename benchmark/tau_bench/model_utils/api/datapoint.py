"""Data point handling for model interactions in the TAU benchmark.

This module provides classes and utilities for handling data points in model interactions,
including classification, generation, parsing, and scoring tasks. It includes functionality
for data point creation, evaluation, and comparison.

The module includes:
- Base Datapoint class and specialized datapoint types
- Evaluation result handling
- Data point comparison utilities
- Factory functions for creating datapoints
"""

from __future__ import annotations

import abc
import json
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from benchmark import tau_bench
from benchmark.tau_bench.model_utils.api._model_methods import MODEL_METHODS
from benchmark.tau_bench.model_utils.api.exception import APIError
from benchmark.tau_bench.model_utils.api.types import PartialObj
from benchmark.tau_bench.model_utils.model.exception import ModelError

T = TypeVar("T", bound=BaseModel)


def _is_trace(obj: dict[str, Any]) -> bool:
    """Check if a dictionary represents a trace.

    Args:
        obj (Dict[str, Any]): The dictionary to check.

    Returns:
        bool: True if the dictionary represents a trace, False otherwise.
    """
    return (
        "method_name" in obj
        and obj["method_name"] in MODEL_METHODS
        and "kwargs" in obj
        and "response" in obj
        and isinstance(obj["kwargs"], dict)
    )


def dict_equal(d1: dict[str, Any], d2: dict[str, Any]) -> bool:
    """Compare two dictionaries for equality.

    This function performs a deep comparison of two dictionaries, handling nested
    dictionaries, lists, sets, and strings with special comparison rules.

    Args:
        d1 (Dict[str, Any]): First dictionary to compare.
        d2 (Dict[str, Any]): Second dictionary to compare.

    Returns:
        bool: True if the dictionaries are equal, False otherwise.
    """
    d1_keys_sorted: list[str] = sorted(d1.keys())
    d2_keys_sorted: list[str] = sorted(d2.keys())
    if d1_keys_sorted != d2_keys_sorted:
        return False
    for k in d1_keys_sorted:
        if isinstance(d1[k], dict) and isinstance(d2[k], dict):
            if not dict_equal(d1[k], d2[k]):
                return False
        elif isinstance(d1[k], list) and isinstance(d2[k], list):
            if not list_equal(d1[k], d2[k]):
                return False
        elif isinstance(d1[k], set) and isinstance(d2[k], set):
            if d1[k] != d2[k]:
                return False
        elif isinstance(d1[k], str) and isinstance(d2[k], str):
            if not str_equal(d1[k], d2[k]):
                return False
        elif d1[k] != d2[k]:
            return False
    return True


def list_equal(l1: list[Any], l2: list[Any]) -> bool:
    """Compare two lists for equality.

    This function performs a deep comparison of two lists, handling nested
    dictionaries, lists, sets, and strings with special comparison rules.

    Args:
        l1 (List[Any]): First list to compare.
        l2 (List[Any]): Second list to compare.

    Returns:
        bool: True if the lists are equal, False otherwise.
    """
    if len(l1) != len(l2):
        return False
    for i1, i2 in zip(l1, l2, strict=False):
        if isinstance(i1, dict) and isinstance(i2, dict):
            if not dict_equal(i1, i2):
                return False
        elif isinstance(i1, list) and isinstance(i2, list):
            if not list_equal(i1, i2):
                return False
        elif isinstance(i1, set) and isinstance(i2, set):
            if i1 != i2:
                return False
        elif isinstance(i1, str) and isinstance(i2, str):
            if not str_equal(i1, i2):
                return False
        elif i1 != i2:
            return False
    return True


def set_equal(s1: set[Any], s2: set[Any]) -> bool:
    """Compare two sets for equality.

    This function performs a deep comparison of two sets, handling nested
    dictionaries, lists, sets, and strings with special comparison rules.

    Args:
        s1 (Set[Any]): First set to compare.
        s2 (Set[Any]): Second set to compare.

    Returns:
        bool: True if the sets are equal, False otherwise.
    """
    if len(s1) != len(s2):
        return False
    for i1, i2 in zip(s1, s2, strict=False):
        if isinstance(i1, dict) and isinstance(i2, dict):
            if not dict_equal(i1, i2):
                return False
        elif isinstance(i1, list) and isinstance(i2, list):
            if not list_equal(i1, i2):
                return False
        elif isinstance(i1, set) and isinstance(i2, set):
            if i1 != i2:
                return False
        elif isinstance(i1, str) and isinstance(i2, str):
            if not str_equal(i1, i2):
                return False
        elif i1 != i2:
            return False
    return True


def str_equal(s1: str, s2: str) -> bool:
    """Compare two strings for equality.

    This function compares two strings after removing special characters and
    normalizing whitespace and case.

    Args:
        s1 (str): First string to compare.
        s2 (str): Second string to compare.

    Returns:
        bool: True if the strings are equal after normalization, False otherwise.
    """

    def remove_special_chars(s: str) -> str:
        return "".join(filter(str.isalnum, s))

    def strip_and_lower(s: str) -> str:
        return s.lower().strip()

    return strip_and_lower(remove_special_chars(s1)) == strip_and_lower(
        remove_special_chars(s2)
    )


class EvaluationResult(BaseModel):
    """Result of evaluating a datapoint.

    This class represents the result of evaluating a datapoint against a model,
    including information about errors and correctness.

    Attributes:
        is_error (bool): Whether an error occurred during evaluation.
        is_correct (bool): Whether the evaluation was correct.
        datapoint (Optional[Dict[str, Any]]): The datapoint that was evaluated.
        response (Optional[Any]): The response from the model.
        error (Optional[str]): Any error message that occurred.
    """

    is_error: bool
    is_correct: bool
    datapoint: dict[str, Any] | None
    response: Any | None
    error: str | None


class Datapoint(BaseModel, abc.ABC):
    """Abstract base class for datapoints.

    This class defines the interface for all datapoint types, including methods
    for creation from traces and dictionaries, and evaluation against a model.

    Methods:
        from_trace: Create a datapoint from a trace dictionary.
        from_dict: Create a datapoint from a dictionary.
        evaluate: Evaluate the datapoint against a model.
    """

    @classmethod
    def from_trace(cls, d: dict[str, Any]) -> Datapoint:
        """Create a datapoint from a trace dictionary.

        Args:
            d (Dict[str, Any]): The trace dictionary.

        Returns:
            Datapoint: A new datapoint instance.

        Raises:
            ValueError: If the dictionary is not a valid trace.
        """
        if not _is_trace(d):
            raise ValueError(f"This is not a trace: {d}")
        response: Any = d["response"]
        kwargs: dict[str, Any] = d["kwargs"]
        return cls(response=response, **kwargs)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Datapoint:
        """Create a datapoint from a dictionary.

        Args:
            d (Dict[str, Any]): The dictionary to create from.

        Returns:
            Datapoint: A new datapoint instance.
        """
        if _is_trace(d):
            return cls.from_trace(d)
        return cls(**d)

    @abc.abstractmethod
    def evaluate(self, api: tau_bench.model_utils.API) -> EvaluationResult:
        """Evaluate the datapoint against a model.

        Args:
            api (tau_bench.model_utils.API): The API to use for evaluation.

        Returns:
            EvaluationResult: The result of the evaluation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class ClassifyDatapoint(Datapoint):
    """Datapoint for classification tasks.

    This class represents a datapoint for classification tasks, including
    instruction, text, options, and optional examples.

    Attributes:
        instruction (str): The instruction for classification.
        text (str): The text to classify.
        options (List[str]): The possible classification options.
        response (Optional[int]): The expected response.
        examples (Optional[List[ClassifyDatapoint]]): Example datapoints.
    """

    instruction: str
    text: str
    options: list[str]
    response: int | None = None
    examples: list[ClassifyDatapoint] | None = None

    def evaluate(self, api: tau_bench.model_utils.API) -> EvaluationResult:
        """Evaluate the classification datapoint.

        Args:
            api (tau_bench.model_utils.API): The API to use for evaluation.

        Returns:
            EvaluationResult: The result of the evaluation.
        """
        return run_and_catch_api_error(
            lambda: api.classify(
                instruction=self.instruction,
                text=self.text,
                options=self.options,
                examples=self.examples,
            ),
            self.response,
            self.model_dump(),
        )


class BinaryClassifyDatapoint(Datapoint):
    """Datapoint for binary classification tasks.

    This class represents a datapoint for binary classification tasks, including
    instruction, text, and optional examples.

    Attributes:
        instruction (str): The instruction for classification.
        text (str): The text to classify.
        response (Optional[bool]): The expected response.
        examples (Optional[List[BinaryClassifyDatapoint]]): Example datapoints.
    """

    instruction: str
    text: str
    response: bool | None = None
    examples: list[BinaryClassifyDatapoint] | None = None

    def evaluate(self, api: tau_bench.model_utils.API) -> EvaluationResult:
        """Evaluate the binary classification datapoint.

        Args:
            api (tau_bench.model_utils.API): The API to use for evaluation.

        Returns:
            EvaluationResult: The result of the evaluation.
        """
        return run_and_catch_api_error(
            lambda: api.binary_classify(
                instruction=self.instruction, text=self.text, examples=self.examples
            ),
            self.response,
            self.model_dump(),
        )


class ScoreDatapoint(Datapoint):
    """Datapoint for scoring tasks.

    This class represents a datapoint for scoring tasks, including instruction,
    text, score range, and optional examples.

    Attributes:
        instruction (str): The instruction for scoring.
        text (str): The text to score.
        min (int): The minimum possible score.
        max (int): The maximum possible score.
        response (Optional[int]): The expected response.
        examples (Optional[List[ScoreDatapoint]]): Example datapoints.
    """

    instruction: str
    text: str
    min: int
    max: int
    response: int | None = None
    examples: list[ScoreDatapoint] | None = None

    def evaluate(self, api: tau_bench.model_utils.API) -> EvaluationResult:
        """Evaluate the scoring datapoint.

        Args:
            api (tau_bench.model_utils.API): The API to use for evaluation.

        Returns:
            EvaluationResult: The result of the evaluation.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError


class ParseDatapoint(Datapoint):
    """Datapoint for parsing tasks.

    This class represents a datapoint for parsing tasks, including text,
    type information, and optional examples.

    Attributes:
        text (str): The text to parse.
        typ (Union[type[T], Dict[str, Any]]): The type to parse into.
        response (Optional[Union[Dict[str, Any], T, PartialObj]]): The expected response.
        examples (Optional[List[ParseDatapoint]]): Example datapoints.
    """

    text: str
    typ: type[T] | dict[str, Any]
    response: dict[str, Any] | T | PartialObj | None = None
    examples: list[ParseDatapoint] | None = None

    def evaluate(self, api: tau_bench.model_utils.API) -> EvaluationResult:
        """Evaluate the parsing datapoint.

        Args:
            api (tau_bench.model_utils.API): The API to use for evaluation.

        Returns:
            EvaluationResult: The result of the evaluation.
        """
        return run_and_catch_api_error(
            lambda: api.parse(text=self.text, typ=self.typ),
            self.response,
            self.model_dump(),
        )


class GenerateDatapoint(Datapoint):
    """Datapoint for generation tasks.

    This class represents a datapoint for generation tasks, including instruction,
    text, and optional examples.

    Attributes:
        instruction (str): The instruction for generation.
        text (str): The text to generate from.
        response (Optional[str]): The expected response.
        examples (Optional[List[GenerateDatapoint]]): Example datapoints.
    """

    instruction: str
    text: str
    response: str | None = None
    examples: list[GenerateDatapoint] | None = None

    def evaluate(
        self, api: tau_bench.model_utils.API
    ) -> tau_bench.model_utils.EvaluationResult:
        """Evaluate the generation datapoint.

        Args:
            api (tau_bench.model_utils.API): The API to use for evaluation.

        Returns:
            EvaluationResult: The result of the evaluation.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError


class ParseForceDatapoint(Datapoint):
    """Datapoint for forced parsing tasks.

    This class represents a datapoint for forced parsing tasks, including
    instruction, type information, optional text, and optional examples.

    Attributes:
        instruction (str): The instruction for parsing.
        typ (Union[type[T], Dict[str, Any]]): The type to parse into.
        text (Optional[str]): The text to parse.
        response (Optional[Union[Dict[str, Any], T]]): The expected response.
        examples (Optional[List[ParseForceDatapoint]]): Example datapoints.
    """

    instruction: str
    typ: type[T] | dict[str, Any]
    text: str | None = None
    response: dict[str, Any] | T | None = None
    examples: list[ParseForceDatapoint] | None = None

    def evaluate(self, api: tau_bench.model_utils.API) -> EvaluationResult:
        """Evaluate the forced parsing datapoint.

        Args:
            api (tau_bench.model_utils.API): The API to use for evaluation.

        Returns:
            EvaluationResult: The result of the evaluation.
        """
        return run_and_catch_api_error(
            lambda: api.parse_force(
                instruction=self.instruction,
                text=self.text,
                typ=self.typ,
                examples=self.examples,
            ),
            self.response,
            self.model_dump(),
        )


def datapoint_factory(d: dict[str, Any]) -> Datapoint:
    """Create a datapoint from a dictionary.

    This function creates an appropriate datapoint instance based on the
    contents of the input dictionary.

    Args:
        d (Dict[str, Any]): The dictionary to create from.

    Returns:
        Datapoint: A new datapoint instance.

    Raises:
        ValueError: If the dictionary does not match any known datapoint type.
    """
    if _is_trace(d):
        method_name: str = d["method_name"]
        kwargs: dict[str, Any] = d["kwargs"]
        data: dict[str, Any] = {"response": d["response"], **kwargs}
        if method_name == "classify":
            return ClassifyDatapoint(**data)
        elif method_name == "binary_classify":
            return BinaryClassifyDatapoint(**data)
        elif method_name == "parse":
            return ParseDatapoint(**data)
        elif method_name == "parse_force":
            return ParseForceDatapoint(**data)
        elif method_name == "generate":
            return GenerateDatapoint(**data)
        elif method_name == "score":
            return ScoreDatapoint(**data)
        else:
            raise ValueError(f"Unknown method name: {method_name}")
    else:
        if all(k in d for k in ["instruction", "text", "options"]) and isinstance(
            d["response"], int
        ):
            return ClassifyDatapoint(**d)
        elif all(k in d for k in ["instruction", "text"]) and isinstance(
            d["response"], bool
        ):
            return BinaryClassifyDatapoint(**d)
        elif all(k in d for k in ["text", "typ"]):
            return ParseDatapoint(**d)
        elif all(k in d for k in ["instruction", "typ"]):
            return ParseForceDatapoint(**d)
        elif all(k in d for k in ["instruction", "text"]) and isinstance(
            d["response"], str
        ):
            return GenerateDatapoint(**d)
        elif all(k in d for k in ["instruction", "text", "min", "max"]):
            return ScoreDatapoint(**d)
        else:
            raise ValueError(f"Unknown datapoint type: {d}")


def run_and_catch_api_error(
    callable: Callable[..., object], response: object, datapoint: dict[str, object]
) -> EvaluationResult:
    """Run a callable and catch any API errors.

    This function runs a callable and catches any API or model errors,
    returning an appropriate evaluation result.

    Args:
        callable (Callable[..., object]): The callable to run.
        response (object): The expected response.
        datapoint (Dict[str, object]): The datapoint being evaluated.

    Returns:
        EvaluationResult: The result of the evaluation.
    """
    try:
        result: Any = callable()
        return EvaluationResult(
            is_error=False,
            is_correct=result == response,
            datapoint=datapoint,
            response=result,
            error=None,
        )
    except (APIError, ModelError) as e:
        return EvaluationResult(
            is_error=True,
            is_correct=False,
            datapoint=datapoint,
            response=None,
            error=str(e),
        )


def load_from_disk(path: str) -> list[Datapoint]:
    """Load datapoints from a file.

    This function loads datapoints from a JSON file on disk.

    Args:
        path (str): The path to the file.

    Returns:
        List[Datapoint]: A list of datapoints loaded from the file.
    """
    with open(path) as f:
        data: list[dict[str, Any]] = json.load(f)
    return [datapoint_factory(d) for d in data]
