import abc
import json
from typing import Any, Dict, List, Optional, TypeVar, Union

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
from benchmark.tau_bench.model_utils.model.exception import ModelError
from benchmark.tau_bench.model_utils.model.general_model import GeneralModel
from benchmark.tau_bench.model_utils.model.utils import (
    approx_num_tokens,
    display_choices,
    json_response_to_obj_or_partial_obj,
    optionalize_type,
    parse_json_or_json_markdown,
    type_to_json_schema_string,
)

T = TypeVar("T", bound=BaseModel)


class Score(BaseModel):
    score: int


class Classification(BaseModel):
    classification: str


def task_prompt(task: str, text: str) -> str:
    return f"# Task\n{task}\n\n{text}"


def force_json_prompt(text: str, with_prefix: bool = False) -> str:
    suffix = (
        'For example:\nassistant:```json\n{"key": "value"}\n```'
        if not with_prefix
        else "\n\n```json\n"
    )
    return f"{text}\n\nThe result should be a valid JSON object in a markdown block only. {suffix}"


def build_score_state(
    instruction: str,
    text: str,
    min: int,
    max: int,
    examples: Optional[List[ScoreDatapoint]] = None,
) -> str:
    def display_sample(
        instr: str, t: str, min: int, max: int, response: Optional[int] = None
    ) -> str:
        p = task_prompt(
            task='Score the following text with the provided instruction and range as an integer value in valid JSON:\n{"score": number}',
            text=force_json_prompt(
                f"Instruction:\n{instr}\n\nText:\n{t}\n\nRange:\n[{min}, {max}]",
                with_prefix=True,
            ),
        )
        if response is not None:
            # the json markdown block is opened in the prompt
            return f'{p}\n{{"score": {response}}}\n```'
        return p

    p = (
        "\n\n".join(
            [
                display_sample(ex.instruction, ex.text, min, max, ex.response)
                for ex in examples
            ]
        )
        if examples is not None
        else ""
    )
    return f"{p}\n\n{display_sample(instr=instruction, t=text, min=min, max=max)}"


def build_parse_force_state(
    instruction: str,
    typ: Union[type[T], Dict[str, Any]],
    text: Optional[str] = None,
    examples: Optional[List[ParseForceDatapoint]] = None,
) -> str:
    def display_sample(
        instr: str,
        t: str,
        ty: Union[type[T], Dict[str, Any]],
        response: Optional[Union[T, Dict[str, Any]]] = None,
    ) -> str:
        if isinstance(ty, dict):
            json_schema_string = json.dumps(ty)
        else:
            json_schema_string = type_to_json_schema_string(ty)
        text_insert = "" if t is None else f"\n\nText:\n{t}"
        input_text = force_json_prompt(
            text=f"Instruction:\n{instr}{text_insert}\n\nSchema:\n{json_schema_string}",
            with_prefix=True,
        )
        if response is not None:
            if isinstance(response, dict):
                response_display = json.dumps(response)
            else:
                response_display = response.model_dump_json()
            # the json markdown block is opened in the prompt
            return f"{input_text}\n{response_display}\n```"
        return input_text

    p = (
        "".join(
            [
                display_sample(
                    instr=ex.instruction,
                    t=ex.text,
                    ty=ex.typ,
                    response=ex.response,
                )
                for ex in examples
            ]
        )
        + "\n\n"
        if examples is not None and len(examples) > 0
        else ""
    )
    p += display_sample(instr=instruction, t=text, ty=typ)
    return task_prompt(
        task="Generate an object with the provided instruction, text, and schema.",
        text=p,
    )


def build_parse_state(
    text: str,
    typ: Union[type[T], Dict[str, Any]],
    examples: Optional[List[ParseDatapoint]] = None,
) -> str:
    instruction = "Parse the following text with the provided JSON schema."

    def display_sample(
        t: str,
        ty: Union[type[T], Dict[str, Any]],
        response: Optional[Union[T, PartialObj, Dict[str, Any]]] = None,
    ) -> str:
        if isinstance(ty, dict):
            json_schema_string = json.dumps(ty)
        else:
            optionalized_typ = optionalize_type(ty)
            json_schema_string = type_to_json_schema_string(optionalized_typ)
        # instruction is repeated to emphasize the task
        prompt = task_prompt(
            task=instruction,
            text=force_json_prompt(
                f"Text:\n{t}\n\nSchema:\n{json_schema_string}", with_prefix=True
            ),
        )
        if response is None:
            return prompt
        if isinstance(response, dict):
            response_display = json.dumps(response)
        else:
            response_display = response.model_dump_json()
        # the json markdown block is opened in the prompt
        json_response = f"{response_display}\n```"
        return f"{prompt}\n{json_response}"

    p = ""
    if examples is not None and len(examples) > 0:
        p = "\n\n".join(
            [
                display_sample(t=ex.text, ty=ex.typ, response=ex.response)
                for ex in examples
            ]
        )
    return f"{p}\n\n{display_sample(t=text, ty=typ)}"


def build_classify_state(
    instruction: str,
    text: str,
    options: List[str],
    examples: Optional[List[ClassifyDatapoint]] = None,
) -> tuple[str, Dict[str, int]]:
    def display_sample(
        instr: str, t: str, opts: List[str], response: Optional[int] = None
    ) -> Union[str, tuple[str, Dict[str, int]]]:
        choices_display, decode_map = display_choices(opts)
        input_text = force_json_prompt(
            f"Instruction:\n{instr}\n\nText:\n{t}\n\nChoices:\n{choices_display}",
            with_prefix=True,
        )
        prompt = task_prompt(task=instr, text=input_text)
        if response is not None:
            label = None
            for k, v in decode_map.items():
                if v == response:
                    label = k
                    break
            assert label is not None
            # the json markdown block is opened in the prompt
            json_display = f'{{"classification": "{label}"}}\n```'
            return f"{prompt}\n{json_display}"
        return prompt, decode_map

    p = 'Classify the following text with the provided instruction and choices. To classify, provide the key of the choice:\n{"classification": string}\n\nFor example, if the correct choice is \'Z. description of choice Z\', then provide \'Z\' as the classification as valid JSON:\n```json\n{"classification": "Z"}\n```'
    if examples is not None and len(examples) > 0:
        example_displays = "\n\n".join(
            [
                display_sample(
                    instr=ex.instruction,
                    t=ex.text,
                    opts=ex.options,
                    response=ex.response,
                )
                for ex in examples
            ]
        )
        p += f"\n\n{example_displays}"
    prompt, decode_map = display_sample(instr=instruction, t=text, opts=options)
    return f"{p}\n\n{prompt}", decode_map


def build_generate_state(
    instruction: str,
    text: str,
    examples: Optional[List[GenerateDatapoint]] = None,
) -> str:
    def display_sample(instr: str, t: str, response: Optional[str] = None) -> str:
        prompt = task_prompt(task=instr, text=t)
        if response is not None:
            return f"{prompt}\n\nText: {response}"
        return prompt

    prompt = (
        "\n\n".join([display_sample(ex.instruction, ex.text) for ex in examples])
        + "\n\n"
        if examples is not None and len(examples) > 0
        else ""
    )
    return f"{prompt}\n\n{display_sample(instruction, text)}\n\nText:"


class CompletionModel(GeneralModel):
    @abc.abstractmethod
    def generate_from_prompt(
        self, prompt: str, temperature: Optional[float] = None
    ) -> str:
        pass

    @abc.abstractmethod
    def parse_force_from_prompt(
        self,
        prompt: str,
        typ: Union[BaseModel, Dict[str, Any]],
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        pass

    def handle_parse_force_response(self, prompt: str, content: str) -> Dict[str, Any]:
        try:
            return parse_json_or_json_markdown(content)
        except Exception as e:
            raise ModelError(
                f"Failed to parse response as JSON: {e}\n\nPrompt:\n{prompt}\n\nResponse:\n{content}"
            )

    def _handle_classify_response(
        self, res: Dict[str, int], decode_map: Dict[str, int]
    ) -> int:
        try:
            return res["classification"]
        except KeyError:
            raise ModelError(
                f"Response missing 'classification' key: {res}\n\nDecode map: {decode_map}"
            )

    def classify(
        self,
        instruction: str,
        text: str,
        options: List[str],
        examples: Optional[List[ClassifyDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> int:
        prompt, decode_map = build_classify_state(
            instruction=instruction,
            text=text,
            options=options,
            examples=examples,
        )
        res = self.parse_force_from_prompt(
            prompt=prompt,
            typ=Classification,
            temperature=temperature,
        )
        return self._handle_classify_response(res=res, decode_map=decode_map)

    def parse(
        self,
        text: str,
        typ: Union[type[T], Dict[str, Any]],
        examples: Optional[List[ParseDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> Union[T, PartialObj, Dict[str, Any]]:
        prompt = build_parse_state(text=text, typ=typ, examples=examples)
        res = self.parse_force_from_prompt(
            prompt=prompt,
            typ=typ,
            temperature=temperature,
        )
        return json_response_to_obj_or_partial_obj(res=res, typ=typ)

    def generate(
        self,
        instruction: str,
        text: str,
        examples: Optional[List[GenerateDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        prompt = build_generate_state(
            instruction=instruction,
            text=text,
            examples=examples,
        )
        return self.generate_from_prompt(prompt=prompt, temperature=temperature)

    def _handle_parse_force_response(self, res: Dict[str, Any], typ: type[T]) -> T:
        try:
            return typ.model_validate(res)
        except Exception as e:
            raise ModelError(f"Failed to validate response as {typ.__name__}: {e}")

    def parse_force(
        self,
        instruction: str,
        typ: Union[type[T], Dict[str, Any]],
        text: Optional[str] = None,
        examples: Optional[List[ParseForceDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> Union[T, Dict[str, Any]]:
        prompt = build_parse_force_state(
            instruction=instruction,
            typ=typ,
            text=text,
            examples=examples,
        )
        res = self.parse_force_from_prompt(
            prompt=prompt,
            typ=typ,
            temperature=temperature,
        )
        if isinstance(typ, dict):
            return res
        return self._handle_parse_force_response(res=res, typ=typ)

    def _handle_score_response(
        self,
        res: Dict[str, Any],
        min: int,
        max: int,
    ) -> int:
        try:
            score = res["score"]
            if not isinstance(score, int):
                raise ModelError(f"Score must be an integer, got {type(score)}")
            if score < min or score > max:
                raise ModelError(f"Score {score} outside range [{min}, {max}]")
            return score
        except KeyError:
            raise ModelError(f"Response missing 'score' key: {res}")

    def score(
        self,
        instruction: str,
        text: str,
        min: int,
        max: int,
        examples: Optional[List[ScoreDatapoint]] = None,
        temperature: Optional[float] = None,
    ) -> int:
        prompt = build_score_state(
            instruction=instruction,
            text=text,
            min=min,
            max=max,
            examples=examples,
        )
        res = self.parse_force_from_prompt(
            prompt=prompt,
            typ=Score,
            temperature=temperature,
        )
        return self._handle_score_response(res=res, min=min, max=max)


def build_prompts(dps: List[Datapoint], include_response: bool = True) -> List[str]:
    prompts: List[str] = []
    for dp in dps:
        if isinstance(dp, ParseDatapoint):
            prompts.extend(build_parse_prompts([dp], include_response))
        elif isinstance(dp, BinaryClassifyDatapoint):
            prompts.extend(build_binary_classify_prompts([dp], include_response))
        elif isinstance(dp, ClassifyDatapoint):
            prompts.extend(build_classify_prompts([dp], include_response))
        elif isinstance(dp, ParseForceDatapoint):
            prompts.extend(build_parse_force_prompts([dp], include_response))
        elif isinstance(dp, GenerateDatapoint):
            prompts.extend(build_generate_prompts([dp], include_response))
        elif isinstance(dp, ScoreDatapoint):
            prompts.extend(build_score_prompts([dp], include_response))
        else:
            raise ValueError(f"Unknown datapoint type: {type(dp)}")
    return prompts


def build_parse_prompts(
    dps: List[ParseDatapoint],
    include_response: bool = True,
) -> List[str]:
    prompts: List[str] = []
    for dp in dps:
        prompt = build_parse_state(
            text=dp.text,
            typ=dp.typ,
            examples=None,
        )
        if include_response and dp.response is not None:
            if isinstance(dp.response, dict):
                response_display = json.dumps(dp.response)
            else:
                response_display = dp.response.model_dump_json()
            prompt += f"\n\n{response_display}"
        prompts.append(prompt)
    return prompts


def build_binary_classify_prompts(
    dps: List[BinaryClassifyDatapoint],
    include_response: bool = True,
) -> List[str]:
    prompts: List[str] = []
    for dp in dps:
        prompt, decode_map = build_classify_state(
            instruction=dp.instruction,
            text=dp.text,
            options=["Yes", "No"],
            examples=None,
        )
        if include_response and dp.response is not None:
            label = "Yes" if dp.response else "No"
            prompt += f'\n\n{{"classification": "{label}"}}'
        prompts.append(prompt)
    return prompts


def build_classify_prompts(
    dps: List[ClassifyDatapoint],
    include_response: bool = True,
) -> List[str]:
    prompts: List[str] = []
    for dp in dps:
        prompt, decode_map = build_classify_state(
            instruction=dp.instruction,
            text=dp.text,
            options=dp.options,
            examples=None,
        )
        if include_response and dp.response is not None:
            prompt += label_idx_to_label_json(dp.response, decode_map)
        prompts.append(prompt)
    return prompts


def label_idx_to_label_json(idx: int, decode_map: Dict[str, int]) -> str:
    label = None
    for k, v in decode_map.items():
        if v == idx:
            label = k
            break
    assert label is not None
    return f'\n\n{{"classification": "{label}"}}'


def build_parse_force_prompts(
    dps: List[ParseForceDatapoint],
    include_response: bool = True,
) -> List[str]:
    prompts: List[str] = []
    for dp in dps:
        prompt = build_parse_force_state(
            instruction=dp.instruction,
            typ=dp.typ,
            text=dp.text,
            examples=None,
        )
        if include_response and dp.response is not None:
            if isinstance(dp.response, dict):
                response_display = json.dumps(dp.response)
            else:
                response_display = dp.response.model_dump_json()
            prompt += f"\n\n{response_display}"
        prompts.append(prompt)
    return prompts


def build_generate_prompts(
    dps: List[GenerateDatapoint], include_response: bool = True
) -> List[str]:
    prompts: List[str] = []
    for dp in dps:
        prompt = build_generate_state(
            instruction=dp.instruction,
            text=dp.text,
            examples=None,
        )
        if include_response and dp.response is not None:
            prompt += f"\n\n{dp.response}"
        prompts.append(prompt)
    return prompts


def build_score_prompts(
    dps: List[ScoreDatapoint],
    include_response: bool = True,
) -> List[str]:
    prompts: List[str] = []
    for dp in dps:
        prompt = build_score_state(
            instruction=dp.instruction,
            text=dp.text,
            min=dp.min,
            max=dp.max,
            examples=None,
        )
        if include_response and dp.response is not None:
            prompt += f'\n\n{{"score": {dp.response}}}'
        prompts.append(prompt)
    return prompts


def approx_prompt_str(dp: Datapoint, include_response: bool = False) -> str:
    return build_prompts([dp], include_response)[0]


def approx_cost_for_datapoint(
    dp: Datapoint,
    price_per_input_token: float,
) -> float:
    prompt = approx_prompt_str(dp)
    return approx_num_tokens(prompt) * price_per_input_token


def approx_latency_for_datapoint(
    dp: Datapoint, latency_ms_per_output_token: float
) -> float:
    prompt = approx_prompt_str(dp)
    return approx_num_tokens(prompt) * latency_ms_per_output_token
