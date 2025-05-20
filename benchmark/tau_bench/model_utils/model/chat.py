import abc
import enum
import json
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

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
    add_md_tag,
    display_choices,
    json_response_to_obj_or_partial_obj,
    optionalize_type,
    parse_json_or_json_markdown,
    type_to_json_schema_string,
)

T = TypeVar("T", bound=BaseModel)


class Role(str, enum.Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class Message(BaseModel):
    role: Role
    content: str
    obj: Optional[Dict[str, Any]] = None

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        if self.obj is not None:
            return super().model_dump(**kwargs)
        return {"role": self.role, "content": self.content}


class PromptSuffixStrategy(str, enum.Enum):
    JSON = "json"
    JSON_MD_BLOCK = "json_md_block"


def force_json_prompt(
    text: str,
    suffix_strategy: PromptSuffixStrategy = PromptSuffixStrategy.JSON,
) -> str:
    if suffix_strategy == PromptSuffixStrategy.JSON:
        return f"{text}\n\nValid JSON:"
    elif suffix_strategy == PromptSuffixStrategy.JSON_MD_BLOCK:
        return f'{text}\n\nThe result should be a valid JSON object (according to the definition in the provided schema) in a markdown block only. For example:\nassistant:```json\n{{"items": ["value"]}}\n```'
    else:
        raise ValueError(f"Invalid suffix strategy: {suffix_strategy}")


def build_generate_state(
    instruction: str,
    text: str,
    examples: Optional[List[GenerateDatapoint]] = None,
) -> List[Message]:
    messages = []
    if examples is not None:
        for example in examples:
            example_msgs = [
                Message(role=Role.SYSTEM, content=example.instruction),
                Message(role=Role.USER, content=example.text),
                Message(role=Role.ASSISTANT, content=example.response),
            ]
            messages.extend(example_msgs)
    messages.append(Message(role=Role.SYSTEM, content=instruction))
    messages.append(Message(role=Role.USER, content=text))
    return messages


def build_parse_force_state(
    instruction: str,
    typ: Union[type[T], Dict[str, Any]],
    text: Optional[str] = None,
    examples: Optional[List[ParseForceDatapoint]] = None,
    suffix_strategy: PromptSuffixStrategy = PromptSuffixStrategy.JSON,
) -> List[Message]:
    def display_sample(
        instr: str,
        ty: Union[type[T], Dict[str, Any]],
        t: Optional[str] = None,
        response: Optional[Union[T, Dict[str, Any]]] = None,
    ) -> Union[Message, List[Message]]:
        if isinstance(ty, dict):
            json_schema_string = json.dumps(ty)
        else:
            json_schema_string = type_to_json_schema_string(ty)
        text_insert = "" if t is None else f"\n\nText:\n{t}"
        input_text = force_json_prompt(
            text=f"Instruction:\n{instr}{text_insert}\n\nSchema:\n{json_schema_string}",
            suffix_strategy=suffix_strategy,
        )
        if response is not None:
            if isinstance(response, dict):
                response_display = json.dumps(response)
            else:
                response_display = json.dumps(response.model_dump())
            return [
                Message(role=Role.USER, content=input_text),
                Message(role=Role.ASSISTANT, content=response_display),
            ]
        else:
            return Message(role=Role.USER, content=input_text)

    messages = [
        Message(
            role=Role.SYSTEM,
            content="Generate an object with the provided instruction, text, and schema.",
        ),
    ]
    if examples is not None:
        for example in examples:
            example_msgs = display_sample(
                instr=example.instruction,
                ty=example.typ,
                t=example.text,
                response=example.response,
            )
            assert isinstance(example_msgs, list) and all(
                isinstance(msg, Message) for msg in example_msgs
            )
            messages.extend(example_msgs)
    messages.append(display_sample(instr=instruction, ty=typ, t=text))
    return messages


def build_score_state(
    instruction: str,
    text: str,
    min: int,
    max: int,
    examples: Optional[List[ScoreDatapoint]] = None,
    suffix_strategy: PromptSuffixStrategy = PromptSuffixStrategy.JSON,
) -> List[Message]:
    def display_sample(
        instr: str, t: str, mn: int, mx: int, response: Optional[int] = None
    ) -> Union[List[Message], Message]:
        if mn > mx:
            raise ValueError(f"Invalid range: [{mn}, {mx}]")
        input_text = force_json_prompt(
            f"Instruction:\n{instr}\n\nText:\n{t}\n\nRange:\n[{mn}, {mx}]",
            suffix_strategy,
        )
        if response is not None:
            return [
                Message(role=Role.USER, content=input_text),
                Message(role=Role.ASSISTANT, content=f'{{"score": {response}}}'),
            ]
        else:
            return Message(role=Role.USER, content=input_text)

    messages = [
        Message(
            role=Role.SYSTEM,
            content='Score the following text with the provided instruction and range as an integer value in valid JSON:\n{"score": number}',
        ),
    ]
    if examples is not None:
        for example in examples:
            example_msgs = display_sample(
                instr=example.instruction,
                t=example.text,
                mn=example.min,
                mx=example.max,
                response=example.response,
            )
            assert isinstance(example_msgs, list) and all(
                isinstance(msg, Message) for msg in example_msgs
            ), example_msgs
            messages.extend(example_msgs)
    messages.append(display_sample(instr=instruction, t=text, mn=min, mx=max))
    return messages


def build_parse_state(
    text: str,
    typ: Union[type[T], Dict[str, Any]],
    examples: Optional[List[ParseDatapoint]] = None,
    suffix_strategy: PromptSuffixStrategy = PromptSuffixStrategy.JSON,
) -> List[Message]:
    def display_sample(
        t: str,
        ty: Union[type[T], Dict[str, Any]],
        response: Optional[Union[T, PartialObj, Dict[str, Any]]] = None,
    ) -> Union[Message, List[Message]]:
        if isinstance(ty, dict):
            json_schema_string = json.dumps(ty)
        else:
            optionalized_typ = optionalize_type(ty)
            json_schema_string = type_to_json_schema_string(optionalized_typ)
        input_text = force_json_prompt(
            f"Text:\n{t}\n\nSchema:\n{json_schema_string}",
            suffix_strategy=suffix_strategy,
        )
        if response is not None:
            if isinstance(response, dict):
                response_display = json.dumps(response)
            else:
                response_display = response.model_dump_json()
            return [
                Message(role=Role.USER, content=input_text),
                Message(role=Role.ASSISTANT, content=response_display),
            ]
        else:
            return Message(role=Role.USER, content=input_text)

    messages = [
        Message(
            role=Role.SYSTEM,
            content="Parse the following text with the provided JSON schema.",
        ),
    ]
    if examples is not None:
        for example in examples:
            example_msgs = display_sample(
                t=example.text, ty=typ, response=example.response
            )
            assert isinstance(example_msgs, list) and all(
                isinstance(msg, Message) for msg in example_msgs
            ), example_msgs
            messages.extend(example_msgs)
    messages.append(display_sample(t=text, ty=typ))
    return messages


def build_classify_state(
    instruction: str,
    text: str,
    options: List[str],
    examples: Optional[List[ClassifyDatapoint]] = None,
    suffix_strategy: PromptSuffixStrategy = PromptSuffixStrategy.JSON,
) -> Tuple[List[Message], Dict[str, int]]:
    def display_sample(
        instr: str, t: str, opts: List[str], response: Optional[int] = None
    ) -> Union[List[Message], Message]:
        choices_display, decode_map = display_choices(opts)
        input_text = force_json_prompt(
            f"Instruction:\n{instr}\n\nText:\n{t}\n\nChoices:\n{choices_display}",
            suffix_strategy=suffix_strategy,
        )
        if response is not None:
            label = None
            for k, v in decode_map.items():
                if v == response:
                    label = k
                    break
            assert label is not None
            return [
                Message(role=Role.USER, content=input_text),
                Message(
                    role=Role.ASSISTANT, content=f'{{"classification": "{label}"}}'
                ),
            ]
        else:
            return Message(role=Role.USER, content=input_text)

    messages = [
        Message(
            role=Role.SYSTEM,
            content='Classify the following text with the provided instruction and choices. To classify, provide the key of the choice:\n{"classification": string}\n\nFor example, if the correct choice is \'Z. description of choice Z\', then provide \'Z\' as the classification as valid JSON:\n```json\n{"classification": "Z"}\n```',
        ),
    ]
    if examples is not None:
        for example in examples:
            example_msgs = display_sample(
                instr=example.instruction,
                t=example.text,
                opts=example.options,
                response=example.response,
            )
            assert isinstance(example_msgs, list) and all(
                isinstance(msg, Message) for msg in example_msgs
            ), example_msgs
            messages.extend(example_msgs)
    prompt, decode_map = display_sample(instr=instruction, t=text, opts=options)
    messages.append(prompt)
    return messages, decode_map


class ChatModel(GeneralModel):
    @abc.abstractmethod
    def generate_message(
        self,
        messages: List[Message],
        force_json: bool,
        temperature: Optional[float] = None,
    ) -> Message:
        pass

    def handle_generate_message_response(
        self,
        prompt: List[Union[Dict[str, str], Message]],
        content: str,
        force_json: bool,
    ) -> Message:
        if force_json:
            try:
                obj = parse_json_or_json_markdown(content)
                return Message(role=Role.ASSISTANT, content=content, obj=obj)
            except Exception as e:
                raise ModelError(
                    f"Failed to parse response as JSON: {e}\n\nPrompt:\n{prompt}\n\nResponse:\n{content}"
                )
        else:
            return Message(role=Role.ASSISTANT, content=content)

    def build_generate_message_state(
        self, messages: List[Message]
    ) -> List[Dict[str, str]]:
        return [msg.model_dump() for msg in messages]

    def _handle_classify_response(
        self, res: Message, decode_map: Dict[str, int]
    ) -> int:
        try:
            obj = res.obj
            if obj is None:
                obj = parse_json_or_json_markdown(res.content)
            return obj["classification"]
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
        res = self.generate_message(
            messages=prompt,
            force_json=True,
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
        res = self.generate_message(
            messages=prompt,
            force_json=True,
            temperature=temperature,
        )
        obj = res.obj
        if obj is None:
            obj = parse_json_or_json_markdown(res.content)
        return json_response_to_obj_or_partial_obj(res=obj, typ=typ)

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
        res = self.generate_message(
            messages=prompt,
            force_json=False,
            temperature=temperature,
        )
        return res.content

    def _handle_parse_force_response(
        self, res: Message, typ: Union[type[T], Dict[str, Any]]
    ) -> Union[T, Dict[str, Any]]:
        obj = res.obj
        if obj is None:
            obj = parse_json_or_json_markdown(res.content)
        if isinstance(typ, dict):
            return obj
        try:
            return typ.model_validate(obj)
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
        res = self.generate_message(
            messages=prompt,
            force_json=True,
            temperature=temperature,
        )
        return self._handle_parse_force_response(res=res, typ=typ)

    def _handle_score_response(
        self,
        res: Message,
        min: int,
        max: int,
    ) -> int:
        try:
            obj = res.obj
            if obj is None:
                obj = parse_json_or_json_markdown(res.content)
            score = obj["score"]
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
        res = self.generate_message(
            messages=prompt,
            force_json=True,
            temperature=temperature,
        )
        return self._handle_score_response(res=res, min=min, max=max)


def build_prompts(
    dps: List[Datapoint], prompt_suffix_strategy: Optional[PromptSuffixStrategy]
) -> List[Union[str, List[Message]]]:
    prompts: List[Union[str, List[Message]]] = []
    for dp in dps:
        if isinstance(dp, ParseDatapoint):
            prompts.extend(build_parse_prompts([dp], prompt_suffix_strategy))
        elif isinstance(dp, BinaryClassifyDatapoint):
            prompts.extend(build_binary_classify_prompts([dp], prompt_suffix_strategy))
        elif isinstance(dp, ClassifyDatapoint):
            prompts.extend(build_classify_prompts([dp], prompt_suffix_strategy))
        elif isinstance(dp, ParseForceDatapoint):
            prompts.extend(build_parse_force_prompts([dp], prompt_suffix_strategy))
        elif isinstance(dp, GenerateDatapoint):
            prompts.extend(build_generate_prompts([dp]))
        elif isinstance(dp, ScoreDatapoint):
            prompts.extend(build_score_prompts([dp], prompt_suffix_strategy))
        else:
            raise ValueError(f"Unknown datapoint type: {type(dp)}")
    return prompts


def build_parse_prompts(
    dps: List[ParseDatapoint],
    suffix_strategy: Optional[PromptSuffixStrategy] = None,
) -> List[Union[str, List[Message]]]:
    prompts: List[Union[str, List[Message]]] = []
    for dp in dps:
        prompt = build_parse_state(
            text=dp.text,
            typ=dp.typ,
            examples=None,
            suffix_strategy=suffix_strategy,
        )
        if dp.response is not None:
            if isinstance(dp.response, dict):
                response_display = json.dumps(dp.response)
            else:
                response_display = dp.response.model_dump_json()
            prompt.append(Message(role=Role.ASSISTANT, content=response_display))
        prompts.append(prompt)
    return prompts


def build_binary_classify_prompts(
    dps: List[BinaryClassifyDatapoint],
    suffix_strategy: Optional[PromptSuffixStrategy] = None,
) -> List[Union[str, List[Message]]]:
    prompts: List[Union[str, List[Message]]] = []
    for dp in dps:
        prompt, decode_map = build_classify_state(
            instruction=dp.instruction,
            text=dp.text,
            options=["Yes", "No"],
            examples=None,
            suffix_strategy=suffix_strategy,
        )
        if dp.response is not None:
            label = "Yes" if dp.response else "No"
            prompt.append(
                Message(role=Role.ASSISTANT, content=f'{{"classification": "{label}"}}')
            )
        prompts.append(prompt)
    return prompts


def build_classify_prompts(
    dps: List[ClassifyDatapoint],
    suffix_strategy: Optional[PromptSuffixStrategy] = None,
) -> List[Union[str, List[Message]]]:
    prompts: List[Union[str, List[Message]]] = []
    for dp in dps:
        prompt, decode_map = build_classify_state(
            instruction=dp.instruction,
            text=dp.text,
            options=dp.options,
            examples=None,
            suffix_strategy=suffix_strategy,
        )
        if dp.response is not None:
            prompt.append(label_idx_to_label_json(dp.response, decode_map))
        prompts.append(prompt)
    return prompts


def label_idx_to_label_json(idx: int, decode_map: Dict[str, int]) -> Message:
    label = None
    for k, v in decode_map.items():
        if v == idx:
            label = k
            break
    assert label is not None
    return Message(role=Role.ASSISTANT, content=f'{{"classification": "{label}"}}')


def build_parse_force_prompts(
    dps: List[ParseForceDatapoint],
    suffix_strategy: Optional[PromptSuffixStrategy] = None,
) -> List[Union[str, List[Message]]]:
    prompts: List[Union[str, List[Message]]] = []
    for dp in dps:
        prompt = build_parse_force_state(
            instruction=dp.instruction,
            typ=dp.typ,
            text=dp.text,
            examples=None,
            suffix_strategy=suffix_strategy,
        )
        if dp.response is not None:
            if isinstance(dp.response, dict):
                response_display = json.dumps(dp.response)
            else:
                response_display = dp.response.model_dump_json()
            prompt.append(Message(role=Role.ASSISTANT, content=response_display))
        prompts.append(prompt)
    return prompts


def build_generate_prompts(
    dps: List[GenerateDatapoint],
) -> List[Union[str, List[Message]]]:
    prompts: List[Union[str, List[Message]]] = []
    for dp in dps:
        prompt = build_generate_state(
            instruction=dp.instruction,
            text=dp.text,
            examples=None,
        )
        if dp.response is not None:
            prompt.append(Message(role=Role.ASSISTANT, content=dp.response))
        prompts.append(prompt)
    return prompts


def build_score_prompts(
    dps: List[ScoreDatapoint],
    suffix_strategy: Optional[PromptSuffixStrategy] = None,
) -> List[Union[str, List[Message]]]:
    prompts: List[Union[str, List[Message]]] = []
    for dp in dps:
        prompt = build_score_state(
            instruction=dp.instruction,
            text=dp.text,
            min=dp.min,
            max=dp.max,
            examples=None,
            suffix_strategy=suffix_strategy,
        )
        if dp.response is not None:
            prompt.append(
                Message(role=Role.ASSISTANT, content=f'{{"score": {dp.response}}}')
            )
        prompts.append(prompt)
    return prompts


def apply_suffix_strategy(response: str, suffix_strategy: PromptSuffixStrategy) -> str:
    if suffix_strategy == PromptSuffixStrategy.JSON:
        return response
    elif suffix_strategy == PromptSuffixStrategy.JSON_MD_BLOCK:
        return add_md_tag(response)
    else:
        raise ValueError(f"Invalid suffix strategy: {suffix_strategy}")
