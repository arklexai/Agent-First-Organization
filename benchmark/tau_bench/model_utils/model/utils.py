import enum
import json
import re
from typing import Any, Optional, TypeVar, Dict, List, Set, Tuple, Union

from pydantic import BaseModel, Field

from benchmark.tau_bench.model_utils.api.types import PartialObj

T = TypeVar("T", bound=BaseModel)


class InputType(enum.Enum):
    CHAT = "chat"
    COMPLETION = "completion"


def display_choices(choices: List[str]) -> Tuple[str, Dict[str, int]]:
    choice_displays: List[str] = []
    decode_map: Dict[str, int] = {}
    for i, choice in enumerate(choices):
        label = index_to_alpha(i)
        choice_display = f"{label}. {choice}"
        choice_displays.append(choice_display)
        decode_map[label] = i
    return "\n".join(choice_displays), decode_map


def index_to_alpha(index: int) -> str:
    alpha = ""
    while index >= 0:
        alpha = chr(index % 26 + ord("A")) + alpha
        index = index // 26 - 1
    return alpha


def type_to_json_schema_string(typ: type[T]) -> str:
    json_schema = typ.model_json_schema()
    return json.dumps(json_schema, indent=4)


def optionalize_type(typ: type[T]) -> type[T]:
    class OptionalModel(typ): ...

    new_fields: Dict[str, Any] = {}
    for name, field in OptionalModel.model_fields.items():
        new_fields[name] = Field(default=None, annotation=Optional[field.annotation])
    OptionalModel.model_fields = new_fields
    OptionalModel.__name__ = typ.__name__
    return OptionalModel


def json_response_to_obj_or_partial_obj(
    response: Dict[str, Any], typ: Union[type[T], Dict[str, Any]]
) -> Union[T, PartialObj, Dict[str, Any]]:
    if isinstance(typ, dict):
        return response
    else:
        required_field_names = [
            name for name, field in typ.model_fields.items() if field.is_required()
        ]
        for name in required_field_names:
            if name not in response.keys() or response[name] is None:
                return response
        return typ.model_validate(response)


def clean_top_level_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    new_d: Dict[str, Any] = {}
    for k, v in d.items():
        new_d[k.strip()] = v
    return new_d


def parse_json_or_json_markdown(text: str) -> Dict[str, Any]:
    def parse(s: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(s)
        except json.decoder.JSONDecodeError:
            return None

    # pass #1: try to parse as json
    parsed = parse(text)
    if parsed is not None:
        return parsed

    # pass #2: try to parse as json markdown
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json") :].strip()
    if stripped.endswith("```"):
        stripped = stripped[: -len("```")].strip()
    parsed = parse(stripped)
    if parsed is not None:
        return parsed

    # pass #3: try to parse an arbitrary md block
    pattern = r"```(?:\w+\n)?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        parsed = parse(content)
        if parsed is not None:
            return parsed

    # pass #4: try to parse arbitrary sections as json
    lines = text.split("\n")
    seen: Set[Tuple[int, int]] = set()
    for i in range(len(lines)):
        for j in range(i + 1, len(lines) + 1):
            if i < j and (i, j) not in seen:
                seen.add((i, j))
                content = "\n".join(lines[i:j])
                parsed = parse(content)
                if parsed is not None:
                    return parsed
    raise ValueError("Could not parse JSON or JSON markdown")


def longest_valid_string(s: str, options: List[str]) -> Optional[str]:
    longest = 0
    longest_str = None
    options_set = set(options)
    for i in range(len(s)):
        if s[: i + 1] in options_set and i + 1 > longest:
            longest = i + 1
            longest_str = s[: i + 1]
    return longest_str


def try_classify_recover(s: str, decode_map: Dict[str, int]) -> Optional[str]:
    lvs = longest_valid_string(s, list(decode_map.keys()))
    if lvs is not None and lvs in decode_map:
        return lvs
    for k, v in decode_map.items():
        if s == v:
            return k
    return None


def approx_num_tokens(text: str) -> int:
    return len(text) // 4


def add_md_close_tag(prompt: str) -> str:
    return f"{prompt}\n```"


def add_md_tag(prompt: str) -> str:
    return f"```json\n{prompt}\n```"
