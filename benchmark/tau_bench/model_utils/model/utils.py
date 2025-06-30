import enum
import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, Field

from benchmark.tau_bench.model_utils.api.types import PartialObj

T = TypeVar("T", bound=BaseModel)


class InputType(enum.Enum):
    CHAT = "chat"
    COMPLETION = "completion"


def display_choices(choices: list[str]) -> tuple[str, dict[str, int]]:
    choice_displays: list[str] = []
    decode_map: dict[str, int] = {}
    for i, choice in enumerate(choices):
        label: str = index_to_alpha(i)
        choice_display: str = f"{label}. {choice}"
        choice_displays.append(choice_display)
        decode_map[label] = i
    return "\n".join(choice_displays), decode_map


def index_to_alpha(index: int) -> str:
    alpha: str = ""
    while index >= 0:
        alpha = chr(index % 26 + ord("A")) + alpha
        index = index // 26 - 1
    return alpha


def type_to_json_schema_string(typ: type[T]) -> str:
    json_schema: dict[str, Any] = typ.model_json_schema()
    return json.dumps(json_schema, indent=4)


def optionalize_type(typ: type[T]) -> type[T]:
    class OptionalModel(typ): ...

    new_fields: dict[str, Any] = {}
    for name, field in OptionalModel.model_fields.items():
        new_fields[name] = Field(default=None, annotation=field.annotation | None)
    OptionalModel.model_fields = new_fields
    OptionalModel.__name__ = typ.__name__
    return OptionalModel


def json_response_to_obj_or_partial_obj(
    response: dict[str, Any], typ: type[T] | dict[str, Any]
) -> T | PartialObj | dict[str, Any]:
    if isinstance(typ, dict):
        return response
    else:
        required_field_names: list[str] = [
            name for name, field in typ.model_fields.items() if field.is_required()
        ]
        for name in required_field_names:
            if name not in response or response[name] is None:
                return response
        return typ.model_validate(response)


def clean_top_level_keys(d: dict[str, Any]) -> dict[str, Any]:
    new_d: dict[str, Any] = {}
    for k, v in d.items():
        new_d[k.strip()] = v
    return new_d


def parse_json_or_json_markdown(text: str) -> dict[str, Any]:
    def parse(s: str) -> dict[str, Any] | None:
        try:
            return json.loads(s)
        except json.decoder.JSONDecodeError:
            return None

    # pass #1: try to parse as json
    parsed: dict[str, Any] | None = parse(text)
    if parsed is not None:
        return parsed

    # pass #2: try to parse as json markdown
    stripped: str = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json") :].strip()
    if stripped.endswith("```"):
        stripped = stripped[: -len("```")].strip()
    parsed = parse(stripped)
    if parsed is not None:
        return parsed

    # pass #3: try to parse an arbitrary md block
    pattern: str = r"```(?:\w+\n)?(.*?)```"
    match: re.Match[str] | None = re.search(pattern, text, re.DOTALL)
    if match:
        content: str = match.group(1).strip()
        parsed = parse(content)
        if parsed is not None:
            return parsed

    # pass #4: try to parse arbitrary sections as json
    lines: list[str] = text.split("\n")
    seen: set[tuple[int, int]] = set()
    for i in range(len(lines)):
        for j in range(i + 1, len(lines) + 1):
            if i < j and (i, j) not in seen:
                seen.add((i, j))
                content = "\n".join(lines[i:j])
                parsed = parse(content)
                if parsed is not None:
                    return parsed
    raise ValueError("Could not parse JSON or JSON markdown")


def longest_valid_string(s: str, options: list[str]) -> str | None:
    longest: int = 0
    longest_str: str | None = None
    options_set: set[str] = set(options)
    for i in range(len(s)):
        if s[: i + 1] in options_set and i + 1 > longest:
            longest = i + 1
            longest_str = s[: i + 1]
    return longest_str


def try_classify_recover(s: str, decode_map: dict[str, int]) -> str | None:
    lvs: str | None = longest_valid_string(s, list(decode_map.keys()))
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
