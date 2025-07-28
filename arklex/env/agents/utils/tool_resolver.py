import importlib
import inspect
import os
from inspect import isclass
from typing import Any

from agents import (
    CodeInterpreterTool,
    FunctionTool,
    RunContextWrapper,
    Tool,
    WebSearchTool,
)
from pydantic import BaseModel, Field, create_model
from undecorated import undecorated

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

# OpenAI Agent SDK built-in tools
BUILT_IN_TOOLS = {
    "web_search": WebSearchTool,
    "code_interpreter": CodeInterpreterTool(
        tool_config={"type": "code_interpreter", "container": {"type": "auto"}}
    ),
}


def resolve_tools_for_agent(tool_specs: list[Any]) -> list[Tool]:
    resolved_tools = []
    for spec in tool_specs:
        if isinstance(spec, dict):
            tool_id = spec["id"]
            path = spec.get("path")
            fixed_args = spec.get("fixed_args", {})
        else:
            log_context.warning(f"[WARN] Invalid tool spec: {spec}")
            continue

        # Filter out placeholder values
        filtered_args = {}
        for key, value in fixed_args.items():
            if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
                log_context.warning(
                    f"[WARN] Placeholder value used for fixed_arg '{key}'. Skipping."
                )
                continue
            filtered_args[key] = value

        tool = resolve_tool(tool_id, path, fixed_args=filtered_args)
        if tool:
            resolved_tools.append(tool)
        else:
            log_context.warning(f"[WARN] Tool '{tool_id}' could not be resolved.")

    return resolved_tools


def resolve_tool(
    tool_id: str, path: str | None, fixed_args: dict | None = None
) -> Tool | None:
    try:
        if path is None:
            tool_func_or_cls = BUILT_IN_TOOLS.get(tool_id)
        else:
            filepath = os.path.join("arklex.env.tools", path)
            module_name = filepath.replace(os.sep, ".").replace(".py", "")
            module = importlib.import_module(module_name)
            tool_func_or_cls = getattr(module, tool_id)

            if fixed_args and inspect.isfunction(tool_func_or_cls):
                # extract the slots
                tool_instance = tool_func_or_cls()
                slots = tool_instance.slots

                # wrapped with register_tool wrapper, we want to undecorate
                try:
                    tool_func_or_cls = undecorated(tool_func_or_cls)
                except Exception:
                    tool_func_or_cls = tool_func_or_cls

                sig = inspect.signature(tool_func_or_cls)
                user_param_names = [
                    name
                    for name, param in sig.parameters.items()
                    if name not in fixed_args
                    and param.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                ]

                # Build a dynamic Pydantic model class for user input
                fields = {}
                for slot in slots:
                    if slot.name in user_param_names:
                        py_type = python_type_from_str(slot.type)
                        default = None if not slot.required else ...
                        metadata = {"description": slot.description}
                        if slot.enum:
                            metadata["enum"] = slot.enum

                        fields[slot.name] = (py_type, Field(default, **metadata))

                model_cls = create_model(f"{tool_id}_InputModel", **fields)

                return wrap_function_tool_with_fixed_args(
                    base_func=tool_func_or_cls,
                    model_cls=model_cls,
                    fixed_args=fixed_args,
                    name=tool_id,
                    description=f"Dynamically wrapped tool {tool_id}",
                )

        return tool_func_or_cls() if isclass(tool_func_or_cls) else tool_func_or_cls
    except Exception as e:
        log_context.error(
            f"[ERROR] Could not load tool '{tool_id}' from path '{path}': {e}"
        )
        return None


def wrap_function_tool_with_fixed_args(
    base_func: callable,
    model_cls: type[BaseModel],
    fixed_args: dict[str, Any],
    name: str,
    description: str,
) -> FunctionTool:
    async def on_invoke(ctx: RunContextWrapper[Any], raw_args: str) -> str:
        user_args = model_cls.model_validate_json(raw_args)
        merged_args = {**fixed_args, **user_args.model_dump()}
        return base_func(**merged_args)

    return FunctionTool(
        name=name,
        description=description,
        params_json_schema=model_cls.model_json_schema(),
        on_invoke_tool=on_invoke,
        strict_json_schema=False,
    )


def python_type_from_str(type_str: str) -> type:
    mapping = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "dict": dict,
        "list": list,
    }
    return mapping.get(type_str, Any)
