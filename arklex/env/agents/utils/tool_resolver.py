import importlib
import os
from inspect import isclass
from typing import Any

from agents import Tool, WebSearchTool

# OpenAI Agent SDK built-in tools
BUILT_IN_TOOLS = {
    "web_search": WebSearchTool,
}


def resolve_tools_for_agent(tool_specs: list[Any]) -> list[Tool]:
    resolved_tools = []
    for spec in tool_specs:
        if isinstance(spec, str):
            tool_id, path = spec, None
        elif isinstance(spec, dict):
            tool_id, path = spec["id"], spec.get("path")
        else:
            print(f"[WARN] Invalid tool spec: {spec}")
            continue

        tool = resolve_tool(tool_id, path)
        if tool:
            resolved_tools.append(tool)
        else:
            print(f"[WARN] Tool '{tool_id}' could not be resolved.")
    return resolved_tools


def resolve_tool(tool_id: str, path: str | None) -> Tool | None:
    try:
        if path is None:
            tool_obj = BUILT_IN_TOOLS.get(tool_id)
            if tool_obj is None:
                raise ValueError(f"Special tool '{tool_id}' is not registered.")
            return tool_obj() if isclass(tool_obj) else tool_obj
        else:
            filepath: str = os.path.join("arklex.env.tools", path)
            module_name: str = filepath.replace(os.sep, ".").replace(".py", "")
            module = importlib.import_module(module_name)
            tool_func_or_cls = getattr(module, tool_id)
            return tool_func_or_cls() if isclass(tool_func_or_cls) else tool_func_or_cls
    except Exception as e:
        print(f"[ERROR] Could not load tool '{tool_id}' from path '{path}': {e}")
        return None
