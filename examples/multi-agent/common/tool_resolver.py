from inspect import isclass

from agents import Tool, WebSearchTool
from tools import citation_finder

# A mapping from string names to actual tool functions
TOOL_REGISTRY = {
    "citation_finder": citation_finder,
    "web_search": WebSearchTool,
    # add more tools here
}


def resolve_tool(name: str) -> Tool:
    tool_cls_or_func = TOOL_REGISTRY.get(name)
    if tool_cls_or_func is None:
        return None
    if isclass(tool_cls_or_func):
        return tool_cls_or_func()  # instantiate if it's a class
    return tool_cls_or_func  # return function tool as is
