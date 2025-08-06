from unittest.mock import MagicMock, Mock, patch

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from agents import FunctionTool, WebSearchTool

from arklex.env.agents.utils import tool_resolver


@pytest.mark.parametrize(
    "type_str,expected_type",
    [
        ("str", str),
        ("int", int),
        ("float", float),
        ("bool", bool),
        ("dict", dict),
        ("list", list),
        ("nonexistent", tool_resolver.Any),
    ],
)
def test_python_type_from_str(type_str: str, expected_type: type) -> None:
    assert tool_resolver.python_type_from_str(type_str) == expected_type


def test_resolve_tools_for_agent_valid_dict(monkeypatch: MonkeyPatch) -> None:
    mock_tool = MagicMock()
    monkeypatch.setattr(tool_resolver, "resolve_tool", lambda *a, **kw: mock_tool)

    specs = [{"id": "tool_x", "path": "fake_path", "fixed_args": {"arg": "value"}}]
    resolved = tool_resolver.resolve_tools_for_agent(specs)
    assert len(resolved) == 1
    assert resolved[0] == mock_tool


def test_resolve_tools_for_agent_invalid_spec(caplog: LogCaptureFixture) -> None:
    result = tool_resolver.resolve_tools_for_agent(["invalid_string"])
    assert result == []
    assert "[WARN] Invalid tool spec" in caplog.text


def test_resolve_tools_for_agent_skips_placeholders(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(tool_resolver, "resolve_tool", lambda *a, **kw: None)

    specs = [
        {"id": "tool_y", "path": "fake_path", "fixed_args": {"token": "<PLACEHOLDER>"}}
    ]
    resolved = tool_resolver.resolve_tools_for_agent(specs)
    assert resolved == []


@patch("arklex.env.agents.utils.tool_resolver.importlib.import_module")
def test_resolve_tool_builtin(mock_import: MagicMock) -> None:
    result = tool_resolver.resolve_tool("web_search", path=None)
    assert isinstance(result, WebSearchTool)


def test_resolve_tool_dynamic_function_tool_real() -> None:
    tool = tool_resolver.resolve_tool(
        "search_products",
        path="shopify/search_products.py",
        fixed_args={
            "llm_provider": "openai",
            "model_type_or_path": "gpt-4o-mini",
        },
    )

    assert isinstance(tool, FunctionTool)
    assert hasattr(tool, "on_invoke_tool")
    assert callable(tool.on_invoke_tool)


@pytest.mark.asyncio
async def test_wrap_function_tool_with_fixed_args_execution() -> None:
    def dummy_func(foo: str, bar: str) -> str:
        return f"{foo}-{bar}"

    InputModel = tool_resolver.create_model("TestModel", foo=(str, ...))

    tool = tool_resolver.wrap_function_tool_with_fixed_args(
        base_func=dummy_func,
        model_cls=InputModel,
        fixed_args={"bar": "XYZ"},
        name="my_tool",
        description="desc",
    )

    assert isinstance(tool, FunctionTool)

    # Now invoke the tool asynchronously
    result = await tool.on_invoke_tool(Mock(), '{"foo": "123"}')
    assert result == "123-XYZ"


@patch(
    "arklex.env.agents.utils.tool_resolver.importlib.import_module",
    side_effect=ImportError("fail"),
)
def test_resolve_tool_import_error(mock_import: MagicMock) -> None:
    result = tool_resolver.resolve_tool("some_tool", path="bad_path")
    assert result is None
