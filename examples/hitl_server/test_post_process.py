# Go to the parent folder of this file (hitl_server), then Run `pytest test_post_process.py` to test the code in this file.
import io
import json
import logging
import os
from contextlib import redirect_stdout
from typing import Any

import pytest

from arklex.env.env import Environment
from arklex.orchestrator.NLU.services.model_service import ModelService
from run import get_api_bot_response

TRIGGER_LIVE_CHAT_PROMPT = "Sorry, I'm not certain about the answer, would you like to connect to a human assistant?"

with open("test_cases_post_process.json", encoding="utf-8") as f:
    TEST_CASES = json.load(f)


@pytest.fixture(scope="session")
def config_and_env(
    request: pytest.FixtureRequest,
) -> tuple[dict[str, Any], Environment, str]:
    """Load config and environment once per test session."""
    with open("taskgraph.json", encoding="utf-8") as f:
        config = json.load(f)

    model = {
        "model_name": str(request.config.getoption("--model")),
        "model_type_or_path": str(request.config.getoption("--model")),
        "llm_provider": str(request.config.getoption("--llm_provider")),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "endpoint": "https://api.openai.com/v1",
    }
    config["input_dir"] = "./"
    config["model"] = model
    os.environ["DATA_DIR"] = config["input_dir"]

    # Initialize model service
    model_service = ModelService(model)

    env = Environment(
        tools=config.get("tools", []),
        workers=config.get("workers", []),
        agents=config.get("agents", []),
        slot_fill_api=config["slotfillapi"],
        planner_enabled=True,
        model_service=model_service,
    )

    for node in config["nodes"]:
        if node[1].get("type", "") == "start":
            start_message = node[1]["attribute"]["value"]
            break
    else:
        raise ValueError("Start node not found in taskgraph.json")

    return config, env, start_message


class TestLiveChatDetection:
    def setup_method(self) -> None:
        self.params: dict[str, Any] = {}

    # run the same test function with multiple sets of arguments
    @pytest.mark.parametrize(
        "test_case",
        TEST_CASES,
        ids=lambda tc: f"id_{tc['test_id']}_{tc['category']}",
    )
    def test_live_chat_detection(
        self,
        test_case: dict[str, Any],
        config_and_env: tuple[dict[str, Any], Environment, str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        config, env, start_message = config_and_env
        user_text = test_case["user_utterance"]
        expected_live_chat = test_case["expect_live_chat"]
        expected_log_message = test_case.get("expect_log_message")

        history: list[dict[str, str]] = [
            {"role": "assistant", "content": start_message}
        ]

        # Configure caplog to capture INFO level messages from all loggers
        caplog.set_level(logging.INFO)

        captured_stdout_sio = io.StringIO()
        with redirect_stdout(captured_stdout_sio):
            output, self.params, hitl = get_api_bot_response(
                config=config,
                history=history,
                user_text=user_text,
                parameters=self.params,
                env=env,
            )

        live_chat_triggered = hitl is not None
        stdout_output = captured_stdout_sio.getvalue()
        logs_output = caplog.text
        full_captured_output = stdout_output + logs_output

        # don't trigger live chat on follow up responses or clear responses
        if not expected_live_chat:
            assert live_chat_triggered == expected_live_chat, (
                f"\nTest Failed\nExpected live chat: {expected_live_chat}, got: {live_chat_triggered}\n"
                f"User: {user_text}\nBot: {output}\nFull Captured Output:\n{full_captured_output}"
            )

        # prompt user to chat with human assistant
        if expected_live_chat:
            assert output == TRIGGER_LIVE_CHAT_PROMPT

        # for irrelevant questions
        if expected_log_message:
            assert expected_log_message in logs_output, (
                f"\nTest Failed: Missing expected log message.\n"
                f"Expected: '{expected_log_message}'\n"
                f"Actual Logs:\n{logs_output}"
            )
