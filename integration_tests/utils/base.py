import json
from enum import Enum
from typing import Any

from arklex.env.env import Environment
from arklex.orchestrator.orchestrator import AgentOrg
from arklex.utils.llm_config import LLMConfig


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class BaseTestOrchestrator:
    def __init__(self, config_file_path: str) -> None:
        with open(config_file_path) as f:
            config: dict[str, Any] = json.load(f)
        self.env: Environment = Environment(
            tools=config.get("tools", []),
            workers=config.get("workers", []),
            agents=config.get("agents", []),
            nodes=config.get("nodes", []),
            llm_config=LLMConfig.model_validate(config.get("model")),
        )
        self.orchestrator = AgentOrg(config=config, env=self.env)

    async def get_response(
        self, text: str, chat_history: list[dict[str, str]], parameters: dict[str, Any]
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": text,
            "chat_history": chat_history,
            "parameters": parameters,
        }
        return await self.orchestrator.get_response(data)

    @classmethod
    def init_params(cls) -> dict[str, Any]:
        return {
            "chat_history": [],
            "parameters": {},
        }
