import json
from enum import Enum
from typing import Any

from arklex.models.llm_config import LLMConfig
from arklex.orchestrator.executor.executor import Executor
from arklex.orchestrator.orchestrator import AgentOrg


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class BaseTestOrchestrator:
    def __init__(self, config_file_path: str) -> None:
        with open(config_file_path) as f:
            config: dict[str, Any] = json.load(f)
        self.executor: Executor = Executor(
            tools=config.get("tools", []),
            workers=config.get("workers", []),
            nodes=config.get("nodes", []),
            llm_config=LLMConfig.model_validate(config.get("llm_config")),
        )
        self.orchestrator = AgentOrg(config=config, executor=self.executor)

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
