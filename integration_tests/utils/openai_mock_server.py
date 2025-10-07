import uuid
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

app = FastAPI()


class Message(BaseModel):
    role: str = Field(default="")
    content: str = Field(default="")
    tool_calls: list[dict[str, Any]] = Field(default=[])


class ChatChoices(BaseModel):
    message: Message


class ChatCompletion(BaseModel):
    choices: list[ChatChoices]


@app.post("/chat/completions")
async def chat_completions(request: Request) -> ChatCompletion:
    request_body = await request.json()
    # stream = request_body.get("stream", False)

    # Extract the last user message
    user_messages = [
        m for m in request_body.get("messages", []) if m.get("role") == "user"
    ]
    last_user_message = user_messages[-1]["content"]
    system_messages = [
        m for m in request_body.get("messages", []) if m.get("role") == "system"
    ]
    last_system_message = system_messages[-1]["content"] if system_messages else ""
    assistant_messages = [
        m for m in request_body.get("messages", []) if m.get("role") == "assistant"
    ]
    last_assistant_message = (
        assistant_messages[-1]["content"] if assistant_messages else ""
    )
    print(f"Last user message: {last_user_message}")
    print(request_body)
    print(f"Last system message: {last_system_message}")
    print(f"Last assistant message: {last_assistant_message}")

    # Mock intent detection
    if last_user_message.startswith(
        "Given the following intents and their definitions"
    ):
        print("Intent detection request received")
        # Intent detection for test_worker_taskgraph.py
        if last_user_message.endswith("user: How is the weather?\n"):
            return ChatCompletion(
                choices=[
                    ChatChoices(
                        message=Message(
                            tool_calls=[
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "IntentDetectionOutput",
                                        "arguments": '{"intent": "ask about weather"}',
                                    },
                                }
                            ],
                            role="assistant",
                        )
                    )
                ]
            )
        elif last_user_message.endswith("user: Which car would you like to buy?\n"):
            return ChatCompletion(
                choices=[
                    ChatChoices(
                        message=Message(
                            tool_calls=[
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "IntentDetectionOutput",
                                        "arguments": '{"intent": "ask about car options"}',
                                    },
                                }
                            ],
                            role="assistant",
                        )
                    )
                ]
            )
        elif last_user_message.endswith("user: Connect me with a human agent\n"):
            return ChatCompletion(
                choices=[
                    ChatChoices(
                        message=Message(
                            tool_calls=[
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "IntentDetectionOutput",
                                        "arguments": '{"intent": "user want to connect with a human agent"}',
                                    },
                                }
                            ],
                            role="assistant",
                        )
                    )
                ]
            )
        # Intent detection for http_tool_agent_taskgraph.json
        elif last_user_message.endswith(
            "user: What services does your company provide?\n"
        ) or last_user_message.endswith(
            "user: I'm interested in user simulator with a budget of 1000\n"
        ):
            return ChatCompletion(
                choices=[
                    ChatChoices(
                        message=Message(
                            tool_calls=[
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "IntentDetectionOutput",
                                        "arguments": '{"intent": "others"}',
                                    },
                                }
                            ],
                            role="assistant",
                        )
                    )
                ]
            )

        else:
            raise ValueError(f"Unknown intent detection request: {last_user_message}")

    # Mock worker response generation
    elif last_user_message.endswith("assistant:\n"):
        print("Worker response generation request received")
        return ChatCompletion(
            choices=[
                ChatChoices(
                    message=Message(
                        role="assistant", content="This is a demo worker response."
                    )
                )
            ]
        )

    # Mock agent function calling
    elif (
        last_system_message
        == "You are an agent developed by Arklex. Follow these steps carefully:\n\n- Use queryService to get all services provided by Arklex.\n- Use contactTeam to submit service requests."
    ):
        print("Agent function calling request received")
        # Agent function calling for http_tool_agent_taskgraph.json
        if last_user_message == "What services does your company provide?":
            if last_assistant_message != "Calling tool: queryService":
                return ChatCompletion(
                    choices=[
                        ChatChoices(
                            message=Message(
                                role="assistant",
                                content="",
                                tool_calls=[
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": "queryService",
                                            "arguments": "{}",
                                        },
                                        "id": f"call_{str(uuid.uuid4()).replace('-', '')}",
                                    }
                                ],
                            )
                        )
                    ]
                )
            else:
                return ChatCompletion(
                    choices=[
                        ChatChoices(
                            message=Message(
                                role="assistant",
                                content="Arklex provides graph-based chatbots, agents, and user simulators.",
                            )
                        )
                    ]
                )
        elif (
            last_user_message
            == "I'm interested in user simulator with a budget of 1000"
        ):
            if last_assistant_message != "Calling tool: contactTeam":
                return ChatCompletion(
                    choices=[
                        ChatChoices(
                            message=Message(
                                role="assistant",
                                content="",
                                tool_calls=[
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": "contactTeam",
                                            "arguments": '{"ServiceInfo":[{"budget":1000,"service":"user simulator"}]}',
                                        },
                                        "id": f"call_{str(uuid.uuid4()).replace('-', '')}",
                                    }
                                ],
                            )
                        )
                    ]
                )
            else:
                return ChatCompletion(
                    choices=[
                        ChatChoices(
                            message=Message(
                                role="assistant",
                                content="Service request for user simulator with a budget of 1000 submitted successfully.",
                            )
                        )
                    ]
                )

    else:
        raise ValueError(f"Unknown request type: {last_user_message}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5005,
    )
