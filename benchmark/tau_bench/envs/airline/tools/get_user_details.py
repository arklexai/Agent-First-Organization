import json
from typing import Any

from benchmark.tau_bench.envs.tool import Tool


class GetUserDetails(Tool):
    @staticmethod
    def invoke(data: dict[str, Any], user_id: str) -> str:
        users = data["users"]
        if user_id in users:
            return json.dumps(users[user_id])
        return "Error: user not found"

    @staticmethod
    def get_info() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_user_details",
                "description": "Get the details of an user, including their reservations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "The user id, such as 'sara_doe_496'.",
                        },
                    },
                    "required": ["user_id"],
                },
            },
        }
