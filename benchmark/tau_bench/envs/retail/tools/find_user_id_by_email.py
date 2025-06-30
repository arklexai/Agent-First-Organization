from typing import Any

from benchmark.tau_bench.envs.tool import Tool


class FindUserIdByEmail(Tool):
    @staticmethod
    def invoke(data: dict[str, Any], email: str) -> str:
        users = data["users"]
        for user_id, profile in users.items():
            if profile["email"].lower() == email.lower():
                return user_id
        raise Exception("Error: user not found")

    @staticmethod
    def get_info() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "find_user_id_by_email",
                "description": "Find user id by email. If the user is not found, the function will return an error message.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "description": "The email of the user, such as 'something@example.com'.",
                        },
                    },
                    "required": ["email"],
                },
            },
        }
