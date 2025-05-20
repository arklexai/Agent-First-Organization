from typing import Any, Dict, List
import json
import inspect
import shopify

from arklex.env.tools.tools import register_tool
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import (
    ShopifyFindUserByEmailSlots,
    ShopifyOutputs,
)
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt

description: str = "Find user id by email. If the user is not found, the function will return an error message."
slots: List[Dict[str, Any]] = ShopifyFindUserByEmailSlots.get_all_slots()
outputs: List[Dict[str, Any]] = [ShopifyOutputs.USER_ID]


@register_tool(description, slots, outputs)
def find_user_id_by_email(user_email: str, **kwargs: Any) -> str:
    func_name: str = inspect.currentframe().f_code.co_name
    auth: Dict[str, str] = authorify_admin(kwargs)

    try:
        with shopify.Session.temp(**auth):
            response: str = shopify.GraphQL().execute(f"""
                {{
                    customers (first: 10, query: "email:{user_email}") {{
                        edges {{
                            node {{
                                id
                            }}
                        }}
                    }}
                }}
                """)
        nodes: List[Dict[str, Any]] = json.loads(response)["data"]["customers"]["edges"]
        if len(nodes) == 1:
            user_id: str = nodes[0]["node"]["id"]
            return user_id
        else:
            raise ToolExecutionError(
                func_name, ShopifyExceptionPrompt.MULTIPLE_USERS_SAME_EMAIL_ERROR_PROMPT
            )
    except Exception as e:
        raise ToolExecutionError(
            func_name, ShopifyExceptionPrompt.USER_NOT_FOUND_ERROR_PROMPT
        )
