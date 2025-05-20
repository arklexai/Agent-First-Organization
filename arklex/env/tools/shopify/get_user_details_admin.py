from typing import Any, Dict, List, Tuple
import json
import inspect
import shopify
from arklex.env.tools.tools import register_tool
from arklex.env.tools.shopify.utils_slots import (
    ShopifyGetUserDetailsAdminSlots,
    ShopifyOutputs,
)
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt
from arklex.exceptions import ToolExecutionError

description: str = "Get the details of a user with Admin API."
slots: List[Dict[str, Any]] = ShopifyGetUserDetailsAdminSlots.get_all_slots()
outputs: List[Dict[str, Any]] = [ShopifyOutputs.USER_DETAILS, *PAGEINFO_OUTPUTS]


@register_tool(description, slots, outputs)
def get_user_details_admin(user_id: str, **kwargs: Any) -> str:
    func_name: str = inspect.currentframe().f_code.co_name
    nav: Tuple[str, bool] = cursorify(kwargs)
    if not nav[1]:
        return nav[0]
    auth: Dict[str, str] = authorify_admin(kwargs)

    try:
        with shopify.Session.temp(**auth):
            response: str = shopify.GraphQL().execute(f"""
                {{
                    customer(id: "{user_id}")  {{ 
                        firstName
                        lastName
                        email
                        phone
                        numberOfOrders
                        amountSpent {{
                            amount
                            currencyCode
                        }}
                        createdAt
                        updatedAt
                        note
                        verifiedEmail
                        validEmailAddress
                        tags
                        lifetimeDuration
                        addresses {{
                            address1
                        }}
                        orders ({nav[0]}) {{
                            nodes {{
                                id
                            }}
                        }}
                    }}
                }}
            """)
            data: Dict[str, Any] = json.loads(response)["data"]["customer"]
            if data:
                return json.dumps(data)
            else:
                raise ToolExecutionError(
                    func_name, ShopifyExceptionPrompt.USER_NOT_FOUND_PROMPT
                )

    except Exception as e:
        raise ToolExecutionError(
            func_name, ShopifyExceptionPrompt.USER_NOT_FOUND_PROMPT
        )
