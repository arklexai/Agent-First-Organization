from typing import Any, Dict
import json

from arklex.env.tools.tools import register_tool

from arklex.env.tools.shopify.utils_slots import ShopifyGetUserDetailsAdminSlots, ShopifyOutputs
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify._exception_prompt import ExceptionPrompt
from arklex.exceptions import ToolExecutionError

import inspect

# Admin API
import shopify

description = "Get the details of a user with Admin API."
slots = ShopifyGetUserDetailsAdminSlots.get_all_slots()
outputs = [
    ShopifyOutputs.USER_DETAILS,
    *PAGEINFO_OUTPUTS
]


@register_tool(description, slots, outputs)
def get_user_details_admin(user_id: str, **kwargs) -> str:
    func_name = inspect.currentframe().f_code.co_name
    nav = cursorify(kwargs)
    if not nav[1]:
        return nav[0]
    auth = authorify_admin(kwargs)
    
    try:
        with shopify.Session.temp(**auth):
            response = shopify.GraphQL().execute(f"""
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
            data = json.loads(response)['data']['customer']
            if data:
                return json.dumps(data)
            else:
                raise ToolExecutionError(func_name, ExceptionPrompt.USER_NOT_FOUND_PROMPT)

    except Exception as e:
        raise ToolExecutionError(func_name, ExceptionPrompt.USER_NOT_FOUND_PROMPT)