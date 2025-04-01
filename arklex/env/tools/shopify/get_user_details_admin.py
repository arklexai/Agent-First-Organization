from typing import Any, Dict
import json

from arklex.env.tools.tools import register_tool

from arklex.env.tools.shopify.utils_slots import ShopifyGetUserDetailsAdminSlots, ShopifyOutputs
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import authorify_admin

from arklex.exceptions import ToolExecutionError

# Admin API
import shopify

description = "Get the details of a user with Admin API."
slots = ShopifyGetUserDetailsAdminSlots.get_all_slots()
outputs = [
    ShopifyOutputs.USER_DETAILS,
    *PAGEINFO_OUTPUTS
]

USER_NOT_FOUND_PROMPT = "Could not find the user. Please try again later or refresh the chat window."


@register_tool(description, slots, outputs)
def get_user_details_admin(user_id: str, **kwargs) -> str:
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
                raise ToolExecutionError(f"get_user_details_admin failed", USER_NOT_FOUND_PROMPT)

    except Exception as e:
        raise ToolExecutionError(f"get_user_details_admin failed: {e}", USER_NOT_FOUND_PROMPT)