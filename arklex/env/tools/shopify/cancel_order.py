import json
import shopify
import logging
import inspect

# general GraphQL navigation utilities
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import ShopifyCancelOrderSlots, ShopifyOutputs

from arklex.env.tools.tools import register_tool
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.shopify._exception_prompt import ExceptionPrompt
logger = logging.getLogger(__name__)

description = "Cancel order by order id."
slots = ShopifyCancelOrderSlots.get_all_slots()
outputs = [
    ShopifyOutputs.CANECEL_REQUEST_DETAILS,
]


@register_tool(description, slots, outputs)
def cancel_order(cancel_order_id: str, **kwargs) -> str:
    func_name = inspect.currentframe().f_code.co_name
    auth = authorify_admin(kwargs)
    
    try:
        with shopify.Session.temp(**auth):
            response = shopify.GraphQL().execute(f"""
            mutation orderCancel {{
            orderCancel(
                orderId: "{cancel_order_id}",
                reason: CUSTOMER,
                notifyCustomer: true,
                restock: true,
                refund: true
            ) {{
                userErrors {{
                    field
                    message
                }}
            }}
            }}
            """)
            response = json.loads(response)["data"]
            if not response.get("orderCancel", {}).get("userErrors"):
                return "The order is successfully cancelled. " + json.dumps(response)
            else:
                raise ToolExecutionError(func_name, json.dumps(response["orderCancel"]["userErrors"]))
    
    except Exception as e:
        raise ToolExecutionError(func_name, ExceptionPrompt.ORDER_CANCEL_ERROR_PROMPT)