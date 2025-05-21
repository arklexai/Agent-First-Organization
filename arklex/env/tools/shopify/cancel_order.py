import json
import shopify
import logging
import inspect
from typing import Any, Dict, List

# general GraphQL navigation utilities
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import ShopifyCancelOrderSlots, ShopifyOutputs

from arklex.env.tools.tools import register_tool
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt

logger = logging.getLogger(__name__)

description: str = "Cancel order by order id."
slots: List[Dict[str, Any]] = ShopifyCancelOrderSlots.get_all_slots()
outputs: List[Dict[str, Any]] = [
    ShopifyOutputs.CANECEL_REQUEST_DETAILS,
]


@register_tool(description, slots, outputs)
def cancel_order(cancel_order_id: str, **kwargs: Dict[str, Any]) -> str:
    func_name: str = inspect.currentframe().f_code.co_name
    auth: Dict[str, str] = authorify_admin(kwargs)

    try:
        with shopify.Session.temp(**auth):
            response: str = shopify.GraphQL().execute(f"""
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
            response: Dict[str, Any] = json.loads(response)["data"]
            if not response.get("orderCancel", {}).get("userErrors"):
                return "The order is successfully cancelled. " + json.dumps(response)
            else:
                raise ToolExecutionError(
                    func_name, json.dumps(response["orderCancel"]["userErrors"])
                )

    except Exception as e:
        raise ToolExecutionError(
            func_name, ShopifyExceptionPrompt.ORDER_CANCEL_ERROR_PROMPT
        )
