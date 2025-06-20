"""
This module provides functionality to cancel orders in the Shopify store using the Admin API.
It supports cancelling orders with automatic customer notification, restocking, and refunding.

Module Name: cancel_order

This file contains the code for cancelling orders using the Shopify Admin API.
"""

import json
from typing import Any, Dict

import shopify
from arklex.utils.logging_utils import LogContext
from arklex.utils.exceptions import ShopifyError

# general GraphQL navigation utilities
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import ShopifyCancelOrderSlots, ShopifyOutputs

from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError

log_context = LogContext(__name__)

description = "Cancel order by order id."
slots = ShopifyCancelOrderSlots.get_all_slots()
outputs = [ShopifyOutputs.CANECEL_REQUEST_DETAILS]


@register_tool(description, slots, outputs)
def cancel_order(cancel_order_id: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Cancel an order in the Shopify store.

    Args:
        cancel_order_id (str): The ID of the order to cancel.
        **kwargs (Any): Additional keyword arguments for authentication.

    Returns:
        Dict[str, Any]: A dictionary containing the cancellation result.

    Raises:
        ShopifyError: If cancellation fails
    """
    try:
        log_context.info(f"Starting order cancellation for order: {cancel_order_id}")
        func_name = inspect.currentframe().f_code.co_name
        auth = authorify_admin(kwargs)

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
                log_context.info(
                    f"Order cancellation completed for order: {cancel_order_id}"
                )
                return "The order is successfully cancelled. " + json.dumps(response)
            else:
                raise ToolExecutionError(
                    func_name, json.dumps(response["orderCancel"]["userErrors"])
                )

    except Exception as e:
        log_context.error(f"Order cancellation failed: {e}")
        raise ShopifyError("Order cancellation failed") from e
