"""
This module provides functionality to cancel orders in the Shopify store using the Admin API.
It supports cancelling orders with automatic customer notification, restocking, and refunding.

Module Name: cancel_order

This file contains the code for cancelling orders using the Shopify Admin API.
"""

import inspect
import json
from typing import TypedDict

import shopify

from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import ShopifyCancelOrderSlots, ShopifyOutputs
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ShopifyError, ToolExecutionError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class CancelOrderParams(TypedDict, total=False):
    """Parameters for the cancel order tool."""

    shop_url: str
    access_token: str


description = "Cancel order by order id."
slots = ShopifyCancelOrderSlots.get_all_slots()
outputs = [ShopifyOutputs.CANECEL_REQUEST_DETAILS]


@register_tool(description, slots, outputs)
def cancel_order(cancel_order_id: str, **kwargs: CancelOrderParams) -> dict[str, str]:
    """
    Cancel an order in the Shopify store.

    Args:
        cancel_order_id (str): The ID of the order to cancel.
        **kwargs (CancelOrderParams): Additional keyword arguments for authentication.

    Returns:
        Dict[str, str]: A dictionary containing the cancellation result.

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
            if "orderCancel" not in response:
                raise ToolExecutionError(
                    func_name, "Invalid response: missing orderCancel key"
                )

            order_cancel_response = response["orderCancel"]
            user_errors = order_cancel_response.get("userErrors")
            if not user_errors:
                log_context.info(
                    f"Order cancellation completed for order: {cancel_order_id}"
                )
                return "The order is successfully cancelled. " + json.dumps(response)
            else:
                raise ToolExecutionError(
                    func_name, json.dumps(order_cancel_response["userErrors"])
                )

    except Exception as e:
        log_context.error(f"Order cancellation failed: {e}")
        raise ShopifyError("Order cancellation failed") from e
