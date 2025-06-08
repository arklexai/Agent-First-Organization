"""
This module provides functionality to cancel orders in the Shopify store using the Admin API.
It supports cancelling orders with automatic customer notification, restocking, and refunding.

Module Name: cancel_order

This file contains the code for cancelling orders using the Shopify Admin API.
"""

import json
from typing import Any

import shopify
import logging
import inspect

# general GraphQL navigation utilities
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import ShopifyCancelOrderSlots, ShopifyOutputs

from arklex.env.tools.tools import register_tool
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt

logger = logging.getLogger(__name__)

description = "Cancel order by order id."
slots = ShopifyCancelOrderSlots.get_all_slots()
outputs = [ShopifyOutputs.CANECEL_REQUEST_DETAILS]


@register_tool(description, slots, outputs)
def cancel_order(cancel_order_id: str, **kwargs: Any) -> str:
    """
    Cancel an order in the Shopify store.

    Args:
        cancel_order_id (str): The ID of the order to cancel.
        **kwargs (Any): Additional keyword arguments for authentication.

    Returns:
        str: A success message with the cancellation details if successful.

    Raises:
        ToolExecutionError: If:
            - The order cannot be cancelled
            - There are user errors during cancellation
            - There's an error in the cancellation process
    """
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
                raise ToolExecutionError(
                    func_name, json.dumps(response["orderCancel"]["userErrors"])
                )

    except Exception as e:
        raise ToolExecutionError(
            func_name, ShopifyExceptionPrompt.ORDER_CANCEL_ERROR_PROMPT
        )
