"""
This module provides functionality to retrieve detailed information about orders from the Shopify store.
It supports filtering orders by customer ID, order IDs, and order names.

Module Name: get_order_details

This file contains the code for retrieving detailed order information from Shopify.
"""

import inspect
import json

import shopify
from pydantic import BaseModel

from arklex.resources.tools.shopify._exception_prompt import ShopifyExceptionPrompt
from arklex.resources.tools.shopify.base.entities import ShopifyAdminAuth
from arklex.resources.tools.shopify.utils.utils import authorify_admin
from arklex.resources.tools.shopify.utils.utils_slots import (
    ShopifyGetOrderDetailsSlots,
)
from arklex.resources.tools.tools import register_tool
from arklex.utils.logging.exceptions import ToolExecutionError


class GetOrderDetailsOutput(BaseModel):
    message_flow: str


@register_tool(
    description="Get the status and details of an order.",
    slots=ShopifyGetOrderDetailsSlots.get_all_slots(),
)
def get_order_details(
    order_ids: list[str],
    order_names: list[str],
    user_id: str,
    auth: ShopifyAdminAuth,
    limit: int = 10,
    **kwargs: object,
) -> GetOrderDetailsOutput:
    """
    Retrieve detailed information about orders from the Shopify store.

    Args:
        order_ids (List[str]): List of order IDs to filter by.
        order_names (List[str]): List of order names to filter by.
        user_id (str): The customer ID to filter orders by.
        limit (int, optional): Maximum number of orders to return. Defaults to 10.
        **kwargs (GetOrderDetailsParams): Additional keyword arguments for authentication.

    Returns:
        GetOrderDetailsOutput: A formatted string containing detailed information about each order, including:
            - Order ID and name
            - Creation and cancellation dates
            - Return status
            - Status page URL
            - Total price
            - Fulfillment status
            - Line items with their details
        str: "You have no orders placed." if no orders are found.

    Raises:
        ToolExecutionError: If there's an error retrieving the orders.
    """
    func_name = inspect.currentframe().f_code.co_name
    auth = authorify_admin(auth)
    limit = int(limit) if limit else 10

    try:
        query = f"customer_id:{user_id.split('/')[-1]}"
        if order_ids:
            order_ids = " OR ".join(f"id:{oid.split('/')[-1]}" for oid in order_ids)
            query += f" AND ({order_ids})"
        if order_names:
            order_names = " OR ".join(f"name:{name}" for name in order_names)
            query += f" AND ({order_names})"
        with shopify.Session.temp(**auth):
            response = shopify.GraphQL().execute(f"""
            {{
                orders (first: {limit}, query:"{query}") {{
                    nodes {{
                        id
                        name
                        createdAt
                        cancelledAt
                        returnStatus
                        statusPageUrl
                        totalPriceSet {{
                            presentmentMoney {{
                                amount
                            }}
                        }}
                        fulfillments {{
                            displayStatus
                            trackingInfo {{
                                number
                                url
                            }}
                        }}
                        lineItems(first: 10) {{
                            edges {{
                                node {{
                                    id
                                    title
                                    quantity
                                    variant {{
                                        id
                                        product {{
                                            id
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            """)
            result = json.loads(response)["data"]["orders"]["nodes"]
            if len(result) == 0:
                return GetOrderDetailsOutput(
                    message_flow="You have no orders placed.",
                )
            response_text = ""
            for order in result:
                response_text += f"Order ID: {order.get('id', 'None')}\n"
                response_text += f"Order Name: {order.get('name', 'None')}\n"
                response_text += f"Created At: {order.get('createdAt', 'None')}\n"
                response_text += f"Cancelled At: {order.get('cancelledAt', 'None')}\n"
                response_text += f"Return Status: {order.get('returnStatus', 'None')}\n"
                response_text += (
                    f"Status Page URL: {order.get('statusPageUrl', 'None')}\n"
                )
                response_text += f"Total Price: {order.get('totalPriceSet', {}).get('presentmentMoney', {}).get('amount', 'None')}\n"
                response_text += (
                    f"Fulfillment Status: {order.get('fulfillments', 'None')}\n"
                )
                response_text += "Line Items:\n"
                for item in order.get("lineItems", {}).get("edges", []):
                    response_text += (
                        f"    Title: {item.get('node', {}).get('title', 'None')}\n"
                    )
                    response_text += f"    Quantity: {item.get('node', {}).get('quantity', 'None')}\n"
                    response_text += (
                        f"    Variant: {item.get('node', {}).get('variant', {})}\n"
                    )
                response_text += "\n"
        return GetOrderDetailsOutput(
            message_flow=response_text,
        )
    except Exception as e:
        raise ToolExecutionError(
            func_name,
            extra_message=ShopifyExceptionPrompt.ORDERS_NOT_FOUND_PROMPT,
        ) from e
