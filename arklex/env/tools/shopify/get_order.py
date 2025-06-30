"""
This module provides functionality to retrieve detailed information about a specific order
from the Shopify store, including its status, total price, and line items.

Note: This module is currently inactive and reserved for future use.
It may contain experimental or planned features (dependence on refresh token).

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: get_order

This file contains the code for retrieving order information from Shopify.
"""

from typing import TypedDict

from arklex.env.tools.shopify.auth_utils import AUTH_ERROR, get_access_token
from arklex.env.tools.shopify.utils import make_query
from arklex.env.tools.shopify.utils_nav import (
    PAGEINFO_OUTPUTS,
    PAGEINFO_SLOTS,
    cursorify,
)

# Customer API
from arklex.env.tools.shopify.utils_slots import ShopifyOutputs, ShopifySlots
from arklex.env.tools.tools import register_tool

# Placeholder values for undefined variables (module is inactive)
customer_url = "https://placeholder.myshopify.com/api/2024-04/graphql.json"
customer_headers = {"Content-Type": "application/json"}

description = "Get the status and details of an order."
slots = [ShopifySlots.REFRESH_TOKEN, ShopifySlots.ORDER_ID, *PAGEINFO_SLOTS]
outputs = [ShopifyOutputs.ORDERS_DETAILS, *PAGEINFO_OUTPUTS]

ORDERS_NOT_FOUND = "error: order not found"
errors = [ORDERS_NOT_FOUND]


class GetOrderParams(TypedDict, total=False):
    """Parameters for the get order tool."""

    limit: str
    navigate: str
    pageInfo: str


@register_tool(description, slots, outputs, lambda x: x not in errors)
def get_order(
    refresh_token: str, order_id: str, **kwargs: GetOrderParams
) -> tuple[dict[str, str], dict[str, str]] | str:
    """
    Retrieve the status and details of a specific order.

    Args:
        refresh_token (str): The refresh token for authentication.
        order_id (str): The ID of the order to retrieve.
        **kwargs (GetOrderParams): Additional keyword arguments for pagination.

    Returns:
        Union[Tuple[Dict[str, str], Dict[str, str]], str]: Either:
            - A tuple containing:
                - Dict[str, str]: Order details including ID, name, total price, and line items
                - Dict[str, str]: Page information for pagination
            - str: Error message if the operation fails

    Raises:
        None: Errors are caught and returned as strings.
    """
    nav = cursorify(kwargs)
    if not nav[1]:
        return nav[0]

    try:
        body = f'''
            query {{ 
                order (id: "{order_id}") {{ 
                    id
                    name
                    totalPrice {{
                        amount
                    }}
                    lineItems({nav[0]}) {{
                        nodes {{
                            id
                            name
                            quantity
                        }}
                        pageInfo {{
                            endCursor
                            hasNextPage
                            hasPreviousPage
                            startCursor
                        }}
                    }}
                }} 
            }}
        '''
        try:
            auth: dict[str, str] = {"Authorization": get_access_token(refresh_token)}
        except Exception:
            return AUTH_ERROR

        try:
            response: dict[str, str] = make_query(
                customer_url, body, {}, customer_headers | auth
            )["data"]["order"]
        except Exception as e:
            return f"error: {e}"

        pageInfo: dict[str, str] = response["lineItems"]["pageInfo"]
        return response, pageInfo
    except Exception:
        return ORDERS_NOT_FOUND
