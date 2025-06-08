"""
This module provides functionality to retrieve cart information from the Shopify store.
It supports retrieving cart details including checkout URL and line items.

Module Name: get_cart

This file contains the code for retrieving cart information from Shopify.
"""

from typing import Any, Dict
import requests
import inspect
import logging

from arklex.env.tools.shopify.utils_slots import ShopifyGetCartSlots, ShopifyOutputs
from arklex.env.tools.shopify.utils_cart import *
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.tools import register_tool
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt

logger = logging.getLogger(__name__)

description = "Get cart information"
slots = ShopifyGetCartSlots.get_all_slots()
outputs = [ShopifyOutputs.GET_CART_DETAILS, *PAGEINFO_OUTPUTS]


@register_tool(description, slots, outputs)
def get_cart(cart_id: str, **kwargs: Any) -> str:
    """
    Retrieve detailed information about a shopping cart.

    Args:
        cart_id (str): The ID of the cart to retrieve.
        **kwargs (Any): Additional keyword arguments for pagination and authentication.

    Returns:
        str: A formatted string containing cart information, including:
            - Checkout URL
            - Product IDs and titles for each line item

    Raises:
        ToolExecutionError: If:
            - The cart is not found
            - There's an error retrieving the cart information
    """
    func_name = inspect.currentframe().f_code.co_name
    nav = cursorify(kwargs)
    if not nav[1]:
        return nav[0]
    auth = authorify_storefront(kwargs)

    variable: Dict[str, str] = {
        "id": cart_id,
    }
    headers: Dict[str, str] = {
        "X-Shopify-Storefront-Access-Token": auth["storefront_token"]
    }
    query = f"""
        query ($id: ID!) {{ 
            cart(id: $id) {{
                id
                checkoutUrl
                lines ({nav[0]}) {{
                    nodes {{
                        id
                        quantity
                        merchandise {{
                            ... on ProductVariant {{
                                id
                                title
                                product {{
                                    title
                                    id
                                }}
                            }}
                        }}
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
    """
    response = requests.post(
        auth["storefront_url"],
        json={"query": query, "variables": variable},
        headers=headers,
    )
    if response.status_code == 200:
        response = response.json()
        cart_data = response["data"]["cart"]
        if not cart_data:
            raise ToolExecutionError(
                func_name, ShopifyExceptionPrompt.CART_NOT_FOUND_ERROR_PROMPT
            )
        response_text = ""
        response_text += f"Checkout URL: {cart_data['checkoutUrl']}\n"
        lines = cart_data["lines"]
        for line in lines["nodes"]:
            product = line.get("merchandise", {}).get("product", {})
            if product:
                response_text += f"Product ID: {product['id']}\n"
                response_text += f"Product Title: {product['title']}\n"
        return response_text
    else:
        raise ToolExecutionError(
            func_name, ShopifyExceptionPrompt.CART_NOT_FOUND_ERROR_PROMPT
        )
