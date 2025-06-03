"""
This module provides functionality to add items to a shopping cart in the Shopify store.
It supports adding multiple product variants to a cart with specified quantities.

Module Name: cart_add_items

This file contains the code for adding items to a shopping cart.
"""

from typing import Any, Dict, List

import json
import inspect
import requests

from arklex.env.tools.shopify.utils_slots import (
    ShopifyCartAddItemsSlots,
    ShopifyOutputs,
)
from arklex.env.tools.shopify.utils_cart import *
from arklex.env.tools.shopify.utils_nav import *
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.tools import register_tool
from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt

description = "Add items to user's shopping cart."
slots = ShopifyCartAddItemsSlots.get_all_slots()
outputs = [ShopifyOutputs.CART_ADD_ITEMS_DETAILS]


@register_tool(description, slots, outputs)
def cart_add_items(cart_id: str, product_variant_ids: List[str], **kwargs: Any) -> str:
    """
    Add items to a shopping cart.

    Args:
        cart_id (str): The ID of the shopping cart.
        product_variant_ids (List[str]): List of product variant IDs to add to the cart.
        **kwargs (Any): Additional keyword arguments for authentication.

    Returns:
        str: A success message with the cart details if successful.

    Raises:
        ToolExecutionError: If:
            - The items cannot be added to the cart
            - There are errors in the cart data
            - There's an error in the request process
    """
    func_name = inspect.currentframe().f_code.co_name
    auth = authorify_storefront(kwargs)

    variable: Dict[str, Any] = {
        "cartId": cart_id,
        "lines": [
            {"merchandiseId": pv_id, "quantity": 1} for pv_id in product_variant_ids
        ],
    }
    headers: Dict[str, str] = {
        "X-Shopify-Storefront-Access-Token": auth["storefront_token"]
    }
    query = """
    mutation cartLinesAdd($cartId: ID!, $lines: [CartLineInput!]!) {
        cartLinesAdd(cartId: $cartId, lines: $lines) {
            cart {
                checkoutUrl
            }
        }
    }
    """
    response = requests.post(
        auth["storefront_url"],
        json={"query": query, "variables": variable},
        headers=headers,
    )
    if response.status_code == 200:
        cart_data = response.json()
        if "errors" in cart_data:
            raise ToolExecutionError(
                func_name, ShopifyExceptionPrompt.CART_ADD_ITEMS_ERROR_PROMPT
            )
        else:
            return "Items are successfully added to the shopping cart. " + json.dumps(
                cart_data["data"]["cartLinesAdd"]["cart"]
            )
    else:
        raise ToolExecutionError(
            func_name, ShopifyExceptionPrompt.CART_ADD_ITEMS_ERROR_PROMPT
        )
