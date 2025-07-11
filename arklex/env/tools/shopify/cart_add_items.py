"""
This module provides functionality to add items to a shopping cart in the Shopify store.
It supports adding multiple product variants to a cart with specified quantities.

Module Name: cart_add_items

This file contains the code for adding items to a shopping cart.
"""

import inspect
import json
from typing import TypedDict

import requests

from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt
from arklex.env.tools.shopify.utils import authorify_storefront
from arklex.env.tools.shopify.utils_slots import (
    ShopifyCartAddItemsSlots,
    ShopifyOutputs,
)
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class CartAddItemsParams(TypedDict, total=False):
    """Parameters for the cart add items tool."""

    shop_url: str
    api_version: str
    storefront_token: str


description = "Add items to user's shopping cart."
slots = ShopifyCartAddItemsSlots.get_all_slots()
outputs = [ShopifyOutputs.CART_ADD_ITEMS_DETAILS]


@register_tool(description, slots, outputs)
def cart_add_items(
    cart_id: str, product_variant_ids: list[str], **kwargs: CartAddItemsParams
) -> str:
    """
    Add items to a shopping cart.

    Args:
        cart_id (str): The ID of the shopping cart.
        product_variant_ids (List[str]): List of product variant IDs to add to the cart.
        **kwargs (CartAddItemsParams): Additional keyword arguments for authentication.

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

    variable: dict[str, list[dict[str, str | int]]] = {
        "cartId": cart_id,
        "lines": [
            {"merchandiseId": pv_id, "quantity": 1} for pv_id in product_variant_ids
        ],
    }
    headers: dict[str, str] = {
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
    try:
        response = requests.post(
            auth["storefront_url"],
            json={"query": query, "variables": variable},
            headers=headers,
        )
        if response.status_code == 200:
            cart_data = response.json()
            if "errors" in cart_data:
                raise ToolExecutionError(
                    func_name,
                    extra_message=ShopifyExceptionPrompt.CART_ADD_ITEMS_ERROR_PROMPT,
                )
            else:
                # Check for required keys in the response
                if "data" not in cart_data:
                    raise ToolExecutionError(
                        func_name,
                        extra_message=ShopifyExceptionPrompt.CART_ADD_ITEMS_ERROR_PROMPT,
                    )

                if "cartLinesAdd" not in cart_data["data"]:
                    raise ToolExecutionError(
                        func_name,
                        extra_message=ShopifyExceptionPrompt.CART_ADD_ITEMS_ERROR_PROMPT,
                    )

                cart_lines_add = cart_data["data"]["cartLinesAdd"]
                if "cart" not in cart_lines_add or cart_lines_add["cart"] is None:
                    raise ToolExecutionError(
                        func_name,
                        extra_message=ShopifyExceptionPrompt.CART_ADD_ITEMS_ERROR_PROMPT,
                    )

                return (
                    "Items are successfully added to the shopping cart. "
                    + json.dumps(cart_lines_add["cart"])
                )
        else:
            raise ToolExecutionError(
                func_name,
                extra_message=ShopifyExceptionPrompt.CART_ADD_ITEMS_ERROR_PROMPT,
            )
    except (requests.RequestException, Exception) as e:
        log_context.error(f"Cart add items failed: {e}")
        raise ToolExecutionError(
            func_name,
            extra_message=ShopifyExceptionPrompt.CART_ADD_ITEMS_ERROR_PROMPT,
        ) from e
