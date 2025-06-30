"""
This module provides functionality to remove items from a shopping cart in the Shopify store.
It supports removing multiple items at once using their line IDs.

Note: This module is currently inactive and reserved for future use.
It may contain experimental or planned features (dependence on shopping cart id).

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: cart_remove_items

This file contains the code for removing items from a shopping cart.
"""

from typing import Any

from arklex.env.tools.shopify.utils import make_query
from arklex.env.tools.shopify.utils_cart import cart_headers, cart_url
from arklex.env.tools.shopify.utils_slots import ShopifySlots
from arklex.env.tools.tools import register_tool

description = "Remove items from a shopping cart by their line IDs."
slots = [
    ShopifySlots.CART_ID,
    ShopifySlots.to_list(ShopifySlots.LINE_IDS),
]
outputs = []
CART_REMOVE_ITEM_ERROR = "error: products could not be removed from cart"
errors = [CART_REMOVE_ITEM_ERROR]


@register_tool(description, slots, outputs, lambda x: x not in errors)
def cart_remove_items(cart_id: str, line_ids: list[str]) -> None | str:
    """
    Remove items from a shopping cart using their line IDs.

    Args:
        cart_id (str): The ID of the shopping cart.
        line_ids (List[str]): List of line item IDs to remove from the cart.

    Returns:
        Union[None, str]: Either:
            - None: If the operation is successful
            - str: Error message if the operation fails

    Raises:
        None: Errors are caught and returned as strings.
    """
    try:
        query = """
        mutation cartLinesRemove($cartId: ID!, $lineIds: [ID!]!) {
            cartLinesRemove(cartId: $cartId, lineIds: $lineIds) {
                cart {
                    checkoutUrl
                }
            }
        }
        """

        variable: dict[str, Any] = {"cartId": cart_id, "lineIds": line_ids}
        make_query(cart_url, query, variable, cart_headers)
        return None
    except Exception:
        return CART_REMOVE_ITEM_ERROR
