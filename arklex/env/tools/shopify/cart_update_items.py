"""
This module provides functionality to update items in a shopping cart in the Shopify store.
It supports modifying item quantities and merchandise IDs for multiple items at once.

Note: This module is currently inactive and reserved for future use.
It may contain experimental or planned features (dependence on shopping cart id).

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: cart_update_items

This file contains the code for updating items in a shopping cart.
"""

from typing import List, Any, Dict, Union, Optional, Tuple

from arklex.env.tools.shopify.utils_slots import ShopifySlots
from arklex.env.tools.shopify.utils_cart import *
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import make_query
from arklex.env.tools.tools import register_tool

description = (
    "Update items in a shopping cart by modifying their quantities or merchandise IDs."
)
slots = [
    ShopifySlots.CART_ID,
    ShopifySlots.UPDATE_LINE_ITEMS,
]
outputs = []

CART_UPDATE_ITEM_ERROR = "error: products could not be updated to cart"
errors = [CART_UPDATE_ITEM_ERROR]


@register_tool(description, slots, outputs, lambda x: x not in errors)
def cart_update_items(
    cart_id: str, items: List[Tuple[str, Optional[str], Optional[int]]]
) -> Union[None, str]:
    """
    Update items in a shopping cart by modifying their quantities or merchandise IDs.

    Args:
        cart_id (str): The ID of the shopping cart.
        items (List[Tuple[str, Optional[str], Optional[int]]]): List of item updates, where each item is a tuple containing:
            - [0]: Line item ID (str)
            - [1]: Optional merchandise ID (str)
            - [2]: Optional quantity (int)

    Returns:
        Union[None, str]: Either:
            - None: If the operation is successful
            - str: Error message if the operation fails

    Raises:
        None: Errors are caught and returned as strings.
    """
    try:
        query = """
        mutation cartLinesUpdate($cartId: ID!, $lines: [CartLineUpdateInput!]!) {
            cartLinesUpdate(cartId: $cartId, lines: $lines) {
                cart {
                    checkoutUrl
                }
            }
        }
        """

        lines: List[Dict[str, Any]] = []
        for i in items:
            lineItem: Dict[str, Any] = {"id": i[0]}
            if i[1]:
                lineItem["merchandiseId"] = i[1]
            if i[2]:
                lineItem["quantity"] = i[2]
            lines.append(lineItem)

        variable: Dict[str, Any] = {"cartId": cart_id, "lines": lines}
        make_query(cart_url, query, variable, cart_headers)
        return None
    except:
        return CART_UPDATE_ITEM_ERROR
