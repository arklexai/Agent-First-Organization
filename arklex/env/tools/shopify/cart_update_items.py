"""
This module is currently inactive.

It is reserved for future use and may contain experimental or planned features (dependence on shopping cart id).

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: cart_update_items

This file contains the code for updating items in a shopping cart.
"""

from typing import Any, Dict, List, Optional, Tuple
from arklex.env.tools.shopify.utils_slots import ShopifySlots
from arklex.env.tools.shopify.utils_cart import *
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import make_query

from arklex.env.tools.tools import register_tool

description: str = "Removes a items from cart based on line ids"
slots: List[Dict[str, Any]] = [
    ShopifySlots.CART_ID,
    ShopifySlots.UPDATE_LINE_ITEMS,
]
outputs: List[Dict[str, Any]] = []

CART_UPDATE_ITEM_ERROR: str = "error: products could not be updated to cart"
errors: List[str] = [CART_UPDATE_ITEM_ERROR]


@register_tool(description, slots, outputs, lambda x: x not in errors)
def cart_update_items(
    cart_id: str, items: List[Tuple[str, Optional[str], Optional[int]]]
) -> Optional[str]:
    try:
        query: str = """
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
