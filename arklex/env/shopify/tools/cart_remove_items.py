"""
This module is currently inactive.

It is reserved for future use and may contain experimental or planned features (dependence on shopping cart id).

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: cart_remove_itmes

This file contains the code for removing items in a shopping cart.
"""
from arklex.env.shopify.tools.utils_slots import ShopifySlots, ShopifyOutputs
from arklex.env.shopify.tools.utils_cart import *
from arklex.env.shopify.tools.utils_nav import *

from arklex.env.tools.tools import register_tool

description = "Get the inventory information and description details of a product."
slots = [
    ShopifySlots.CART_ID,
    ShopifySlots.to_list(ShopifySlots.LINE_IDS),
]
outputs = []
CART_REMOVE_ITEM_ERROR = "error: products could not be removed from cart"
errors = [CART_REMOVE_ITEM_ERROR]

@register_tool(description, slots, outputs, lambda x: x not in errors)
def cart_remove_items(cart_id, line_ids):
    try:
        query = '''
        mutation cartLinesRemove($cartId: ID!, $lineIds: [ID!]!) {
            cartLinesRemove(cartId: $cartId, lineIds: $lineIds) {
                cart {
                    checkoutUrl
                }
            }
        }
        '''
        
        variable = {
            "cartId": cart_id,
            "lineIds": line_ids
        }
        make_query(cart_url, query, variable, cart_headers)
        return
    except:
        return CART_REMOVE_ITEM_ERROR