from typing import Any, Dict, List
from arklex.env.tools.shopify.utils_slots import (
    ShopifyCartAddItemsSlots,
    ShopifyOutputs,
)
from arklex.env.tools.shopify.utils_cart import *
from arklex.env.tools.shopify.utils_nav import *
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.tools import register_tool
from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt
import inspect
import json
import requests

description: str = "Add items to user's shopping cart."
slots: List[Dict[str, Any]] = ShopifyCartAddItemsSlots.get_all_slots()
outputs: List[Dict[str, Any]] = [ShopifyOutputs.CART_ADD_ITEMS_DETAILS]


@register_tool(description, slots, outputs)
def cart_add_items(
    cart_id: str, product_variant_ids: List[str], **kwargs: Dict[str, Any]
) -> str:
    func_name: str = inspect.currentframe().f_code.co_name
    auth: Dict[str, str] = authorify_storefront(kwargs)

    variable: Dict[str, Any] = {
        "cartId": cart_id,
        "lines": [
            {"merchandiseId": pv_id, "quantity": 1} for pv_id in product_variant_ids
        ],
    }
    headers: Dict[str, str] = {
        "X-Shopify-Storefront-Access-Token": auth["storefront_token"]
    }
    query: str = """
    mutation cartLinesAdd($cartId: ID!, $lines: [CartLineInput!]!) {
        cartLinesAdd(cartId: $cartId, lines: $lines) {
            cart {
                checkoutUrl
            }
        }
    }
    """
    response: requests.Response = requests.post(
        auth["storefront_url"],
        json={"query": query, "variables": variable},
        headers=headers,
    )
    if response.status_code == 200:
        cart_data: Dict[str, Any] = response.json()
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
