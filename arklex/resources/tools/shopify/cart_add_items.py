"""
This module provides functionality to add items to a shopping cart in the Shopify store.
It supports adding multiple product variants to a cart with specified quantities.

Module Name: cart_add_items

This file contains the code for adding items to a shopping cart.
"""

import inspect
import json

import requests
from pydantic import BaseModel

from arklex.resources.tools.shopify._exception_prompt import ShopifyExceptionPrompt
from arklex.resources.tools.shopify.base.entities import ShopifyStorefrontAuth
from arklex.resources.tools.shopify.utils.utils import authorify_storefront
from arklex.resources.tools.shopify.utils.utils_slots import (
    ShopifyCartAddItemsSlots,
)
from arklex.resources.tools.tools import register_tool
from arklex.utils.logging.exceptions import ToolExecutionError
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class CartAddItemsOutput(BaseModel):
    message_flow: str


@register_tool(
    description="Add items to user's shopping cart.",
    slots=ShopifyCartAddItemsSlots.get_all_slots(),
)
def cart_add_items(
    cart_id: str,
    product_variant_ids: list[str],
    auth: ShopifyStorefrontAuth,
    **kwargs: object,
) -> CartAddItemsOutput:
    """
    Add items to a shopping cart.

    Args:
        cart_id (str): The ID of the shopping cart.
        product_variant_ids (list[str]): List of product variant IDs to add to the cart.
        auth (ShopifyStorefrontAuth): Authentication credentials for the Shopify store.
        **kwargs: Additional keyword arguments for llm configuration.

    Returns:
        CartAddItemsOutput: A success message with the cart details if successful.

    Raises:
        ToolExecutionError: If:
            - The items cannot be added to the cart
            - There are errors in the cart data
            - There's an error in the request process
    """
    func_name = inspect.currentframe().f_code.co_name
    auth = authorify_storefront(auth)

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

                return CartAddItemsOutput(
                    message_flow="Items are successfully added to the shopping cart. "
                    + json.dumps(cart_lines_add["cart"]),
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
