"""
This module provides functionality to retrieve cart information from the Shopify store.
It supports retrieving cart details including checkout URL and line items.

Module Name: get_cart

This file contains the code for retrieving cart information from Shopify.
"""

import inspect

import requests
from pydantic import BaseModel

from arklex.resources.tools.shopify._exception_prompt import ShopifyExceptionPrompt
from arklex.resources.tools.shopify.base.entities import ShopifyStorefrontAuth
from arklex.resources.tools.shopify.legacy.utils_nav import cursorify
from arklex.resources.tools.shopify.utils.utils import authorify_storefront
from arklex.resources.tools.shopify.utils.utils_slots import ShopifyGetCartSlots
from arklex.resources.tools.tools import register_tool
from arklex.utils.logging.exceptions import ToolExecutionError
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class GetCartOutput(BaseModel):
    message_flow: str


@register_tool(
    description="Get cart information", slots=ShopifyGetCartSlots.get_all_slots()
)
def get_cart(
    cart_id: str, auth: ShopifyStorefrontAuth, **kwargs: object
) -> GetCartOutput:
    """
    Retrieve detailed information about a shopping cart.

    Args:
        cart_id (str): The ID of the cart to retrieve.
        auth (ShopifyStorefrontAuth): Authentication credentials for the Shopify store.
        **kwargs: Additional keyword arguments for llm configuration.

    Returns:
        GetCartOutput: A formatted string containing cart information, including:
            - Checkout URL
            - Product IDs and titles for each line item

    Raises:
        ToolExecutionError: If:
            - The cart is not found
            - There's an error retrieving the cart information
    """
    func_name = inspect.currentframe().f_code.co_name
    auth = authorify_storefront(auth)
    nav = cursorify(kwargs)

    variable: dict[str, str] = {
        "id": cart_id,
    }
    headers: dict[str, str] = {
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
    try:
        response = requests.post(
            auth["storefront_url"],
            json={"query": query, "variables": variable},
            headers=headers,
        )
    except Exception:
        raise ToolExecutionError(
            func_name,
            extra_message=ShopifyExceptionPrompt.CART_NOT_FOUND_ERROR_PROMPT,
        ) from None

    if response.status_code == 200:
        try:
            response_data = response.json()
        except Exception as err:
            raise ToolExecutionError(
                func_name,
                extra_message=ShopifyExceptionPrompt.CART_NOT_FOUND_ERROR_PROMPT,
            ) from err

        # Check if response has the expected structure
        if "data" not in response_data:
            raise ToolExecutionError(
                func_name,
                extra_message=ShopifyExceptionPrompt.CART_NOT_FOUND_ERROR_PROMPT,
            )

        cart_data = response_data["data"].get("cart")
        if not cart_data:
            raise ToolExecutionError(
                func_name,
                extra_message=ShopifyExceptionPrompt.CART_NOT_FOUND_ERROR_PROMPT,
            )

        response_text = ""
        response_text += f"Checkout URL: {cart_data['checkoutUrl']}\n"

        # Safely handle lines data
        lines = cart_data.get("lines")
        if lines and isinstance(lines, dict) and "nodes" in lines:
            for line in lines["nodes"]:
                product = line.get("merchandise", {}).get("product", {})
                if product:
                    response_text += f"Product ID: {product['id']}\n"
                    response_text += f"Product Title: {product['title']}\n"
        elif lines is None or not isinstance(lines, dict) or "nodes" not in lines:
            # Handle malformed lines data
            raise ToolExecutionError(
                func_name,
                extra_message=ShopifyExceptionPrompt.CART_NOT_FOUND_ERROR_PROMPT,
            )

        return GetCartOutput(
            message_flow=response_text,
        )
    else:
        raise ToolExecutionError(
            func_name,
            extra_message=ShopifyExceptionPrompt.CART_NOT_FOUND_ERROR_PROMPT,
        )
