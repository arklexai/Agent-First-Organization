"""
This module provides functionality to retrieve detailed information about multiple products
from the Shopify store, including their inventory, descriptions, and variants.

Module Name: get_products

This file contains the code for retrieving product information from Shopify.
"""

import inspect
import json
from typing import TypedDict

import shopify

from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_nav import PAGEINFO_OUTPUTS, cursorify

# ADMIN
from arklex.env.tools.shopify.utils_slots import ShopifyGetProductsSlots, ShopifyOutputs
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class GetProductsParams(TypedDict, total=False):
    """Parameters for the get products tool."""

    shop_url: str
    api_version: str
    admin_token: str
    limit: str
    navigate: str
    pageInfo: str


description = (
    "Get the inventory information and description details of multiple products."
)
slots = ShopifyGetProductsSlots.get_all_slots()
outputs = [ShopifyOutputs.PRODUCTS_DETAILS, *PAGEINFO_OUTPUTS]


@register_tool(description, slots, outputs)
def get_products(product_ids: list[str], **kwargs: GetProductsParams) -> str:
    """
    Retrieve detailed information about multiple products from the Shopify store.

    Args:
        product_ids (List[str]): List of product IDs to retrieve information for.
        **kwargs (GetProductsParams): Additional keyword arguments for pagination and authentication.

    Returns:
        str: A formatted string containing detailed information about each product, including:
            - Product ID
            - Title
            - Description
            - Total Inventory
            - Options
            - Variants (with name, ID, price, and inventory quantity)

    Raises:
        ToolExecutionError: If no products are found or if there's an error retrieving the products.
    """
    func_name = inspect.currentframe().f_code.co_name
    nav = cursorify(kwargs)
    if not nav[1]:
        return nav[0]
    auth = authorify_admin(kwargs)

    try:
        ids = " OR ".join(f"id:{pid.split('/')[-1]}" for pid in product_ids)
        with shopify.Session.temp(**auth):
            response = shopify.GraphQL().execute(f"""
                {{
                    products ({nav[0]}, query:"{ids}") {{
                        nodes {{
                            id
                            title
                            description
                            totalInventory
                            onlineStoreUrl
                            options {{
                                name
                                values
                            }}
                            category {{
                                name
                            }}
                            variants (first: 3) {{
                                nodes {{
                                    displayName
                                    id
                                    price
                                    inventoryQuantity
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
            """)
            result = json.loads(response)["data"]["products"]
            response = result["nodes"]
            if len(response) == 0:
                raise ToolExecutionError(
                    func_name,
                    extra_message=ShopifyExceptionPrompt.PRODUCTS_NOT_FOUND_PROMPT,
                )
            response_text = ""
            for product in response:
                response_text += f"Product ID: {product.get('id', 'None')}\n"
                response_text += f"Title: {product.get('title', 'None')}\n"
                response_text += f"Description: {product.get('description', 'None')}\n"
                response_text += (
                    f"Total Inventory: {product.get('totalInventory', 'None')}\n"
                )
                response_text += f"Options: {product.get('options', 'None')}\n"
                response_text += "The following are several variants of the product:\n"
                for variant in product.get("variants", {}).get("nodes", []):
                    response_text += f"Variant name: {variant.get('displayName', 'None')}, Variant ID: {variant.get('id', 'None')}, Price: {variant.get('price', 'None')}, Inventory Quantity: {variant.get('inventoryQuantity', 'None')}\n"
                response_text += "\n"
            return response_text
    except Exception as e:
        raise ToolExecutionError(
            func_name,
            extra_message=ShopifyExceptionPrompt.PRODUCTS_NOT_FOUND_PROMPT,
        ) from e
