"""
This module provides functionality to retrieve detailed information about multiple products
from the Shopify store, including their inventory, descriptions, and variants.

Module Name: get_products

This file contains the code for retrieving product information from Shopify.
"""

import inspect
import json

import shopify
from pydantic import BaseModel

from arklex.resources.tools.shopify._exception_prompt import ShopifyExceptionPrompt
from arklex.resources.tools.shopify.base.entities import ShopifyAdminAuth
from arklex.resources.tools.shopify.legacy.utils_nav import cursorify
from arklex.resources.tools.shopify.utils.utils import authorify_admin
from arklex.resources.tools.shopify.utils.utils_slots import ShopifyGetProductsSlots
from arklex.resources.tools.tools import register_tool
from arklex.utils.logging.exceptions import ToolExecutionError
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)


class GetProductsOutput(BaseModel):
    message_flow: str


@register_tool(
    description="Get the inventory information and description details of multiple products.",
    slots=ShopifyGetProductsSlots.get_all_slots(),
)
def get_products(
    product_ids: list[str], auth: ShopifyAdminAuth, **kwargs: object
) -> GetProductsOutput:
    """
    Retrieve detailed information about multiple products from the Shopify store.

    Args:
        product_ids (List[str]): List of product IDs to retrieve information for.
        auth (ShopifyAdminAuth): Authentication credentials for the Shopify store.
        **kwargs: Additional keyword arguments for llm configuration.

    Returns:
        GetProductsOutput: A formatted string containing detailed information about each product, including:
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
    auth = authorify_admin(auth)
    nav = cursorify(kwargs)

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
            return GetProductsOutput(
                message_flow=response_text,
            )
    except Exception as e:
        raise ToolExecutionError(
            func_name,
            extra_message=ShopifyExceptionPrompt.PRODUCTS_NOT_FOUND_PROMPT,
        ) from e
