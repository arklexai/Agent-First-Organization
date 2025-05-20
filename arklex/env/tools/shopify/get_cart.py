from typing import Any, Dict, List, Tuple
import inspect
import logging
import requests
from arklex.env.tools.shopify.utils_slots import ShopifyGetCartSlots, ShopifyOutputs
from arklex.env.tools.shopify.utils_cart import *
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.tools import register_tool
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt

logger = logging.getLogger(__name__)

description: str = "Get cart information"
slots: List[Dict[str, Any]] = ShopifyGetCartSlots.get_all_slots()
outputs: List[Dict[str, Any]] = [ShopifyOutputs.GET_CART_DETAILS, *PAGEINFO_OUTPUTS]


@register_tool(description, slots, outputs)
def get_cart(cart_id: str, **kwargs: Any) -> str:
    func_name: str = inspect.currentframe().f_code.co_name
    nav: Tuple[str, bool] = cursorify(kwargs)
    if not nav[1]:
        return nav[0]
    auth: Dict[str, str] = authorify_storefront(kwargs)

    variable: Dict[str, str] = {
        "id": cart_id,
    }
    headers: Dict[str, str] = {
        "X-Shopify-Storefront-Access-Token": auth["storefront_token"]
    }
    query: str = f"""
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
    response: requests.Response = requests.post(
        auth["storefront_url"],
        json={"query": query, "variables": variable},
        headers=headers,
    )
    if response.status_code == 200:
        response_data: Dict[str, Any] = response.json()
        cart_data: Dict[str, Any] = response_data["data"]["cart"]
        if not cart_data:
            raise ToolExecutionError(
                func_name, ShopifyExceptionPrompt.CART_NOT_FOUND_ERROR_PROMPT
            )
        response_text: str = ""
        response_text += f"Checkout URL: {cart_data['checkoutUrl']}\n"
        lines: Dict[str, List[Dict[str, Any]]] = cart_data["lines"]
        for line in lines["nodes"]:
            product: Dict[str, Any] = line.get("merchandise", {}).get("product", {})
            if product:
                response_text += f"Product ID: {product['id']}\n"
                response_text += f"Product Title: {product['title']}\n"
        return response_text
    else:
        raise ToolExecutionError(
            func_name, ShopifyExceptionPrompt.CART_NOT_FOUND_ERROR_PROMPT
        )
