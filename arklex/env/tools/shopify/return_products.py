"""
This module provides functionality to process product returns in the Shopify store.
It handles the retrieval of returnable fulfillment items and submission of return requests.

Module Name: return_products

This file contains the code for processing product returns in Shopify.
"""

import inspect
import json
from typing import TypedDict

import shopify

from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt

# general GraphQL navigation utilities
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import (
    ShopifyOutputs,
    ShopifyReturnProductsSlots,
)
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class ReturnProductsParams(TypedDict, total=False):
    """Parameters for the return products tool."""

    shop_url: str
    api_version: str
    admin_token: str


description = "Return order by order id. If no fulfillments are found, the function will return an error message."
slots = ShopifyReturnProductsSlots.get_all_slots()
# change output
outputs = [
    ShopifyOutputs.RETURN_REQUEST_DETAILS,
]


@register_tool(description, slots, outputs)
def return_products(return_order_id: str, **kwargs: ReturnProductsParams) -> str:
    """
    Process a return request for a Shopify order.

    Args:
        return_order_id (str): The ID of the order to be returned.
        **kwargs (ReturnProductsParams): Additional keyword arguments for authentication.

    Returns:
        str: A success message with return request details if successful.

    Raises:
        ToolExecutionError: If:
            - No fulfillments are found for the order
            - There's an error parsing the response
            - The return request submission fails
    """
    func_name = inspect.currentframe().f_code.co_name
    auth = authorify_admin(kwargs)

    try:
        with shopify.Session.temp(**auth):
            response = shopify.GraphQL().execute(f"""
            {{
                returnableFulfillments (orderId: "{return_order_id}", first: 10) {{
                    edges {{
                        node {{
                            id
                            fulfillment {{
                                id
                            }}
                            returnableFulfillmentLineItems(first: 10) {{
                                edges {{
                                    node {{
                                        fulfillmentLineItem {{
                                            id
                                        }}
                                        quantity
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            """)
            try:
                response = json.loads(response)
                # Extract all fulfillment line item IDs
                fulfillment_items = []
                for fulfillment in response["data"]["returnableFulfillments"]["edges"]:
                    for line_item in fulfillment["node"][
                        "returnableFulfillmentLineItems"
                    ]["edges"]:
                        line_item_id = line_item["node"]["fulfillmentLineItem"]["id"]
                        line_item_quantity = line_item["node"]["quantity"]
                        fulfillment_items.append(
                            {
                                "fulfillmentLineItemId": line_item_id,
                                "quantity": line_item_quantity,
                            }
                        )
                if not fulfillment_items:
                    raise ToolExecutionError(
                        func_name,
                        extra_message=ShopifyExceptionPrompt.NO_FULFILLMENT_FOUND_ERROR_PROMPT,
                    )
                log_context.info(f"Found {len(fulfillment_items)} fulfillment items.")
            except Exception as e:
                log_context.error(f"Error parsing response: {e}")
                raise ToolExecutionError(
                    func_name,
                    extra_message=ShopifyExceptionPrompt.PRODUCT_RETURN_ERROR_PROMPT,
                ) from e

            # Submit the return request
            fulfillment_string = ""
            for item in fulfillment_items:
                fulfillment_string += f'{{fulfillmentLineItemId: "{item["fulfillmentLineItemId"]}", quantity: {item["quantity"]}, returnReason: UNKNOWN}},'
            fulfillment_string = "[" + fulfillment_string + "]"
            response = shopify.GraphQL().execute(f"""
            mutation ReturnRequestMutation {{
            returnRequest(
                input: {{
                orderId: "{return_order_id}",
                returnLineItems: {fulfillment_string}
                }}
            ) {{
                return {{
                    id
                    status
                }}
                userErrors {{
                    field
                    message
                }}
            }}
            }}
            """)
            try:
                response = json.loads(response)["data"]
                if response.get("returnRequest"):
                    return (
                        "The product return request is successfully submitted. "
                        + json.dumps(response)
                    )
                else:
                    raise ToolExecutionError(
                        func_name,
                        extra_message=ShopifyExceptionPrompt.PRODUCT_RETURN_ERROR_PROMPT,
                    )
            except Exception as e:
                log_context.error(f"Error parsing response: {e}")
                raise ToolExecutionError(
                    func_name,
                    extra_message=ShopifyExceptionPrompt.PRODUCT_RETURN_ERROR_PROMPT,
                ) from e

    except Exception as e:
        raise ToolExecutionError(
            func_name,
            extra_message=ShopifyExceptionPrompt.PRODUCT_RETURN_ERROR_PROMPT,
        ) from e
