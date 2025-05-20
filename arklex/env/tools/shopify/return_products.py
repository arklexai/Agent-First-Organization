import json
import shopify
import logging
import inspect
from typing import Any, Dict, List

# general GraphQL navigation utilities
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import (
    ShopifyReturnProductsSlots,
    ShopifyOutputs,
)

from arklex.env.tools.tools import register_tool
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt

logger: logging.Logger = logging.getLogger(__name__)

description: str = "Return order by order id. If no fulfillments are found, the function will return an error message."
slots: List[Dict[str, Any]] = ShopifyReturnProductsSlots.get_all_slots()
# change output
outputs: List[Dict[str, Any]] = [
    ShopifyOutputs.RETURN_REQUEST_DETAILS,
]


@register_tool(description, slots, outputs)
def return_products(return_order_id: str, **kwargs: Any) -> str:
    func_name: str = inspect.currentframe().f_code.co_name
    auth: Dict[str, str] = authorify_admin(kwargs)

    try:
        with shopify.Session.temp(**auth):
            response: str = shopify.GraphQL().execute(f"""
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
                response: Dict[str, Any] = json.loads(response)
                # Extract all fulfillment line item IDs
                fulfillment_items: List[Dict[str, Any]] = []
                for fulfillment in response["data"]["returnableFulfillments"]["edges"]:
                    for line_item in fulfillment["node"][
                        "returnableFulfillmentLineItems"
                    ]["edges"]:
                        line_item_id: str = line_item["node"]["fulfillmentLineItem"][
                            "id"
                        ]
                        line_item_quantity: int = line_item["node"]["quantity"]
                        fulfillment_items.append(
                            {
                                "fulfillmentLineItemId": line_item_id,
                                "quantity": line_item_quantity,
                            }
                        )
                if not fulfillment_items:
                    raise ToolExecutionError(
                        func_name,
                        ShopifyExceptionPrompt.NO_FULFILLMENT_FOUND_ERROR_PROMPT,
                    )
                logger.info(f"Found {len(fulfillment_items)} fulfillment items.")
            except Exception as e:
                logger.error(f"Error parsing response: {e}")
                raise ToolExecutionError(
                    func_name, ShopifyExceptionPrompt.PRODUCT_RETURN_ERROR_PROMPT
                )

            # Submit the return request
            fulfillment_string: str = ""
            for item in fulfillment_items:
                fulfillment_string += f'{{fulfillmentLineItemId: "{item["fulfillmentLineItemId"]}", quantity: {item["quantity"]}, returnReason: UNKNOWN}},'
            fulfillment_string = "[" + fulfillment_string + "]"
            response: str = shopify.GraphQL().execute(f"""
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
                response: Dict[str, Any] = json.loads(response)["data"]
                if response.get("returnRequest"):
                    return (
                        "The product return request is successfully submitted. "
                        + json.dumps(response)
                    )
                else:
                    raise ToolExecutionError(
                        func_name, ShopifyExceptionPrompt.PRODUCT_RETURN_ERROR_PROMPT
                    )
            except Exception as e:
                logger.error(f"Error parsing response: {e}")
                raise ToolExecutionError(
                    func_name, ShopifyExceptionPrompt.PRODUCT_RETURN_ERROR_PROMPT
                )

    except Exception as e:
        raise ToolExecutionError(
            func_name, ShopifyExceptionPrompt.PRODUCT_RETURN_ERROR_PROMPT
        )
