import json
from typing import Any, Dict

import shopify

from arklex.env.tools.tools import register_tool
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import ShopifyGetOrderDetailsSlots, ShopifyOutputs
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.shopify._exception_prompt import ExceptionPrompt
import inspect

description = "Get the status and details of an order."
slots = ShopifyGetOrderDetailsSlots.get_all_slots()
outputs = [
    ShopifyOutputs.ORDERS_DETAILS
]


@register_tool(description, slots, outputs)
def get_order_details(order_ids: list, order_names: list, user_id: str, limit=10, **kwargs) -> str:
    func_name = inspect.currentframe().f_code.co_name
    limit = int(limit) if limit else 10
    auth = authorify_admin(kwargs)
    
    try:
        query = f"customer_id:{user_id.split('/')[-1]}"
        if order_ids:
            order_ids = ' OR '.join(f'id:{oid.split("/")[-1]}' for oid in order_ids)
            query += f" AND ({order_ids})"
        if order_names:
            order_names = ' OR '.join(f'name:{name}' for name in order_names)
            query += f" AND ({order_names})"
        with shopify.Session.temp(**auth):
            response = shopify.GraphQL().execute(f"""
            {{
                orders (first: {limit}, query:"{query}") {{
                    nodes {{
                        id
                        name
                        createdAt
                        cancelledAt
                        returnStatus
                        statusPageUrl
                        totalPriceSet {{
                            presentmentMoney {{
                                amount
                            }}
                        }}
                        fulfillments {{
                            displayStatus
                            trackingInfo {{
                                number
                                url
                            }}
                        }}
                        lineItems(first: 10) {{
                            edges {{
                                node {{
                                    id
                                    title
                                    quantity
                                    variant {{
                                        id
                                        product {{
                                            id
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            """)
            result = json.loads(response)["data"]["orders"]["nodes"]
            if len(result) == 0:
                return "You have no orders placed."
            response_text = ""
            for order in result:
                response_text += f"Order ID: {order.get('id', 'None')}\n"
                response_text += f"Order Name: {order.get('name', 'None')}\n"
                response_text += f"Created At: {order.get('createdAt', 'None')}\n"
                response_text += f"Cancelled At: {order.get('cancelledAt', 'None')}\n"
                response_text += f"Return Status: {order.get('returnStatus', 'None')}\n"
                response_text += f"Status Page URL: {order.get('statusPageUrl', 'None')}\n"
                response_text += f"Total Price: {order.get('totalPriceSet', {}).get('presentmentMoney', {}).get('amount', 'None')}\n"
                response_text += f"Fulfillment Status: {order.get('fulfillments', 'None')}\n"
                response_text += "Line Items:\n"
                for item in order.get('lineItems', {}).get('edges', []):
                    response_text += f"    Title: {item.get('node', {}).get('title', 'None')}\n"
                    response_text += f"    Quantity: {item.get('node', {}).get('quantity', 'None')}\n"
                    response_text += f"    Variant: {item.get('node', {}).get('variant', {})}\n"
                response_text += "\n"
        return response_text
    except Exception as e:
        raise ToolExecutionError(func_name, ExceptionPrompt.ORDERS_NOT_FOUND_PROMPT)
