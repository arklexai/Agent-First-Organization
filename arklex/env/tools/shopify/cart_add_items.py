from arklex.env.tools.shopify.utils_slots import ShopifyCartAddItemsSlots, ShopifyOutputs
from arklex.env.tools.shopify.utils_cart import *
from arklex.env.tools.shopify.utils_nav import *
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.tools import register_tool
from arklex.env.tools.shopify._exception_prompt import ExceptionPrompt
import inspect

description = "Add items to user's shopping cart."
slots = ShopifyCartAddItemsSlots.get_all_slots()
outputs = [
    ShopifyOutputs.CART_ADD_ITEMS_DETAILS
]


@register_tool(description, slots, outputs)
def cart_add_items(cart_id: str, add_line_items: list, **kwargs):
    func_name = inspect.currentframe().f_code.co_name
    auth = authorify_storefront(kwargs)
    
    variable = {
        "cartId": cart_id,
        "lines": [
            {"merchandiseId": item,
                "quantity": 1
            } for item in add_line_items
        ]
    }
    headers = {
        "X-Shopify-Storefront-Access-Token": auth["storefront_token"]
    }
    query = '''
    mutation cartLinesAdd($cartId: ID!, $lines: [CartLineInput!]!) {
        cartLinesAdd(cartId: $cartId, lines: $lines) {
            cart {
                checkoutUrl
            }
        }
    }
    '''
    response = requests.post(auth["storefront_url"], json={"query": query, "variables": variable}, headers=headers)
    if response.status_code == 200:
        cart_data = response.json()
        if "errors" in cart_data:
            raise ToolExecutionError(func_name, ExceptionPrompt.CART_ADD_ITEMS_ERROR_PROMPT)
        else:
            return "Items are successfully added to the shopping cart. " + json.dumps(cart_data["data"]["cartLinesAdd"]["cart"])
    else:
        raise ToolExecutionError(func_name, ExceptionPrompt.CART_ADD_ITEMS_ERROR_PROMPT)
