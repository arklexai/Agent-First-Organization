from arklex.env.shopify.tools.utils_slots import ShopifyCartAddItemsSlots, ShopifyOutputs
from arklex.env.shopify.tools.utils_cart import *
from arklex.env.shopify.tools.utils_nav import *
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.tools import register_tool
from arklex.env.shopify.tools._exception_prompt import ShopifyExceptionPrompt

import inspect

description = "Add items to user's shopping cart."
slots = ShopifyCartAddItemsSlots.get_all_slots()
outputs = [
    ShopifyOutputs.CART_ADD_ITEMS_DETAILS
]


@register_tool(description, slots, outputs)
def cart_add_items(cart_id: str, product_variant_ids: list[str], **kwargs):
    func_name = inspect.currentframe().f_code.co_name
    auth = authorify_storefront(kwargs)
    
    variable = {
        "cartId": cart_id,
        "lines": [
            {"merchandiseId": pv_id,
                "quantity": 1
            } for pv_id in product_variant_ids
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
            raise ToolExecutionError(func_name, ShopifyExceptionPrompt.CART_ADD_ITEMS_ERROR_PROMPT)
        else:
            return "Items are successfully added to the shopping cart. " + json.dumps(cart_data["data"]["cartLinesAdd"]["cart"])
    else:
        raise ToolExecutionError(func_name, ShopifyExceptionPrompt.CART_ADD_ITEMS_ERROR_PROMPT)
