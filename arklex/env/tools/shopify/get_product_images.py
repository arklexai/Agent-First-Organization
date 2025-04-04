import json
import logging

import shopify

# general GraphQL navigation utilities
from arklex.env.tools.shopify.utils_nav import *
from arklex.env.tools.shopify.utils import authorify_admin

# ADMIN
from arklex.env.tools.shopify.utils_slots import ShopifyGetProductImagesSlots, ShopifyOutputs
from arklex.env.tools.tools import register_tool

from arklex.utils.model_provider_config import PROVIDER_MAP
from arklex.utils.model_config import MODEL
from langchain_openai import ChatOpenAI
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.shopify._exception_prompt import ExceptionPrompt
import inspect
logger = logging.getLogger(__name__)

description = "Get the product image url of a product."
slots = ShopifyGetProductImagesSlots.get_all_slots()
outputs = [
    ShopifyOutputs.PRODUCTS_DETAILS,
    *PAGEINFO_OUTPUTS
]


@register_tool(description, slots, outputs, isResponse=True)
def get_product_images(product_ids: list, **kwargs) -> str:
    func_name = inspect.currentframe().f_code.co_name
    nav = cursorify(kwargs)
    if not nav[1]:
        return nav[0]
    auth = authorify_admin(kwargs)

    try:
        ids = ' OR '.join(f'id:{pid.split("/")[-1]}' for pid in product_ids)
        with shopify.Session.temp(**auth):
            response = shopify.GraphQL().execute(f"""
                {{
                    products ({nav[0]}, query:"{ids}") {{
                        nodes {{
                            title
                            description
                            onlineStoreUrl
                            images(first: 1) {{
                                edges {{
                                    node {{
                                        src
                                        altText
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
            """)
            result = json.loads(response)['data']['products']
            response = result["nodes"]
            answer = "Here are images of products:\n"
            card_list = []
            for product in response:
                product_dict = {
                    "id": product.get('id'),
                    "title": product.get('title'), 
                    "description": product.get('description', "None")[:180] + "...", 
                    "link_url": product.get('onlineStoreUrl'),
                    "image_url" : product.get('images', {}).get('edges', [{}])[0].get('node', {}).get('src', ""),
                }
                card_list.append(product_dict)
            if card_list:
                try:
                    llm = PROVIDER_MAP.get(MODEL['llm_provider'], ChatOpenAI)(model=MODEL["model_type_or_path"], timeout=30000)
                    message = [
                        {"role": "system", "content": f"You are helping a customer search products based on the query and get results below and those results will be presented using product card format.\n\n{json.dumps(card_list)}\n\nGenerate a response to continue the conversation without explicitly mentioning contents of the search result. Include one or two questions about those products to know the user's preference. Keep the response within 50 words.\nDIRECTLY GIVE THE RESPONSE."},
                    ]
                    answer = llm.invoke(message).content
                except Exception as e:
                    logger.info(f"llm error in search_products: {e}")
                    pass
                return json.dumps({
                    "answer": answer,
                    "card_list": card_list
                })
            else:
                raise ToolExecutionError(func_name, ExceptionPrompt.PRODUCTS_NOT_FOUND_PROMPT)
    except Exception as e:
        raise ToolExecutionError(func_name, ExceptionPrompt.PRODUCTS_NOT_FOUND_PROMPT)
