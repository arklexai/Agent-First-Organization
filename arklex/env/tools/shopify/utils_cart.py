"""
This module is currently inactive.

It is reserved for future use and may contain experimental or planned features.

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: utils_cart

This file contains the code for operations related to shopping cart.
"""

import os

from dotenv import load_dotenv

from arklex.env.tools.shopify.utils import make_query

load_dotenv()

cart_url = "https://xu1e3z-yi.myshopify.com/api/2024-04/graphql.json"
cart_headers = {
    "X-Shopify-Storefront-Access-Token": os.environ.get(
        "SHOPIFY_STOREFRONT_ACCESS_TOKEN"
    )
}


def create_cart() -> str:
    query = """
            mutation cartCreate($input: CartInput) {
                cartCreate(input: $input) {
                    cart {
                        # attributes {
                        #   key
                        #   value
                        # }
                        id
                        lines (first: 10) {
                            nodes {
                                id
                                quantity
                            }
                        }
                        checkoutUrl
                    }
                }
            }
            """
    variable = {
        "input": {
            # "attributes": [
            #   {
            #     "key": "test",
            #     "value": "test"
            #   }
            # ]
        }
    }

    cart_dict = make_query(cart_url, query, variable, cart_headers)
    return cart_dict["data"]["cartCreate"]["cart"]["id"]
