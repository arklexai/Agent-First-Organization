class ShopifySlots:
    def to_list(baseSlot: dict) -> dict:
        slot = baseSlot.copy()
        slot["name"] += "s"
        slot["type"] = f"list[{slot['type']}]"
        slot["description"] = f"List of {slot['name']}. {slot['description']}"
        return slot

    @classmethod
    def get_all_slots(cls) -> list[dict]:
        return [slot for slot in cls.__dict__.values() if isinstance(slot, dict | list)]

    USER_ID = {
        "name": "user_id",
        "type": "str",
        "description": "The user id, such as 'gid://shopify/Customer/13573257450893'.",
        "prompt": "In order to proceed, please login to Shopify.",
        "verified": True,
    }

    PRODUCT_ID = {
        "name": "product_id",
        "type": "str",
        "description": "The product id, such as 'gid://shopify/Product/2938501948327'.",  # If there is only 1 product, return in list with single item. If there are multiple product ids, please return all of them in a list.",
        "prompt": "In order to proceed, please choose a specific product.",
        "verified": True,
    }
    PRODUCT_IDS = to_list(PRODUCT_ID)

    CART_ID = {
        "name": "cart_id",
        "type": "str",
        "description": "The cart id, such as 'gid://shopify/Cart/2938501948327'.",
        "prompt": "In order to proceed, please create a shopping cart.",
        "verified": True,
    }

    # Not in user as of 03.11.2025
    # LINE_ID = {
    #     "name": "line_id",
    #     "type": "str",
    #     "description": "The line id for a line entry in the cart such as 'gid://shopify/CartLine/b3dbff2e-4e9a-4ce0-9f15-5fa8f61882e1?cart=Z2NwLXVzLWVhc3QxOjAxSkpDTjBQSDVLR1JaRkZHMkE3UlZSVjhX'",
    #     "prompt": "",
    #     "verified": True
    # }
    # LINE_IDS = to_list(LINE_ID)

    # UPDATE_LINE_ITEM = {
    #     "name": "line_ids",
    #     "type": "list",
    #     "items": "str",
    #     "description": "list of (line_id, item_id, quantity) tuples of lineItem to add to the cart such as [('gid://shopify/CartLine/db5cb3dd-c830-482e-88cd-99afe8eafa3f?cart=Z2NwLXVzLWVhc3QxOjAxSkpEM0JLNU1KMUI2UFRYRTNFS0NNTllW', None, 69)]",
    #     "prompt": "",
    #     "verified": True
    # }

    # REFRESH_TOKEN = {
    #     "name": "refresh_token",
    #     "type": "str",
    #     "description": "customer's shopify refresh_token retrieved from authenticating",
    #     "prompt": "",
    #     "verified": True
    # }


class ShopifyCancelOrderSlots(ShopifySlots):
    CANCEL_ORDER_ID = {
        "name": "cancel_order_id",
        "type": "str",
        "description": "The order id to cancel, such as gid://shopify/Order/1289503851427.",
        "prompt": "Please provide the order id that you would like to cancel.",
        "required": True,
        "verified": True,
    }


class ShopifyCartAddItemsSlots(ShopifySlots):
    CART_ID = {**ShopifySlots.CART_ID, "required": True}
    PRODUCT_VARIANT_IDS = ShopifySlots.to_list(
        {
            "name": "product_variant_id",
            "type": "str",
            "description": "ProductVariant id to be added to the shopping cart, such as gid://shopify/ProductVariant/41552094527601.",
            "prompt": "Please confirm the items to add to the cart.",
            "required": True,
            "verified": True,
        }
    )


class ShopifyFindUserByEmailSlots(ShopifySlots):
    USER_EMAIL = {
        "name": "user_email",
        "type": "str",
        "description": "The email of the user, such as 'something@example.com'.",
        "prompt": "In order to proceed, please provide the email for identity verification.",
        "required": True,
        "verified": True,
    }


class ShopifyGetCartSlots(ShopifySlots):
    CART_ID = {**ShopifySlots.CART_ID, "required": True}


class ShopifyGetOrderDetailsSlots(ShopifySlots):
    USER_ID = {**ShopifySlots.USER_ID, "required": True}
    ORDER_IDS = ShopifySlots.to_list(
        {
            "name": "order_id",
            "type": "str",
            "description": "The order id, such as gid://shopify/Order/1289503851427.",
            "prompt": "Please provide the order id to get the details of the order.",
            "required": False,
            "verified": True,
        }
    )
    ORDER_NAMES = ShopifySlots.to_list(
        {
            "name": "order_name",
            "type": "str",
            "description": "The order name, such as '#1001'.",
            "prompt": "Please provide the order name to get the details of the order.",
            "required": False,
            "verified": True,
        }
    )


class ShopifyGetProductsSlots(ShopifySlots):
    PRODUCT_IDS = {**ShopifySlots.PRODUCT_IDS, "required": True}


class ShopifyGetUserDetailsAdminSlots(ShopifySlots):
    USER_ID = {**ShopifySlots.USER_ID, "required": True}


class ShopifyGetWebProductSlots(ShopifySlots):
    WEB_PRODUCT_ID = {
        "name": "web_product_id",
        "type": "str",
        "description": "The product id that the user is currently seeing, such as 'gid://shopify/Product/2938501948327'.",  # If there is only 1 product, return in list with single item. If there are multiple product ids, please return all of them in a list.",
        "prompt": "In order to proceed, please choose a specific product.",
        "required": True,
        "verified": True,
    }


class ShopifyReturnProductsSlots(ShopifySlots):
    RETURN_ORDER_ID = {
        "name": "return_order_id",
        "type": "str",
        "description": "The order id to return products, such as gid://shopify/Order/1289503851427.",
        "prompt": "Please provide the order id that you would like to return products.",
        "required": True,
        "verified": True,
    }


class ShopifySearchProductsSlots(ShopifySlots):
    SEARCH_PRODUCT_QUERY = {
        "name": "product_query",
        "type": "str",
        "description": "The string query to search products, such as 'Hats'. If query is empty string, it returns all products.",
        "prompt": "In order to proceed, please provide a query for the products search.",
        "required": False,
        "verified": True,
    }
