from arklex.env.types import Resource
RESOURCES = [

    Resource(
        id = "2a2750cb-6226-4068-ba05-a4db83da3e16",
        name = "get_order_details",
        path = "shopify/get_order_details.py",
        fixed_args = {"admin_token": "<admin_token>", "shop_url": "<url>", "api_version": "<version>"},
      
    ),
    Resource(
        id = "22fae76f-085c-4098-9011-2ae1e1eb8dc3",
        name = "get_products",
        path = "shopify/get_products.py",
        fixed_args = {"admin_token": "<admin_token>", "shop_url": "<url>", "api_version": "<version>"},

    ),
    Resource(
        id = "2b275abc-6226-2013-ba05-t4ab83daalc3",
        name = "search_products",
        path = "shopify/search_products.py",
        fixed_args = {"admin_token": "<admin_token>", "shop_url": "<url>", "api_version": "<version>"},
    ),
    Resource(
        id = "alla05l2-3kd1-x9iw-10k3-algk3xenfsl9",
        name = "cancel_order",
        path = "shopify/cancel_order.py",
        fixed_args = {"admin_token": "<admin_token>", "shop_url": "<url>", "api_version": "<version>"},
        
    ),
    Resource(
        id = "2alak3db-sl36-4zk9-aa35-a4dlkfm3se16",
        name = "cart_add_items",
        path = "shopify/cart_add_items.py",
        fixed_args = {"admin_token": "<admin_token>", "shop_url": "<url>", "api_version": "<version>"},
    ),
    Resource(
        id = "alfseal2-al94-2kdq-slci-aldjcjenfead",
        name = "get_cart",
        path = "shopify/get_cart.py",
        fixed_args = {"storefront_token": "<storefront_token>", "shop_url": "<url>", "api_version": "<version>"},
    ),
    Resource(
        id = "55011bc1-2a55-4e21-bf39-e9624729c8d8",
        name = "get_user_details_admin",
        path = "shopify/get_user_details_admin.py",
        fixed_args = {"admin_token": "<admin_token>", "shop_url": "<url>", "api_version": "<version>"},
    ),
    Resource(
        id = "xl34e76f-025c-4xl2-0s2j-l4e1eal2naak",
        name = "get_web_product",
        path = "shopify/get_web_product.py",
        fixed_args = {"admin_token": "<admin_token>", "shop_url": "<url>", "api_version": "<version>"},
        
    ),
    Resource(
        id = "alfse0ls-lx4f-a01m-1mch-a4dfsl010end",
        name = "return_products",
        path = "shopify/return_products.py",
        fixed_args = {"admin_token": "<admin_token>", "shop_url": "<url>", "api_version": "<version>"},
        
    ),
   
]
