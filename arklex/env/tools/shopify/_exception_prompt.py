class ExceptionPrompt:
    # cancel_order exception prompt
    ORDER_CANCEL_ERROR_PROMPT = "Order cancel failed, please try again later or refresh the chat window."
    
    # cart_add_items exception prompt
    CART_ADD_ITEMS_ERROR_PROMPT = "Products could not be added to cart, please try again later or refresh the chat window."

    # find_user_id_by_email exception prompt
    USER_NOT_FOUND_ERROR_PROMPT = "User not found"
    MULTIPLE_USERS_SAME_EMAIL_ERROR_PROMPT = "There are multiple users with the same email"

    # get_cart exception prompt
    CART_NOT_FOUND_ERROR_PROMPT  = "Shopping cart not found, please add any item to the cart to initialize the cart."

    # get_order_details exception prompt
    ORDERS_NOT_FOUND_PROMPT = "There is a problem with the order. Please try again later or refresh the chat window."

    # get_product_images exception prompt
    PRODUCTS_NOT_FOUND_PROMPT = "Product not found, please try again later or refresh the chat window."

    # get_products exception prompt
    PRODUCTS_NOT_FOUND_PROMPT = "Could not find any product information. Please try again later or refresh the chat window."

    # get_user_details_admin exception prompt
    USER_NOT_FOUND_PROMPT = "Could not find the user. Please try again later or refresh the chat window."

    # get_web_product exception prompt
    PRODUCT_NOT_FOUND_PROMPT = "Could not find the product. Please try another product."

    # return_products exception prompt
    PRODUCT_RETURN_ERROR_PROMPT = "Product return failed, please try again later or refresh the chat window."
    NO_FULFILLMENT_FOUND_ERROR_PROMPT = "There is no item in the order that can be returned."

    # search_products exception prompt
    PRODUCT_SEARCH_ERROR_PROMPT = "Could not find any products matching the query. Please try again with a different query or refresh the chat window."
