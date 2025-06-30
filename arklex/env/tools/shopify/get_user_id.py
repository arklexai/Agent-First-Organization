"""
This module provides functionality to retrieve a user's ID from the Shopify store using their refresh token.
It supports authentication and user identification through the Shopify Customer API.

The module:
1. Uses the Shopify Customer API to fetch user ID
2. Handles authentication using refresh tokens
3. Provides error handling for authentication and user lookup failures
4. Returns either the user ID or an error message

Note: This module is currently inactive and reserved for future use.
It may contain experimental or planned features (dependence on refresh token).

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: get_user_id

This file contains the code for retrieving a user's ID using their refresh token.
"""

from typing import Literal

from arklex.env.tools.shopify.auth_utils import AUTH_ERROR

# Customer API
from arklex.env.tools.shopify.utils_slots import ShopifyOutputs, ShopifySlots
from arklex.env.tools.tools import register_tool


# Placeholder function for undefined get_id (module is inactive)
def get_id(refresh_token: str) -> str:
    """Placeholder function for getting user ID (module is inactive)."""
    return "gid://shopify/Customer/placeholder"


description = "Find user id by refresh token. If the user is not found, the function will return an error message."
slots = [
    ShopifySlots.REFRESH_TOKEN,
]
outputs = [ShopifyOutputs.USER_ID]

USER_NOT_FOUND_ERROR = "error: user not found"
errors = [AUTH_ERROR, USER_NOT_FOUND_ERROR]


@register_tool(
    description,
    slots,
    outputs,
    lambda x: x not in errors or not x.startswith("error: "),
)
def get_user_id(refresh_token: str) -> str | Literal["error: user not found"]:
    """
    Retrieve a user's ID using their refresh token.

    Args:
        refresh_token (str): The refresh token used for authentication.
            This token is used to obtain an access token for API requests.

    Returns:
        Union[str, Literal["error: user not found"]]: Either:
            - str: The user's ID if successful (format: "gid://shopify/Customer/123456")
            - Literal["error: user not found"]: Error message if the user is not found
              or authentication fails

    Raises:
        None: All errors are caught and returned as strings.

    Note:
        This function requires a valid refresh token for authentication.
        The user ID is returned in Shopify's Global ID format.
    """
    try:
        return get_id(refresh_token)
    except Exception:
        return USER_NOT_FOUND_ERROR
