"""
This module is currently inactive.

It is reserved for future use and may contain experimental or planned features (dependence on refresh token).

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: get_user_id

This file contains the code for getting the user id by refresh token.
"""
from typing import Any, Dict

from arklex.env.tools.tools import register_tool

# Customer API
from arklex.env.shopify.tools.utils_slots import ShopifySlots, ShopifyOutputs
from arklex.env.shopify.tools.utils import *
from arklex.env.shopify.tools.auth_utils import *

description = "Find user id by refresh token. If the user is not found, the function will return an error message."
slots = [
    ShopifySlots.REFRESH_TOKEN,
]
outputs = [
    ShopifyOutputs.USER_ID
]

USER_NOT_FOUND_ERROR = "error: user not found"
errors = [AUTH_ERROR, USER_NOT_FOUND_ERROR]

@register_tool(description, slots, outputs, lambda x: x not in errors or not x.startswith("error: "))
def get_user_id(refresh_token) -> str:
    try:
        return get_id(refresh_token)
    except Exception:
        return USER_NOT_FOUND_ERROR