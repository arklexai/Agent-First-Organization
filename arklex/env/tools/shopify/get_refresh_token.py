"""
This module is currently inactive.

It is reserved for future use and may contain experimental or planned features (dependence on refresh token).

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: get_refresh_token

This file contains the code for getting the refresh token of a customer through the Auth process.
"""

from arklex.env.tools.shopify.auth import authenticate
from arklex.env.tools.tools import register_tool

description = "Get refresh token of customer through the Auth process."
slots = []
outputs = [
    {
        "name": "refresh_token",
        "type": "str",
        "description": "customer's shopify refresh_token retrieved from authenticating",
    },
]

REFRESH_TOKEN_ERROR = "error: refresh token error"
errors = [REFRESH_TOKEN_ERROR]


@register_tool(description, slots, outputs, lambda x: x not in errors)
def get_order() -> str:
    try:
        return authenticate()
    except Exception:
        return REFRESH_TOKEN_ERROR
