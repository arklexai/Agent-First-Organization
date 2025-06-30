"""
This module provides functionality to retrieve detailed user information from the Shopify store
using the Admin API. It supports retrieving comprehensive user profiles including contact details,
order history, and spending information.

The module:
1. Uses the Shopify Admin API to fetch user data
2. Supports pagination and cursor-based navigation
3. Handles authentication and error cases
4. Returns user information in a structured JSON format

Module Name: get_user_details_admin

This file contains the code for retrieving user details using the Shopify Admin API.
"""

import inspect
import json
from typing import TypedDict

# Admin API
import shopify

from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_nav import PAGEINFO_OUTPUTS, cursorify
from arklex.env.tools.shopify.utils_slots import (
    ShopifyGetUserDetailsAdminSlots,
    ShopifyOutputs,
)
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError

description = "Get the details of a user with Admin API."
slots = ShopifyGetUserDetailsAdminSlots.get_all_slots()
outputs = [ShopifyOutputs.USER_DETAILS, *PAGEINFO_OUTPUTS]


class GetUserDetailsAdminParams(TypedDict, total=False):
    """Parameters for the get user details admin tool."""

    shop_url: str
    api_version: str
    admin_token: str
    limit: str
    navigate: str
    pageInfo: str


@register_tool(description, slots, outputs)
def get_user_details_admin(user_id: str, **kwargs: GetUserDetailsAdminParams) -> str:
    """
    Retrieve detailed information about a user using the Shopify Admin API.

    Args:
        user_id (str): The ID of the user to retrieve information for.
            Can be a full user ID or just the numeric portion.
        **kwargs (GetUserDetailsAdminParams): Additional keyword arguments for pagination and authentication.

    Returns:
        str: A JSON string containing detailed user information, including:
            - First and last name
            - Email and phone
            - Order count and spending amount
            - Account creation and update dates
            - Account notes and verification status
            - Tags and lifetime duration
            - Addresses
            - Order IDs

    Raises:
        ToolExecutionError: If:
            - The user is not found
            - There's an error retrieving the user information
            - The API request fails

    Note:
        The function expects a valid Shopify user ID. The response is returned
        as a JSON string that can be parsed to access individual user details.
    """
    func_name = inspect.currentframe().f_code.co_name
    nav = cursorify(kwargs)
    if not nav[1]:
        return nav[0]
    auth = authorify_admin(kwargs)

    try:
        with shopify.Session.temp(**auth):
            response = shopify.GraphQL().execute(f"""
                {{
                    customer(id: "{user_id}")  {{ 
                        firstName
                        lastName
                        email
                        phone
                        numberOfOrders
                        amountSpent {{
                            amount
                            currencyCode
                        }}
                        createdAt
                        updatedAt
                        note
                        verifiedEmail
                        validEmailAddress
                        tags
                        lifetimeDuration
                        addresses {{
                            address1
                        }}
                        orders ({nav[0]}) {{
                            nodes {{
                                id
                            }}
                        }}
                    }}
                }}
            """)
            data: dict[str, str] | None = json.loads(response)["data"]["customer"]
            if data:
                return json.dumps(data)
            else:
                raise ToolExecutionError(
                    func_name,
                    extra_message=ShopifyExceptionPrompt.USER_NOT_FOUND_PROMPT,
                )

    except Exception as e:
        raise ToolExecutionError(
            func_name,
            extra_message=ShopifyExceptionPrompt.USER_NOT_FOUND_PROMPT,
        ) from e
