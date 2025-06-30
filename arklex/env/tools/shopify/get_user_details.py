"""
This module provides functionality to retrieve detailed user information from the Shopify store.
It supports retrieving user profile details, contact information, and order history.

The module:
1. Uses the Shopify Customer API to fetch user data
2. Supports pagination and cursor-based navigation
3. Handles authentication using refresh tokens
4. Returns user information in a structured format

Note: This module is currently inactive and reserved for future use.
It may contain experimental or planned features (dependence on refresh token).

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: get_user_details

This file contains the code for retrieving user details from Shopify.
"""

from typing import TypedDict

from arklex.env.tools.shopify.auth_utils import AUTH_ERROR, get_access_token
from arklex.env.tools.shopify.utils import make_query
from arklex.env.tools.shopify.utils_nav import (
    PAGEINFO_OUTPUTS,
    PAGEINFO_SLOTS,
    cursorify,
)

# Customer API
from arklex.env.tools.shopify.utils_slots import ShopifyOutputs, ShopifySlots
from arklex.env.tools.tools import register_tool

# Placeholder values for undefined variables (module is inactive)
customer_url = "https://placeholder.myshopify.com/api/2024-04/graphql.json"
customer_headers = {"Content-Type": "application/json"}

description = "Get the details of a user."
slots = [ShopifySlots.REFRESH_TOKEN, *PAGEINFO_SLOTS]
outputs = [ShopifyOutputs.USER_DETAILS, *PAGEINFO_OUTPUTS]

USER_NOT_FOUND_ERROR = "error: user not found"
errors = [USER_NOT_FOUND_ERROR]


class GetUserDetailsParams(TypedDict, total=False):
    """Parameters for the get user details tool."""

    limit: str
    navigate: str
    pageInfo: str


@register_tool(description, slots, outputs, lambda x: x not in errors)
def get_user_details(
    refresh_token: str, **kwargs: GetUserDetailsParams
) -> tuple[dict[str, str], dict[str, str]] | str:
    """
    Retrieve detailed information about a user's profile and their orders.

    Args:
        refresh_token (str): The refresh token used for authentication.
            This token is used to obtain an access token for API requests.
        **kwargs (GetUserDetailsParams): Additional keyword arguments for pagination.

    Returns:
        Union[Tuple[Dict[str, str], Dict[str, str]], str]: Either:
            - A tuple containing:
                - Dict[str, str]: User details including:
                    - ID, first name, last name
                    - Email address and phone number
                    - Creation date
                    - Default address
                    - Order IDs
                - Dict[str, str]: Page information for pagination including:
                    - endCursor
                    - hasNextPage
                    - hasPreviousPage
                    - startCursor
            - str: Error message if the operation fails

    Raises:
        PermissionError: If the user is not authorized to access the details.
        Exception: If there's an error retrieving the user information or
                  if the authentication process fails.

    Note:
        This function requires a valid refresh token for authentication.
        The response includes both user details and pagination information
        for order history.
    """
    nav = cursorify(kwargs)
    if not nav[1]:
        return nav[0]
    try:
        body = f"""
            query {{ 
                customer {{ 
                    id
                    firstName
                    lastName
                    emailAddress {{
                        emailAddress
                    }}
                    phoneNumber {{
                        phoneNumber
                    }}
                    creationDate
                    defaultAddress {{
                        formatted
                    }}
                    orders ({nav[0]}) {{
                        nodes {{
                            id
                        }}
                        pageInfo {{
                            endCursor
                            hasNextPage
                            hasPreviousPage
                            startCursor
                        }}
                    }}
                }}
            }}
        """
        try:
            auth: dict[str, str] = {"Authorization": get_access_token(refresh_token)}
        except Exception:
            return AUTH_ERROR

        try:
            response: dict[str, str] = make_query(
                customer_url, body, {}, customer_headers | auth
            )["data"]["customer"]
        except Exception as e:
            return f"error: {e}"

        pageInfo: dict[str, str] = response["orders"]["pageInfo"]
        return response, pageInfo

    except Exception as e:
        raise PermissionError from e
