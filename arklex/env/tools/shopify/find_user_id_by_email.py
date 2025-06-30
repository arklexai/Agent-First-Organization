"""
This module provides functionality to find a user's ID in the Shopify store using their email address.
It supports searching for users through the Admin API and handles cases where multiple users
might have the same email address.

Module Name: find_user_id_by_email

This file contains the code for finding a user's ID using their email address.
"""

import inspect
import json
from typing import TypedDict

import shopify

from arklex.env.tools.shopify._exception_prompt import ShopifyExceptionPrompt
from arklex.env.tools.shopify.utils import authorify_admin
from arklex.env.tools.shopify.utils_slots import (
    ShopifyFindUserByEmailSlots,
    ShopifyOutputs,
)
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


class FindUserParams(TypedDict, total=False):
    """Parameters for the find user by email tool."""

    shop_url: str
    api_version: str
    admin_token: str


description = "Find user id by email. If the user is not found, the function will return an error message."
slots = ShopifyFindUserByEmailSlots.get_all_slots()
outputs = [ShopifyOutputs.USER_ID]


@register_tool(description, slots, outputs)
def find_user_id_by_email(user_email: str, **kwargs: FindUserParams) -> str:
    """
    Find a user's ID using their email address.

    Args:
        user_email (str): The email address of the user to find.
        **kwargs (FindUserParams): Additional keyword arguments for authentication.

    Returns:
        str: The user's ID if exactly one user is found with the given email.

    Raises:
        ToolExecutionError: If:
            - No user is found with the given email
            - Multiple users are found with the same email
            - There's an error in the search process
    """
    func_name = inspect.currentframe().f_code.co_name
    auth = authorify_admin(kwargs)

    try:
        with shopify.Session.temp(**auth):
            response = shopify.GraphQL().execute(f"""
                {{
                    customers (first: 10, query: "email:{user_email}") {{
                        edges {{
                            node {{
                                id
                            }}
                        }}
                    }}
                }}
                """)
        nodes = json.loads(response)["data"]["customers"]["edges"]
        if len(nodes) == 1:
            user_id = nodes[0]["node"]["id"]
            return user_id
        else:
            raise ToolExecutionError(
                func_name,
                extra_message=ShopifyExceptionPrompt.MULTIPLE_USERS_SAME_EMAIL_ERROR_PROMPT,
            )
    except Exception as err:
        raise ToolExecutionError(
            func_name,
            extra_message=ShopifyExceptionPrompt.USER_NOT_FOUND_ERROR_PROMPT,
        ) from err
