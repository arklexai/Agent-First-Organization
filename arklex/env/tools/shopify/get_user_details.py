"""
This module is currently inactive.

It is reserved for future use and may contain experimental or planned features (dependence on refresh token).

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: get_user_details

This file contains the code for getting the details of a user.
"""

from typing import Any, Dict, List, Tuple, Union
from arklex.env.tools.tools import register_tool

# Customer API
from arklex.env.tools.shopify.utils_slots import ShopifySlots, ShopifyOutputs
from arklex.env.tools.shopify.utils import *
from arklex.env.tools.shopify.auth_utils import *
from arklex.env.tools.shopify.utils_nav import *

description: str = "Get the details of a user."
slots: List[Dict[str, Any]] = [ShopifySlots.REFRESH_TOKEN, *PAGEINFO_SLOTS]
outputs: List[Dict[str, Any]] = [ShopifyOutputs.USER_DETAILS, *PAGEINFO_OUTPUTS]

USER_NOT_FOUND_ERROR: str = "error: user not found"
errors: List[str] = [USER_NOT_FOUND_ERROR]


@register_tool(description, slots, outputs, lambda x: x not in errors)
def get_user_details(
    refresh_token: str, **kwargs: Any
) -> Union[str, Tuple[Dict[str, Any], Dict[str, Any]]]:
    nav: Tuple[str, bool] = cursorify(kwargs)
    if not nav[1]:
        return nav[0]
    try:
        body: str = f"""
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
            auth: Dict[str, str] = {"Authorization": get_access_token(refresh_token)}
        except:
            return AUTH_ERROR

        try:
            response: Dict[str, Any] = make_query(
                customer_url, body, {}, customer_headers | auth
            )["data"]["customer"]
        except Exception as e:
            return f"error: {e}"

        pageInfo: Dict[str, Any] = response["orders"]["pageInfo"]
        return response, pageInfo

    except Exception:
        raise PermissionError
