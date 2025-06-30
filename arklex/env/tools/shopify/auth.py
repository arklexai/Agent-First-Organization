"""
This module provides functionality for authenticating with Shopify's OAuth 2.0 system.
It handles the complete authentication flow, including generating auth links,
receiving authorization codes, and obtaining refresh tokens.

Note: This module is currently inactive and reserved for future use.
It may contain experimental or planned features.

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: auth

This file implements the function of Shopify authentication.
"""

# from arklex.tools.shopify_new.auth_utils import get_auth_link, get_refresh_token, get_access_token
# from arklex.tools.shopify_new.auth_server import authenticate_server
import os

from auth_server import authenticate_server
from auth_utils import get_auth_link, get_refresh_token


def authenticate() -> str:
    """
    Perform the complete Shopify OAuth 2.0 authentication flow.

    This function:
    1. Generates an authentication link
    2. Waits for user authentication and receives the authorization code
    3. Exchanges the code for a refresh token

    Returns:
        str: The refresh token obtained from the authentication process.

    Note:
        The access token is not currently retrieved as it's commented out in the code.
    """
    auth_link = get_auth_link()
    print("Authenticate Link here: ", auth_link)
    code = authenticate_server()
    refresh_token = get_refresh_token(code)
    # access_token = get_access_token(refresh_token)
    return refresh_token


if __name__ == "__main__":
    refresh_token = authenticate()

    os.environ["SHOPIFY_CUSTOMER_API_REFRESH_TOKEN"] = refresh_token
    print(f"Refresh token: {refresh_token}")
