"""
Utility functions for HubSpot tool authentication in the Arklex framework.

This module provides helper functions for authenticating HubSpot API requests within the Arklex tool system.
"""

from typing import Dict, Any
from arklex.exceptions import AuthenticationError

# Error message for missing HubSpot authentication parameters
HUBSPOT_AUTH_ERROR: str = "Missing some or all required hubspot authentication parameters: access_token. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'access_token': <hubspot_access_token>}"


def authenticate_hubspot(kwargs: Dict[str, Any]) -> str:
    """
    Authenticate with HubSpot using the provided access token.

    Args:
        kwargs (Dict[str, Any]): Dictionary containing authentication parameters

    Returns:
        str: HubSpot access token

    Raises:
        AuthenticationError: If access token is missing
    """
    access_token: str = kwargs.get("access_token")
    if not access_token:
        raise AuthenticationError(HUBSPOT_AUTH_ERROR)

    return access_token
