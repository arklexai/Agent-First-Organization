"""
Utility functions for HubSpot tool authentication in the Arklex framework.

This module provides helper functions for authenticating HubSpot API requests within the Arklex tool system.
"""

import os
from datetime import datetime, timedelta
from typing import Any

import requests
from pydantic import BaseModel

from arklex.env.tools.types import ResourceAuthGroup
from arklex.utils.exceptions import AuthenticationError
from arklex.utils.logging_utils import LogContext
from arklex.utils.mysql import mysql_pool

log_context = LogContext(__name__)
# Error message for missing HubSpot authentication parameters
HUBSPOT_AUTH_ERROR: str = "Missing some or all required hubspot authentication parameters: access_token. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'access_token': <hubspot_access_token>}"


def authenticate_hubspot(kwargs: dict[str, Any]) -> str:
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


class HubspotNotIntegratedError(Exception):
    def __init__(self, hubspot_account_id: str, bot_id: str, bot_version: str) -> None:
        self.hubspot_account_id = hubspot_account_id
        self.bot_id = bot_id
        self.bot_version = bot_version
        super().__init__(
            f"HubSpot not integrated for bot {bot_id} version {bot_version}"
        )


class HubspotAuthTokens(BaseModel):
    access_token: str
    refresh_token: str
    expiry_time_str: str


def refresh_token_if_needed(
    bot_id: str, bot_version: str, hubspot_auth_tokens: HubspotAuthTokens
) -> HubspotAuthTokens:
    """Get the valid access token for HubSpot from the database.

    Args:
        bot_id: The ID of the bot
        bot_version: The version of the bot
        hubspot_auth_tokens: The current HubSpot auth tokens

    Returns:
        The valid access token string

    Raises:
        HubspotNotIntegratedError: If HubSpot is not integrated for the bot
    """
    log_context.info(
        f"Refreshing HubSpot auth tokens for bot {bot_id} version {bot_version}"
    )
    hubspot_client_id = os.getenv("HUBSPOT_CLIENT_ID")
    hubspot_client_secret = os.getenv("HUBSPOT_CLIENT_SECRET")
    if not hubspot_client_id or not hubspot_client_secret:
        raise Exception(
            "HubSpot client ID and secret not found in environment variables"
        )

    try:
        # Check if token is expired
        try:
            expiry = datetime.fromisoformat(
                hubspot_auth_tokens.expiry_time_str.replace("Z", "+00:00")
            )
            if datetime.now(expiry.tzinfo) < expiry - timedelta(minutes=15):
                return hubspot_auth_tokens
        except ValueError:
            # If expiry time is invalid, proceed with refresh
            pass
        log_context.info("hubspot token is expired, refreshing it")
        # Token is expired, refresh it
        token_refresh_url = "https://api.hubapi.com/oauth/v1/token"
        req_body = {
            "grant_type": "refresh_token",
            "client_id": hubspot_client_id,
            "client_secret": hubspot_client_secret,
            "refresh_token": hubspot_auth_tokens.refresh_token,
        }

        resp = requests.post(token_refresh_url, data=req_body)
        resp.raise_for_status()
        token_response = resp.json()

        # Create new tokens
        new_token = HubspotAuthTokens(
            access_token=token_response["access_token"],
            refresh_token=token_response["refresh_token"],
            expiry_time_str=datetime.now().replace(microsecond=0).isoformat() + "Z",
        )

        # Update tokens in database
        mysql_pool.execute(
            """
            UPDATE qa_bot_resource_permission 
            SET auth = %s 
            WHERE qa_bot_id = %s 
            AND qa_bot_version = %s 
            AND qa_bot_resource_auth_group_id = %s
            """,
            (
                new_token.model_dump_json(),
                bot_id,
                bot_version,
                ResourceAuthGroup.HUBSPOT.value,
            ),
        )

        return new_token

    except requests.exceptions.RequestException:
        # If refresh fails, return the old token
        return hubspot_auth_tokens
    except Exception as e:
        if isinstance(e, HubspotNotIntegratedError):
            raise
        raise Exception(f"Failed to get HubSpot access token: {str(e)}") from e
