"""
This module provides utility functions for OAuth 2.0 authentication with Shopify.
It supports generating authentication links, handling OAuth code flow, and managing access tokens.

Note: This module is currently inactive and reserved for future use.
It may contain experimental or planned features.

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: auth_utils

This file contains utility functions for authenticating with Shopify.
"""

import base64
import hashlib
import json
import os
import secrets
import time

import requests
from dotenv import load_dotenv

load_dotenv()

AUTH_ERROR = "error: cannot retrieve access token"


def generateCodeVerifier() -> str:
    """
    Generate a random code verifier for PKCE (Proof Key for Code Exchange).

    Returns:
        str: A URL-safe base64-encoded random string.
    """
    return (
        base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("utf-8")
    )


def generateCodeChallenge(verifier: str) -> str:
    """
    Generate a code challenge from a code verifier using SHA-256.

    Args:
        verifier (str): The code verifier to generate the challenge from.

    Returns:
        str: A URL-safe base64-encoded SHA-256 hash of the verifier.
    """
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("utf-8")


def generateState() -> str:
    """
    Generate a unique state parameter for OAuth flow.

    Returns:
        str: A unique state string combining timestamp and random bits.
    """
    return str(int(time.time()) + secrets.randbits(32))


verifier = generateCodeVerifier()
challenge = generateCodeChallenge(verifier)
state = generateState()

clientID = os.environ.get("SHOPIFY_CLIENT_ID")
shop_id = "60183707761"
redirect_uri = "https://causal-bluejay-humble.ngrok-free.app/callback"

auth_url = f"https://shopify.com/authentication/{shop_id}/oauth/authorize"
auth_params: dict[str, str] = {
    "scope": "openid email customer-account-api:full",
    "client_id": clientID,
    "response_type": "code",
    "redirect_uri": "<redirect_uri>",
    "state": state,
    "code_challenge": challenge,
    "code_challenge_method": "S256",
}


def get_auth_link(redirect_uri: str = redirect_uri) -> str:
    """
    Generate an OAuth authorization URL.

    Args:
        redirect_uri (str, optional): The redirect URI for the OAuth flow. Defaults to the configured URI.

    Returns:
        str: The complete authorization URL with all parameters.
    """
    params = auth_params.copy()
    params["redirect_uri"] = redirect_uri

    return (
        auth_url
        + "?"
        + "&".join([f"{k}={v.replace(' ', '%20')}" for k, v in params.items()])
    )


token_url = f"https://shopify.com/authentication/{shop_id}/oauth/token"
token_params: dict[str, str] = {
    "grant_type": "authorization_code",
    "client_id": clientID,
    "redirect_uri": redirect_uri,
    "code": "<code>",
    "code_verifier": verifier,
}


def get_refresh_token(code: str) -> str:
    """
    Exchange an authorization code for a refresh token.

    Args:
        code (str): The authorization code received from the OAuth flow.

    Returns:
        str: The refresh token.

    Raises:
        KeyError: If the response does not contain a refresh token.
    """
    params = token_params.copy()
    params["code"] = code
    response = requests.post(token_url, params=params)
    return json.loads(response.text)["refresh_token"]


refresh_params: dict[str, str] = {
    "grant_type": "refresh_token",
    "client_id": clientID,
    "refresh_token": "<refresh_token>",
}


def get_access_token(refresh_token: str) -> str:
    """
    Exchange a refresh token for an access token.

    Args:
        refresh_token (str): The refresh token to exchange.

    Returns:
        str: The access token.

    Raises:
        KeyError: If the response does not contain an access token.
    """
    params = refresh_params.copy()
    params["refresh_token"] = refresh_token
    response = requests.post(token_url, params=params)
    return json.loads(response.text)["access_token"]
