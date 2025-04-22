"""
This module is currently inactive.

It is reserved for future use and may contain experimental or planned features.

Status:
    - Not in use (as of 2025-02-18)
    - Intended for future feature expansion

Module Name: auth_utils

This file contains utility functions for authenticating with Shopify.
"""
import os
import requests
import json
import time

import secrets
import hashlib
import base64

from dotenv import load_dotenv
load_dotenv()

AUTH_ERROR = "error: cannot retrieve access token"

def generateCodeVerifier():
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b'=').decode('utf-8')

def generateCodeChallenge(verifier):
    digest = hashlib.sha256(verifier.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode('utf-8')

def generateState():
    return str(int(time.time()) + secrets.randbits(32))

verifier = generateCodeVerifier()
challenge = generateCodeChallenge(verifier)
state = generateState()

clientID = os.environ.get('SHOPIFY_CLIENT_ID')
shop_id = '60183707761'
redirect_uri = "https://causal-bluejay-humble.ngrok-free.app/callback"

auth_url = f"https://shopify.com/authentication/{shop_id}/oauth/authorize"
auth_params = {
  'scope': 'openid email customer-account-api:full',
  'client_id': clientID,
  'response_type': 'code',
  'redirect_uri': '<redirect_uri>',
  'state': state,
  'code_challenge': challenge,
  'code_challenge_method': 'S256',
}
def get_auth_link(redirect_uri=redirect_uri):
    params = auth_params.copy()
    params['redirect_uri'] = redirect_uri
    
    return auth_url + '?' + '&'.join([f"{k}={v.replace(' ', '%20')}" for k, v in params.items()])

token_url = f"https://shopify.com/authentication/{shop_id}/oauth/token"
token_params = {
  'grant_type': 'authorization_code',
  'client_id': clientID,
  'redirect_uri': redirect_uri,
  'code': '<code>',
  'code_verifier': verifier,
}
def get_refresh_token(code):
  params = token_params.copy()
  params['code'] = code
  response = requests.post(token_url, params=params)
  return json.loads(response.text)['refresh_token']

refresh_params = {
  'grant_type': 'refresh_token',
  'client_id': clientID,
  'refresh_token': '<refresh_token>'
}
def get_access_token(refresh_token: str) -> str:
    params = refresh_params.copy()
    params['refresh_token'] = refresh_token
    response = requests.post(token_url, params=params)
    return json.loads(response.text)['access_token']