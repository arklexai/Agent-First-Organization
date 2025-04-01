import requests
from arklex.env.tools.shopify.auth_utils import get_access_token
from arklex.exceptions import AuthenticationError

SHOPIFY_ADMIN_AUTH_ERROR_MSG = "Missing some or all required Shopify admin authentication parameters: shop_url, api_version, admin_token. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'admin_token': <shopify_access_token>, 'shop_url': <shopify_shop_url>, 'api_version': <Shopify API version>}}"
SHOPIFY_STOREFRONT_AUTH_ERROR_MSG = "Missing some or all required Shopify storefront authentication parameters: shop_url, api_version, storefront_token. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'storefront_token': <shopify_access_token>, 'shop_url': <shopify_shop_url>, 'api_version': <Shopify API version>}}"

def authorify_admin(kwargs):
    auth = {
        "domain": kwargs.get("shop_url"),
        "version": kwargs.get("api_version"),
        "token": kwargs.get("admin_token")
    }

    if not all(auth.values()):
        raise AuthenticationError(f"Shopify admin authentication failed: {SHOPIFY_ADMIN_AUTH_ERROR_MSG}")
    return auth


def authorify_storefront(kwargs):
    auth_dict = {
        "domain": kwargs.get("shop_url"),
        "version": kwargs.get("api_version"),
        "token": kwargs.get("storefront_token")
    }
    
    if not all(auth_dict.values()):
        raise AuthenticationError(f"Shopify storefront authentication failed: {SHOPIFY_STOREFRONT_AUTH_ERROR_MSG}")
    auth = {
        "storefront_token": auth_dict["token"],
        "storefront_url": f"{auth_dict['domain']}/api/{auth_dict['version']}/graphql.json"
    }
    return auth


def make_query(url, query, variables, headers):
    """
    Make query response
    """
    request = requests.post(url, json={'query': query, 'variables': variables}, headers=headers)
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))
