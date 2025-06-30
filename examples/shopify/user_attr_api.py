import argparse
import json
from typing import Any

import shopify
import uvicorn
from fastapi import FastAPI

from arklex.env.tools.shopify.utils import authorify_admin
from arklex.utils.exceptions import AuthenticationError

USER_NOT_FOUND_ERROR: str = "error: No user found"
PRODUCTS_NOT_FOUND: str = "error: No products found"

app = FastAPI()


def get_users(kwargs: dict[str, Any]) -> list[dict[str, Any]] | str:
    try:
        auth: dict[str, str] = authorify_admin(kwargs)
    except AuthenticationError as e:
        print("Authentication error: ", e)
        raise AuthenticationError(e) from e

    try:
        with shopify.Session.temp(**auth):
            response: str = shopify.GraphQL().execute(r"""
                {
                    customers(first: 50) {
                        nodes {
                        id
                        firstName
                        lastName
                        email
                        phone
                        createdAt
                        updatedAt
                        numberOfOrders
                        orders(first: 5) {
                            edges {
                                node {
                                    id
                                    name
                                    createdAt
                                    cancelledAt
                                    returnStatus
                                    statusPageUrl
                                    totalPriceSet {
                                        presentmentMoney {
                                            amount
                                        }
                                    }
                                    fulfillments {
                                        displayStatus
                                        trackingInfo {
                                            number
                                            url
                                        }
                                    }
                                    lineItems(first: 10) {
                                        edges {
                                            node {
                                                id
                                                title
                                                quantity
                                                variant {
                                                    id
                                                    product {
                                                        id
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        amountSpent {
                            amount
                            currencyCode
                        }
                        lastOrder {
                            id
                            name
                        }
                        addresses {
                            id
                            firstName
                            lastName
                            company
                            address1
                            address2
                            city
                            province
                            country
                            zip
                            phone
                            name
                            provinceCode
                            countryCodeV2
                        }
                        }
                    }
                }
            """)
            data: list[dict[str, Any]] = json.loads(response)["data"]["customers"][
                "nodes"
            ]
            return data

    except Exception as e:
        print(e)
        return USER_NOT_FOUND_ERROR


def get_products(kwargs: dict[str, Any]) -> list[dict[str, Any]] | dict[str, str]:
    try:
        auth: dict[str, str] = authorify_admin(kwargs)
    except Exception as e:
        return {"error": str(e)}

    try:
        with shopify.Session.temp(**auth):
            response: str = shopify.GraphQL().execute("""
                {
                    products (first: 20) {
                        nodes {
                            id
                            title
                            description
                            totalInventory
                            onlineStoreUrl
                            options {
                                name
                                values
                            }
                            category {
                                name
                            }
                            variants (first: 3) {
                                nodes {
                                    displayName
                                    id
                                    price
                                    inventoryQuantity
                                }
                            }
                        }
                        pageInfo {
                            endCursor
                            hasNextPage
                            hasPreviousPage
                            startCursor
                        }
                    }
                }
            """)
            data: list[dict[str, Any]] = json.loads(response)["data"]["products"][
                "nodes"
            ]
            response_list: list[dict[str, Any]] = []
            for product in data:
                response_text: str = ""
                response_text += f"Product ID: {product.get('id', 'None')}\n"
                response_text += f"Title: {product.get('title', 'None')}\n"
                response_text += f"Description: {product.get('description', 'None')}\n"
                response_text += (
                    f"Total Inventory: {product.get('totalInventory', 'None')}\n"
                )
                response_text += f"Options: {product.get('options', 'None')}\n"
                response_text += "The following are several variants of the product:\n"
                for variant in product.get("variants", {}).get("nodes", []):
                    response_text += f"Variant name: {variant.get('displayName', 'None')}, Variant ID: {variant.get('id', 'None')}, Price: {variant.get('price', 'None')}, Inventory Quantity: {variant.get('inventoryQuantity', 'None')}\n"
                response_list.append(
                    {"id": product.get("id", "None"), "attribute": response_text}
                )
            return response_list
    except Exception:
        return PRODUCTS_NOT_FOUND


@app.get("/users")
def get_users_route() -> list[dict[str, Any]] | dict[str, str]:
    users: list[dict[str, Any]] = []
    try:
        response: list[dict[str, Any]] | str = get_users(kwargs)
    except AuthenticationError:
        return {
            "error": "Missing some or all required Shopify admin authentication parameters: shop_url, api_version, admin_token."
        }, 401
    except Exception as e:
        return {"error": str(e)}, 500

    for user in response:
        # attribute = f"Your name is {user['firstName']} {user['lastName']} and your email is {user['email']}. Your phone number is {user['phone']}. You registered at {user['createdAt']}. The last time you entered our store was {user['updatedAt']}. You have {user['numberOfOrders']} orders. You have spent {user['amountSpent']['amount']} {user['amountSpent']['currencyCode']} at our store."
        attribute: dict[str, Any] = user
        single_user: dict[str, Any] = {"input": user["id"], "attribute": attribute}
        users.append(single_user)
    return users


@app.get("/products")
def get_products_route() -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    response: list[dict[str, Any]] | dict[str, str] = get_products(kwargs)
    for product in response:
        print("============product============")
        print(product)
        # attribute = f"The product is {product['title']}. The description is {product['description']}. The product is in the {product['category']} category."
        single_product: dict[str, Any] = {
            "input": product["id"],
            "attribute": product["attribute"],
        }
        products.append(single_product)
        # add 50% products as home page, no product id
        single_product = {"input": None, "attribute": "home page"}
        products.append(single_product)
    return products


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    kwargs: dict[str, str] = {
        "shop_url": "<your_shop_url>",
        "api_version": "2024-10",
        "admin_token": "<your_admin_token>",
    }
    # print(get_users(kwargs))
    # print(get_products(kwargs))

    # run server
    uvicorn.run(app, host="0.0.0.0", port=args.port)
