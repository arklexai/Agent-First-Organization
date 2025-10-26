import json
from unittest.mock import MagicMock, Mock, patch

from arklex.resources.tools.shopify.search_products import search_products


@patch("arklex.resources.tools.shopify.search_products.shopify.Session.temp")
@patch("arklex.resources.tools.shopify.search_products.shopify.GraphQL")
def test_search_products(mock_graphql: Mock, mock_session_temp: Mock) -> None:
    mock_session = MagicMock()
    mock_session_temp.return_value.__enter__.return_value = mock_session
    mock_graphql_instance = MagicMock()
    mock_response = {
        "data": {
            "products": {
                "nodes": [
                    {
                        "id": "gid://shopify/Product/123",
                        "title": "Test Product",
                        "description": "This is a test product description that is longer than 180 characters to test the truncation functionality in the search products function.",
                        "handle": "test-product",
                        "onlineStoreUrl": "https://test-shop.myshopify.com/products/test-product",
                        "images": {
                            "edges": [
                                {
                                    "node": {
                                        "src": "https://cdn.shopify.com/test-image.jpg",
                                        "altText": "Test Product Image",
                                    }
                                }
                            ]
                        },
                        "variants": {
                            "nodes": [
                                {
                                    "displayName": "Small",
                                    "id": "gid://shopify/ProductVariant/456",
                                    "price": "19.99",
                                    "inventoryQuantity": 10,
                                },
                                {
                                    "displayName": "Medium",
                                    "id": "gid://shopify/ProductVariant/457",
                                    "price": "21.99",
                                    "inventoryQuantity": 5,
                                },
                            ]
                        },
                    }
                ],
                "pageInfo": {
                    "endCursor": "cursor1",
                    "hasNextPage": False,
                    "hasPreviousPage": False,
                    "startCursor": "cursor1",
                },
            }
        }
    }
    mock_graphql_instance.execute.return_value = json.dumps(mock_response)
    mock_graphql.return_value = mock_graphql_instance

    search_product_function = search_products.func
    result = search_product_function(
        product_query="test product",
        auth={
            "shop_url": "https://test-shop.myshopify.com",
            "admin_token": "test_admin_token",
            "api_version": "2024-10",
        },
        llm_config={"llm_provider": "openai", "model_type_or_path": "gpt-4o"},
    )

    assert hasattr(result, "response")
    response_data = json.loads(result.response)
    assert len(response_data["answer"]) > 0
    assert len(response_data["card_list"]) > 0
