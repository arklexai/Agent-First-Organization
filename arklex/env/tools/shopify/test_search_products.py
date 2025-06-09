import pytest
import json
from unittest.mock import patch, MagicMock
from arklex.env.tools.shopify.search_products import search_products
from arklex.exceptions import ToolExecutionError, AuthenticationError

# Replace these with your real Shopify credentials for live testing
SHOP_URL = "yourshop.myshopify.com"
API_VERSION = "2024-01"
ADMIN_TOKEN = "your-access-token"

# Mock response data
MOCK_PRODUCTS_RESPONSE = {
    "data": {
        "products": {
            "edges": [
                {
                    "node": {
                        "id": "gid://shopify/Product/1",
                        "title": "Test Product",
                        "description": "Test Description",
                        "handle": "test-product",
                        "priceRange": {
                            "minVariantPrice": {
                                "amount": "19.99",
                                "currencyCode": "USD",
                            }
                        },
                        "images": {
                            "edges": [{"node": {"url": "https://test.com/image.jpg"}}]
                        },
                        "variants": {
                            "edges": [
                                {
                                    "node": {
                                        "id": "gid://shopify/ProductVariant/1",
                                        "title": "Default Title",
                                        "price": "19.99",
                                        "inventoryQuantity": 10,
                                    }
                                }
                            ]
                        },
                    }
                }
            ]
        }
    }
}


@pytest.fixture
def mock_auth():
    return {
        "shop_url": SHOP_URL,
        "api_version": API_VERSION,
        "admin_token": ADMIN_TOKEN,
    }


@pytest.fixture
def mock_session():
    with patch("shopify.Session") as mock:
        yield mock


@pytest.fixture
def mock_graphql():
    with patch("shopify.GraphQL") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        mock_instance.execute.return_value = json.dumps(MOCK_PRODUCTS_RESPONSE)
        yield mock_instance


def test_search_products_success(mock_auth, mock_session, mock_graphql):
    """Test successful product search."""
    kwargs = {
        "product_query": "test",
        "shop_url": SHOP_URL,
        "api_version": API_VERSION,
        "admin_token": ADMIN_TOKEN,
    }

    with patch(
        "arklex.env.tools.shopify.utils.authorify_admin", return_value=mock_auth
    ):
        result = search_products(**kwargs)
        result_data = json.loads(result)

        assert "products" in result_data
        assert len(result_data["products"]) == 1
        product = result_data["products"][0]
        assert product["title"] == "Test Product"
        assert product["price"] == "19.99"
        assert product["currency"] == "USD"


def test_search_products_no_query(mock_auth, mock_session, mock_graphql):
    """Test product search with no query."""
    kwargs = {
        "shop_url": SHOP_URL,
        "api_version": API_VERSION,
        "admin_token": ADMIN_TOKEN,
    }

    with patch(
        "arklex.env.tools.shopify.utils.authorify_admin", return_value=mock_auth
    ):
        result = search_products(**kwargs)
        result_data = json.loads(result)

        assert "products" in result_data
        assert len(result_data["products"]) == 1


def test_search_products_auth_error(mock_auth):
    """Test product search with authentication error."""
    kwargs = {"product_query": "test"}

    with patch(
        "arklex.env.tools.shopify.utils.authorify_admin",
        side_effect=AuthenticationError("Auth failed"),
    ):
        with pytest.raises(AuthenticationError):
            search_products(**kwargs)


def test_search_products_graphql_error(mock_auth, mock_session):
    """Test product search with GraphQL error."""
    kwargs = {
        "product_query": "test",
        "shop_url": SHOP_URL,
        "api_version": API_VERSION,
        "admin_token": ADMIN_TOKEN,
    }

    with patch(
        "arklex.env.tools.shopify.utils.authorify_admin", return_value=mock_auth
    ):
        with patch("shopify.GraphQL") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            mock_instance.execute.side_effect = Exception("GraphQL error")

            with pytest.raises(ToolExecutionError):
                search_products(**kwargs)
