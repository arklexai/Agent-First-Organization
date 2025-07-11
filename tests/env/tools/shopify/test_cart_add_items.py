"""Tests for arklex.env.tools.shopify.cart_add_items module."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.tools.shopify.cart_add_items import cart_add_items
from arklex.utils.exceptions import ToolExecutionError


class TestCartAddItems:
    """Test cases for cart_add_items function."""

    def setup_method(self) -> None:
        """Set up test environment."""
        os.environ["ARKLEX_TEST_ENV"] = "local"

    def teardown_method(self) -> None:
        """Clean up test environment."""
        if "ARKLEX_TEST_ENV" in os.environ:
            del os.environ["ARKLEX_TEST_ENV"]

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_success(self, mock_post: Mock) -> None:
        """Test successful addition of items to cart."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cartLinesAdd": {
                    "cart": {"checkoutUrl": "https://test-shop.myshopify.com/checkout"}
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function
        result = cart_add_items().func(
            cart_id="gid://shopify/Cart/12345",
            product_variant_ids=[
                "gid://shopify/ProductVariant/67890",
                "gid://shopify/ProductVariant/67891",
            ],
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
        )

        # Verify result
        assert "successfully added" in result
        assert "checkoutUrl" in result

        # Verify request was made with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert (
            call_args[0][0]
            == "https://test-shop.myshopify.com/api/2024-10/graphql.json"
        )

        # Verify headers
        headers = call_args[1]["headers"]
        assert headers["X-Shopify-Storefront-Access-Token"] == "test_storefront_token"

        # Verify JSON payload
        json_data = call_args[1]["json"]
        assert json_data["query"] is not None
        assert "cartLinesAdd" in json_data["query"]
        assert json_data["variables"]["cartId"] == "gid://shopify/Cart/12345"
        assert len(json_data["variables"]["lines"]) == 2
        assert (
            json_data["variables"]["lines"][0]["merchandiseId"]
            == "gid://shopify/ProductVariant/67890"
        )
        assert json_data["variables"]["lines"][0]["quantity"] == 1
        assert (
            json_data["variables"]["lines"][1]["merchandiseId"]
            == "gid://shopify/ProductVariant/67891"
        )
        assert json_data["variables"]["lines"][1]["quantity"] == 1

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_graphql_errors(self, mock_post: Mock) -> None:
        """Test cart add items when GraphQL returns errors."""
        # Setup mock response with GraphQL errors
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "errors": [
                {
                    "message": "Product variant not found",
                    "locations": [{"line": 1, "column": 10}],
                }
            ]
        }
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            cart_add_items().func(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/99999"],
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_http_error(self, mock_post: Mock) -> None:
        """Test cart add items when HTTP request fails."""
        # Setup mock response with non-200 status code
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad Request"}
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            cart_add_items().func(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/67890"],
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_request_exception(self, mock_post: Mock) -> None:
        """Test cart add items when requests.post raises an exception."""
        # Setup mock to raise exception
        mock_post.side_effect = Exception("Network error")

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            cart_add_items().func(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/67890"],
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_empty_variant_list(self, mock_post: Mock) -> None:
        """Test cart add items with empty product variant list."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cartLinesAdd": {
                    "cart": {"checkoutUrl": "https://test-shop.myshopify.com/checkout"}
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function
        result = cart_add_items().func(
            cart_id="gid://shopify/Cart/12345",
            product_variant_ids=[],
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
        )

        # Verify result
        assert "successfully added" in result

        # Verify request was made with empty lines array
        call_args = mock_post.call_args
        json_data = call_args[1]["json"]
        assert json_data["variables"]["lines"] == []

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_single_variant(self, mock_post: Mock) -> None:
        """Test cart add items with single product variant."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cartLinesAdd": {
                    "cart": {"checkoutUrl": "https://test-shop.myshopify.com/checkout"}
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function
        result = cart_add_items().func(
            cart_id="gid://shopify/Cart/12345",
            product_variant_ids=["gid://shopify/ProductVariant/67890"],
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
        )

        # Verify result
        assert "successfully added" in result

        # Verify request was made with single line
        call_args = mock_post.call_args
        json_data = call_args[1]["json"]
        assert len(json_data["variables"]["lines"]) == 1
        assert (
            json_data["variables"]["lines"][0]["merchandiseId"]
            == "gid://shopify/ProductVariant/67890"
        )
        assert json_data["variables"]["lines"][0]["quantity"] == 1

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_malformed_response(self, mock_post: Mock) -> None:
        """Test cart add items when response is malformed."""
        # Setup mock response with malformed data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cartLinesAdd": {
                    "cart": None  # Malformed response
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            cart_add_items().func(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/67890"],
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_missing_cart_data(self, mock_post: Mock) -> None:
        """Test cart add items when cart data is missing from response."""
        # Setup mock response with missing cart data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"cartLinesAdd": {}}}
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            cart_add_items().func(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/67890"],
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_missing_data_key(self, mock_post: Mock) -> None:
        """Test cart add items when response is missing 'data' key."""
        # Setup mock response with missing data key
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "cartLinesAdd": {
                "cart": {"checkoutUrl": "https://test-shop.myshopify.com/checkout"}
            }
        }
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            cart_add_items().func(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/67890"],
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_missing_cart_lines_add_key(
        self, mock_post: Mock
    ) -> None:
        """Test cart add items when response is missing 'cartLinesAdd' key."""
        # Setup mock response with missing cartLinesAdd key
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "someOtherKey": {
                    "cart": {"checkoutUrl": "https://test-shop.myshopify.com/checkout"}
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            cart_add_items().func(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/67890"],
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    def test_cart_add_items_function_registration(self) -> None:
        """Test that the cart_add_items function is properly registered as a tool."""
        # Create a tool instance to access the attributes
        tool_instance = cart_add_items()

        # Verify the function has the expected attributes from register_tool decorator
        assert hasattr(tool_instance, "func")
        assert hasattr(tool_instance, "description")
        assert hasattr(tool_instance, "slots")
        assert hasattr(tool_instance, "output")

        # Verify the description matches expected value
        assert "Add items to user's shopping cart" in tool_instance.description

        # Verify the function signature
        import inspect

        sig = inspect.signature(tool_instance.func)
        assert "cart_id" in sig.parameters
        assert "product_variant_ids" in sig.parameters
        assert "kwargs" in sig.parameters

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_large_variant_list(self, mock_post: Mock) -> None:
        """Test cart add items with a large list of product variants."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cartLinesAdd": {
                    "cart": {"checkoutUrl": "https://test-shop.myshopify.com/checkout"}
                }
            }
        }
        mock_post.return_value = mock_response

        # Create a large list of variant IDs
        variant_ids = [f"gid://shopify/ProductVariant/{i}" for i in range(100)]

        # Execute function
        result = cart_add_items().func(
            cart_id="gid://shopify/Cart/12345",
            product_variant_ids=variant_ids,
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
        )

        # Verify result
        assert "successfully added" in result

        # Verify request was made with correct number of lines
        call_args = mock_post.call_args
        json_data = call_args[1]["json"]
        assert len(json_data["variables"]["lines"]) == 100
        assert all(line["quantity"] == 1 for line in json_data["variables"]["lines"])
