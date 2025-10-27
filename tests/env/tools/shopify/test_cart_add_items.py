"""Tests for arklex.resources.tools.shopify.cart_add_items module."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.resources.tools.shopify.cart_add_items import cart_add_items
from arklex.utils.logging.exceptions import ToolExecutionError


class TestCartAddItems:
    """Test cases for cart_add_items function."""

    def setup_method(self) -> None:
        """Set up test environment."""
        os.environ["ARKLEX_TEST_ENV"] = "local"

    def get_original_function(self) -> callable:
        """Get the original function from the decorated function."""
        # Access the original function from the Tool instance
        return cart_add_items.func

    def teardown_method(self) -> None:
        """Clean up test environment."""
        if "ARKLEX_TEST_ENV" in os.environ:
            del os.environ["ARKLEX_TEST_ENV"]

    @patch("arklex.resources.tools.shopify.cart_add_items.requests.post")
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
        result = self.get_original_function()(
            cart_id="gid://shopify/Cart/12345",
            product_variant_ids=[
                "gid://shopify/ProductVariant/67890",
                "gid://shopify/ProductVariant/67891",
            ],
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "storefront_token": "test_storefront_token",
                "api_version": "2024-10",
            },
        )

        # Verify result
        assert "successfully added to the shopping cart" in result.message_flow

        # Verify the post request was made correctly
        mock_post.assert_called_once()

    @patch("arklex.resources.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_graphql_errors(self, mock_post: Mock) -> None:
        """Test cart addition when GraphQL returns errors."""
        # Setup mock response with errors
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"cartLinesAdd": {"userErrors": [{"message": "Product not found"}]}}
        }
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/99999"],
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_http_error(self, mock_post: Mock) -> None:
        """Test cart addition when HTTP request fails."""
        # Setup mock to raise HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("HTTP 500 Error")
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/67890"],
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_request_exception(self, mock_post: Mock) -> None:
        """Test cart addition when request raises exception."""
        # Setup mock to raise exception
        mock_post.side_effect = Exception("Connection error")

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/67890"],
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_empty_variant_list(self, mock_post: Mock) -> None:
        """Test cart addition with empty variant list."""
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
        result = self.get_original_function()(
            cart_id="gid://shopify/Cart/12345",
            product_variant_ids=[],
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "storefront_token": "test_storefront_token",
                "api_version": "2024-10",
            },
        )

        # Verify result
        assert "successfully added to the shopping cart" in result.message_flow

    @patch("arklex.resources.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_single_variant(self, mock_post: Mock) -> None:
        """Test cart addition with single variant."""
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
        result = self.get_original_function()(
            cart_id="gid://shopify/Cart/12345",
            product_variant_ids=["gid://shopify/ProductVariant/67890"],
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "storefront_token": "test_storefront_token",
                "api_version": "2024-10",
            },
        )

        # Verify result
        assert "successfully added to the shopping cart" in result.message_flow

    @patch("arklex.resources.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_malformed_response(self, mock_post: Mock) -> None:
        """Test cart addition with malformed JSON response."""
        # Setup mock response with malformed JSON
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/67890"],
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_missing_cart_data(self, mock_post: Mock) -> None:
        """Test cart addition when response is missing cart data."""
        # Setup mock response without cart data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"cartLinesAdd": {}}}
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/67890"],
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_missing_data_key(self, mock_post: Mock) -> None:
        """Test cart addition when response is missing 'data' key."""
        # Setup mock response without data key
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "cartLinesAdd": {"cart": {"checkoutUrl": "test"}}
        }
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/67890"],
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_missing_cart_lines_add_key(
        self, mock_post: Mock
    ) -> None:
        """Test cart addition when response is missing 'cartLinesAdd' key."""
        # Setup mock response without cartLinesAdd key
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"someOtherKey": {}}}
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/67890"],
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    def test_cart_add_items_function_registration(self) -> None:
        """Test that the cart_add_items function is properly registered as a tool."""
        # Verify the function returns a Tool instance when called
        tool_instance = cart_add_items
        from arklex.resources.tools.tools import Tool

        assert isinstance(tool_instance, Tool)

        # Verify the tool has the expected attributes
        assert hasattr(tool_instance, "description")
        assert hasattr(tool_instance, "slots")

        # Verify the description matches expected value
        assert "Add items to user's shopping cart" in tool_instance.description

    @patch("arklex.resources.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_large_variant_list(self, mock_post: Mock) -> None:
        """Test cart addition with large number of variants."""
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

        # Create large list of variant IDs
        large_variant_list = [f"gid://shopify/ProductVariant/{i}" for i in range(100)]

        # Execute function
        result = self.get_original_function()(
            cart_id="gid://shopify/Cart/12345",
            product_variant_ids=large_variant_list,
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "storefront_token": "test_storefront_token",
                "api_version": "2024-10",
            },
        )

        # Verify result
        assert "successfully added to the shopping cart" in result.message_flow
