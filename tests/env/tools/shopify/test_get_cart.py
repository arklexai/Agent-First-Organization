"""Tests for arklex.resources.tools.shopify.get_cart module."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.resources.tools.shopify.get_cart import get_cart
from arklex.utils.logging.exceptions import ToolExecutionError


class TestGetCart:
    """Test cases for get_cart function."""

    def setup_method(self) -> None:
        """Set up test environment."""
        os.environ["ARKLEX_TEST_ENV"] = "local"

    def get_original_function(self) -> callable:
        """Get the original function from the decorated function."""
        # Access the original function from the Tool instance
        return get_cart.func

    def teardown_method(self) -> None:
        """Clean up test environment."""
        if "ARKLEX_TEST_ENV" in os.environ:
            del os.environ["ARKLEX_TEST_ENV"]

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_success(self, mock_post: Mock) -> None:
        """Test successful cart retrieval."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout",
                    "lines": {
                        "nodes": [
                            {
                                "id": "gid://shopify/CartLine/1",
                                "quantity": 2,
                                "merchandise": {
                                    "id": "gid://shopify/ProductVariant/123",
                                    "title": "Test Product - Variant 1",
                                    "product": {
                                        "id": "Test Product ID",
                                        "title": "Test Product",
                                        "handle": "test-product",
                                    },
                                },
                            }
                        ]
                    },
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function
        result = self.get_original_function()(
            cart_id="gid://shopify/Cart/12345",
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "storefront_token": "test_storefront_token",
                "api_version": "2024-10",
            },
        )

        # Verify result
        assert "Test Product" in result.message_flow

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_not_found(self, mock_post: Mock) -> None:
        """Test cart retrieval when cart is not found."""
        # Setup mock response for cart not found
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"cart": None}}
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/99999",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_http_error(self, mock_post: Mock) -> None:
        """Test cart retrieval when HTTP request fails."""
        # Setup mock response with non-200 status code
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad Request"}
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_request_exception(self, mock_post: Mock) -> None:
        """Test cart retrieval when requests.post raises an exception."""
        # Setup mock to raise exception
        mock_post.side_effect = Exception("Network error")

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_empty_cart(self, mock_post: Mock) -> None:
        """Test cart retrieval with empty cart (no line items)."""
        # Setup mock response with empty cart lines
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "id": "gid://shopify/Cart/12345",
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout",
                    "lines": {
                        "nodes": [],
                        "pageInfo": {
                            "endCursor": None,
                            "hasNextPage": False,
                            "hasPreviousPage": False,
                            "startCursor": None,
                        },
                    },
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function
        result = self.get_original_function()(
            cart_id="gid://shopify/Cart/12345",
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "storefront_token": "test_storefront_token",
                "api_version": "2024-10",
            },
        )

        # Verify result contains checkout URL but no product information
        assert (
            "Checkout URL: https://test-shop.myshopify.com/checkout"
            in result.message_flow
        )
        assert "Product ID:" not in result.message_flow
        assert "Product Title:" not in result.message_flow

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_missing_product_info(self, mock_post: Mock) -> None:
        """Test cart retrieval when line items are missing product information."""
        # Setup mock response with missing product info
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "id": "gid://shopify/Cart/12345",
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout",
                    "lines": {
                        "nodes": [
                            {
                                "id": "gid://shopify/CartLine/1",
                                "quantity": 2,
                                "merchandise": {
                                    "id": "gid://shopify/ProductVariant/123",
                                    "title": "Test Product - Variant 1",
                                    # Missing product field
                                },
                            }
                        ],
                        "pageInfo": {
                            "endCursor": "cursor1",
                            "hasNextPage": False,
                            "hasPreviousPage": False,
                            "startCursor": "cursor1",
                        },
                    },
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function
        result = self.get_original_function()(
            cart_id="gid://shopify/Cart/12345",
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "storefront_token": "test_storefront_token",
                "api_version": "2024-10",
            },
        )

        # Verify result contains checkout URL but no product information
        assert (
            "Checkout URL: https://test-shop.myshopify.com/checkout"
            in result.message_flow
        )
        assert "Product ID:" not in result.message_flow
        assert "Product Title:" not in result.message_flow

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_missing_data_key(self, mock_post: Mock) -> None:
        """Test cart retrieval when response is missing 'data' key."""
        # Setup mock response with missing data key
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "cart": {
                "id": "gid://shopify/Cart/12345",
                "checkoutUrl": "https://test-shop.myshopify.com/checkout",
            }
        }
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_missing_cart_key(self, mock_post: Mock) -> None:
        """Test cart retrieval when response is missing 'cart' key."""
        # Setup mock response with missing cart key
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"someOtherKey": {"id": "gid://shopify/Cart/12345"}}
        }
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_pagination_parameters(self, mock_post: Mock) -> None:
        """Test cart retrieval with pagination parameters."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "id": "gid://shopify/Cart/12345",
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout",
                    "lines": {
                        "nodes": [
                            {
                                "id": "gid://shopify/CartLine/1",
                                "quantity": 1,
                                "merchandise": {
                                    "id": "gid://shopify/ProductVariant/67890",
                                    "title": "Test Product - Small",
                                    "product": {
                                        "id": "gid://shopify/Product/12345",
                                        "title": "Test Product",
                                    },
                                },
                            }
                        ],
                        "pageInfo": {
                            "endCursor": "cursor1",
                            "hasNextPage": True,
                            "hasPreviousPage": False,
                            "startCursor": "cursor1",
                        },
                    },
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function with pagination parameters
        result = self.get_original_function()(
            cart_id="gid://shopify/Cart/12345",
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "storefront_token": "test_storefront_token",
                "api_version": "2024-10",
            },
            limit="10",
            navigate="next",
            pageInfo='{"endCursor": "cursor1", "hasNextPage": true}',
        )

        # Verify result
        assert (
            "Checkout URL: https://test-shop.myshopify.com/checkout"
            in result.message_flow
        )
        assert "Product ID: gid://shopify/Product/12345" in result.message_flow
        assert "Product Title: Test Product" in result.message_flow

        # Verify request was made with pagination parameters
        call_args = mock_post.call_args
        json_data = call_args[1]["json"]
        query = json_data["query"]
        assert "first: 10" in query
        assert 'after: "cursor1"' in query

    def test_get_cart_function_registration(self) -> None:
        """Test that the get_cart function is properly registered as a tool."""
        # Verify the function returns a Tool instance when called
        tool_instance = get_cart
        from arklex.resources.tools.tools import Tool

        assert isinstance(tool_instance, Tool)

        # Verify the tool has the expected attributes
        assert hasattr(tool_instance, "description")
        assert hasattr(tool_instance, "slots")

        # Verify the description matches expected value
        assert "Get cart information" in tool_instance.description

        # Verify the function signature
        import inspect

        sig = inspect.signature(tool_instance.func)
        assert "cart_id" in sig.parameters
        assert "kwargs" in sig.parameters

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_malformed_lines_data(self, mock_post: Mock) -> None:
        """Test cart retrieval when lines data is malformed."""
        # Setup mock response with malformed lines data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "id": "gid://shopify/Cart/12345",
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout",
                    "lines": None,  # Malformed lines data
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_missing_lines_key(self, mock_post: Mock) -> None:
        """Test cart retrieval when cart is missing 'lines' key."""
        # Setup mock response with missing lines key
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "id": "gid://shopify/Cart/12345",
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout",
                    # Missing lines key
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_json_parsing_error(self, mock_post: Mock) -> None:
        """Test cart retrieval when JSON parsing fails."""
        # Setup mock response that raises exception on json() call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_lines_not_dict(self, mock_post: Mock) -> None:
        """Test cart retrieval when lines is not a dictionary."""
        # Setup mock response with lines as a string instead of dict
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "id": "gid://shopify/Cart/12345",
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout",
                    "lines": "invalid_lines_data",  # Not a dict
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_lines_missing_nodes(self, mock_post: Mock) -> None:
        """Test cart retrieval when lines dict is missing 'nodes' key."""
        # Setup mock response with lines dict missing nodes
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "id": "gid://shopify/Cart/12345",
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout",
                    "lines": {
                        "pageInfo": {
                            "endCursor": "cursor1",
                            "hasNextPage": False,
                            "hasPreviousPage": False,
                            "startCursor": "cursor1",
                        }
                        # Missing nodes key
                    },
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_partial_product_info(self, mock_post: Mock) -> None:
        """Test cart retrieval when product info is partially missing."""
        # Setup mock response with partial product info
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "id": "gid://shopify/Cart/12345",
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout",
                    "lines": {
                        "nodes": [
                            {
                                "id": "gid://shopify/CartLine/1",
                                "quantity": 1,
                                "merchandise": {
                                    "id": "gid://shopify/ProductVariant/67890",
                                    "title": "Test Product - Small",
                                    "product": {
                                        "id": "gid://shopify/Product/12345",
                                        # Missing title
                                    },
                                },
                            },
                            {
                                "id": "gid://shopify/CartLine/2",
                                "quantity": 2,
                                "merchandise": {
                                    "id": "gid://shopify/ProductVariant/67891",
                                    "title": "Another Product - Medium",
                                    "product": {
                                        # Missing id
                                        "title": "Another Product",
                                    },
                                },
                            },
                        ],
                        "pageInfo": {
                            "endCursor": "cursor1",
                            "hasNextPage": False,
                            "hasPreviousPage": False,
                            "startCursor": "cursor1",
                        },
                    },
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function and verify KeyError (since the code directly accesses missing fields)
        with pytest.raises(KeyError) as exc_info:
            self.get_original_function()(
                cart_id="gid://shopify/Cart/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "storefront_token": "test_storefront_token",
                    "api_version": "2024-10",
                },
            )

        assert "title" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_prev_navigation(self, mock_post: Mock) -> None:
        """Test cart retrieval with previous navigation."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "id": "gid://shopify/Cart/12345",
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout",
                    "lines": {
                        "nodes": [
                            {
                                "id": "gid://shopify/CartLine/1",
                                "quantity": 1,
                                "merchandise": {
                                    "id": "gid://shopify/ProductVariant/67890",
                                    "title": "Test Product - Small",
                                    "product": {
                                        "id": "gid://shopify/Product/12345",
                                        "title": "Test Product",
                                    },
                                },
                            }
                        ],
                        "pageInfo": {
                            "endCursor": "cursor1",
                            "hasNextPage": True,
                            "hasPreviousPage": True,
                            "startCursor": "cursor1",
                        },
                    },
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function with previous navigation
        result = self.get_original_function()(
            cart_id="gid://shopify/Cart/12345",
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "storefront_token": "test_storefront_token",
                "api_version": "2024-10",
            },
            limit="5",
            navigate="prev",
            pageInfo='{"startCursor": "cursor1", "hasPreviousPage": true}',
        )

        # Verify result
        assert (
            "Checkout URL: https://test-shop.myshopify.com/checkout"
            in result.message_flow
        )
        assert "Product ID: gid://shopify/Product/12345" in result.message_flow
        assert "Product Title: Test Product" in result.message_flow

        # Verify request was made with previous navigation parameters
        call_args = mock_post.call_args
        json_data = call_args[1]["json"]
        query = json_data["query"]
        assert "last: 5" in query
        assert 'before: "cursor1"' in query

    @patch("arklex.resources.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_stay_navigation(self, mock_post: Mock) -> None:
        """Test cart retrieval with stay navigation (default)."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "id": "gid://shopify/Cart/12345",
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout",
                    "lines": {
                        "nodes": [
                            {
                                "id": "gid://shopify/CartLine/1",
                                "quantity": 1,
                                "merchandise": {
                                    "id": "gid://shopify/ProductVariant/67890",
                                    "title": "Test Product - Small",
                                    "product": {
                                        "id": "gid://shopify/Product/12345",
                                        "title": "Test Product",
                                    },
                                },
                            }
                        ],
                        "pageInfo": {
                            "endCursor": "cursor1",
                            "hasNextPage": False,
                            "hasPreviousPage": False,
                            "startCursor": "cursor1",
                        },
                    },
                }
            }
        }
        mock_post.return_value = mock_response

        # Execute function with stay navigation
        result = self.get_original_function()(
            cart_id="gid://shopify/Cart/12345",
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "storefront_token": "test_storefront_token",
                "api_version": "2024-10",
            },
            limit="3",
            navigate="stay",
        )

        # Verify result
        assert (
            "Checkout URL: https://test-shop.myshopify.com/checkout"
            in result.message_flow
        )
        assert "Product ID: gid://shopify/Product/12345" in result.message_flow
        assert "Product Title: Test Product" in result.message_flow

        # Verify request was made with default navigation parameters
        call_args = mock_post.call_args
        json_data = call_args[1]["json"]
        query = json_data["query"]
        assert "first: 3" in query

    def test_get_cart_module_imports(self) -> None:
        """Test that all required modules are properly imported."""
        import arklex.resources.tools.shopify.get_cart as get_cart_module

        # Verify all required imports are available
        assert hasattr(get_cart_module, "inspect")
        assert hasattr(get_cart_module, "requests")
        assert hasattr(get_cart_module, "ShopifyExceptionPrompt")
        assert hasattr(get_cart_module, "authorify_storefront")
        assert hasattr(get_cart_module, "register_tool")
        assert hasattr(get_cart_module, "ToolExecutionError")
