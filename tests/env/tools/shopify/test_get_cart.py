"""Tests for arklex.env.tools.shopify.get_cart module."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.tools.shopify.get_cart import get_cart
from arklex.utils.exceptions import ToolExecutionError


class TestGetCart:
    """Test cases for get_cart function."""

    def setup_method(self) -> None:
        """Set up test environment."""
        os.environ["ARKLEX_TEST_ENV"] = "local"

    def teardown_method(self) -> None:
        """Clean up test environment."""
        if "ARKLEX_TEST_ENV" in os.environ:
            del os.environ["ARKLEX_TEST_ENV"]

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_success(self, mock_post: Mock) -> None:
        """Test successful cart retrieval."""
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
                                "quantity": 2,
                                "merchandise": {
                                    "id": "gid://shopify/ProductVariant/67890",
                                    "title": "Test Product - Small",
                                    "product": {
                                        "id": "gid://shopify/Product/12345",
                                        "title": "Test Product",
                                    },
                                },
                            },
                            {
                                "id": "gid://shopify/CartLine/2",
                                "quantity": 1,
                                "merchandise": {
                                    "id": "gid://shopify/ProductVariant/67891",
                                    "title": "Another Product - Medium",
                                    "product": {
                                        "id": "gid://shopify/Product/12346",
                                        "title": "Another Product",
                                    },
                                },
                            },
                        ],
                        "pageInfo": {
                            "endCursor": "cursor2",
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
        result = get_cart().func(
            cart_id="gid://shopify/Cart/12345",
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
        )

        # Verify result
        assert "Checkout URL: https://test-shop.myshopify.com/checkout" in result
        assert "Product ID: gid://shopify/Product/12345" in result
        assert "Product Title: Test Product" in result
        assert "Product ID: gid://shopify/Product/12346" in result
        assert "Product Title: Another Product" in result

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
        assert "cart(id: $id)" in json_data["query"]
        assert json_data["variables"]["id"] == "gid://shopify/Cart/12345"

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_not_found(self, mock_post: Mock) -> None:
        """Test cart retrieval when cart is not found."""
        # Setup mock response with null cart
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"cart": None}}
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            get_cart().func(
                cart_id="gid://shopify/Cart/99999",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_http_error(self, mock_post: Mock) -> None:
        """Test cart retrieval when HTTP request fails."""
        # Setup mock response with non-200 status code
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad Request"}
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_request_exception(self, mock_post: Mock) -> None:
        """Test cart retrieval when requests.post raises an exception."""
        # Setup mock to raise exception
        mock_post.side_effect = Exception("Network error")

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
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
        result = get_cart().func(
            cart_id="gid://shopify/Cart/12345",
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
        )

        # Verify result contains checkout URL but no product information
        assert "Checkout URL: https://test-shop.myshopify.com/checkout" in result
        assert "Product ID:" not in result
        assert "Product Title:" not in result

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
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
                                    "id": "gid://shopify/ProductVariant/67890",
                                    "title": "Test Product - Small",
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
        result = get_cart().func(
            cart_id="gid://shopify/Cart/12345",
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
        )

        # Verify result contains checkout URL but no product information
        assert "Checkout URL: https://test-shop.myshopify.com/checkout" in result
        assert "Product ID:" not in result
        assert "Product Title:" not in result

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
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
            get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
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
            get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
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
        result = get_cart().func(
            cart_id="gid://shopify/Cart/12345",
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
            limit="10",
            navigate="next",
            pageInfo='{"endCursor": "cursor1", "hasNextPage": true}',
        )

        # Verify result
        assert "Checkout URL: https://test-shop.myshopify.com/checkout" in result
        assert "Product ID: gid://shopify/Product/12345" in result
        assert "Product Title: Test Product" in result

        # Verify request was made with pagination parameters
        call_args = mock_post.call_args
        json_data = call_args[1]["json"]
        query = json_data["query"]
        assert "first: 10" in query
        assert 'after: "cursor1"' in query

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_navigation_return_early(self, mock_post: Mock) -> None:
        """Test cart retrieval when navigation returns early."""
        # This test simulates the case where cursorify returns early
        # We need to mock the cursorify function to return (nav_string, False)
        with patch("arklex.env.tools.shopify.get_cart.cursorify") as mock_cursorify:
            mock_cursorify.return_value = ("first: 10", False)

            # Execute function
            result = get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
                limit="10",
            )

            # Verify result is the navigation string
            assert result == "first: 10"

            # Verify no HTTP request was made
            mock_post.assert_not_called()

    def test_get_cart_function_registration(self) -> None:
        """Test that the get_cart function is properly registered as a tool."""
        # Get the tool instance
        tool_instance = get_cart()

        # Verify the function has the expected attributes
        assert hasattr(tool_instance, "func")
        assert hasattr(tool_instance, "description")
        assert hasattr(tool_instance, "slots")
        assert hasattr(tool_instance, "output")

        # Verify the description matches expected value
        assert "Get cart information" in tool_instance.description

        # Verify the function signature
        import inspect

        sig = inspect.signature(tool_instance.func)
        assert "cart_id" in sig.parameters
        assert "kwargs" in sig.parameters

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
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
            get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
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
            get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_json_parsing_error(self, mock_post: Mock) -> None:
        """Test cart retrieval when JSON parsing fails."""
        # Setup mock response that raises exception on json() call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
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
            get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
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
            get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
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
            get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )

        assert "title" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
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
        result = get_cart().func(
            cart_id="gid://shopify/Cart/12345",
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
            limit="5",
            navigate="prev",
            pageInfo='{"startCursor": "cursor1", "hasPreviousPage": true}',
        )

        # Verify result
        assert "Checkout URL: https://test-shop.myshopify.com/checkout" in result
        assert "Product ID: gid://shopify/Product/12345" in result
        assert "Product Title: Test Product" in result

        # Verify request was made with previous navigation parameters
        call_args = mock_post.call_args
        json_data = call_args[1]["json"]
        query = json_data["query"]
        assert "last: 5" in query
        assert 'before: "cursor1"' in query

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
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
        result = get_cart().func(
            cart_id="gid://shopify/Cart/12345",
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
            limit="3",
            navigate="stay",
        )

        # Verify result
        assert "Checkout URL: https://test-shop.myshopify.com/checkout" in result
        assert "Product ID: gid://shopify/Product/12345" in result
        assert "Product Title: Test Product" in result

        # Verify request was made with default navigation parameters
        call_args = mock_post.call_args
        json_data = call_args[1]["json"]
        query = json_data["query"]
        assert "first: 3" in query

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_invalid_pageinfo_json(self, mock_post: Mock) -> None:
        """Test cart retrieval with invalid pageInfo JSON."""
        # This test simulates the case where pageInfo is invalid JSON
        with patch("arklex.env.tools.shopify.get_cart.cursorify") as mock_cursorify:
            mock_cursorify.return_value = (
                "error: cannot navigate without reference cursor",
                False,
            )

            # Execute function
            result = get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
                limit="10",
                navigate="next",
                pageInfo="invalid json",
            )

            # Verify result is the error message
            assert result == "error: cannot navigate without reference cursor"

            # Verify no HTTP request was made
            mock_post.assert_not_called()

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_no_next_page(self, mock_post: Mock) -> None:
        """Test cart retrieval when there's no next page."""
        # This test simulates the case where there's no next page
        with patch("arklex.env.tools.shopify.get_cart.cursorify") as mock_cursorify:
            mock_cursorify.return_value = ("error: no more pages after", False)

            # Execute function
            result = get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
                limit="10",
                navigate="next",
                pageInfo='{"endCursor": "cursor1", "hasNextPage": false}',
            )

            # Verify result is the error message
            assert result == "error: no more pages after"

            # Verify no HTTP request was made
            mock_post.assert_not_called()

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_no_prev_page(self, mock_post: Mock) -> None:
        """Test cart retrieval when there's no previous page."""
        # This test simulates the case where there's no previous page
        with patch("arklex.env.tools.shopify.get_cart.cursorify") as mock_cursorify:
            mock_cursorify.return_value = ("error: no more pages before", False)

            # Execute function
            result = get_cart().func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
                limit="10",
                navigate="prev",
                pageInfo='{"startCursor": "cursor1", "hasPreviousPage": false}',
            )

            # Verify result is the error message
            assert result == "error: no more pages before"

            # Verify no HTTP request was made
            mock_post.assert_not_called()

    def test_get_cart_typing_annotations(self) -> None:
        """Test that the get_cart function has proper typing annotations."""
        from typing import get_type_hints

        # Get the function signature
        type_hints = get_type_hints(get_cart().func)

        # Verify parameter types
        assert type_hints["cart_id"] is str
        # Note: kwargs type is GetCartParams, not dict
        assert "kwargs" in type_hints

        # Verify return type
        assert type_hints["return"] is str

    def test_get_cart_module_imports(self) -> None:
        """Test that all required modules are properly imported."""
        import arklex.env.tools.shopify.get_cart as get_cart_module

        # Verify all required imports are available
        assert hasattr(get_cart_module, "inspect")
        assert hasattr(get_cart_module, "TypedDict")
        assert hasattr(get_cart_module, "requests")
        assert hasattr(get_cart_module, "ShopifyExceptionPrompt")
        assert hasattr(get_cart_module, "authorify_storefront")
        assert hasattr(get_cart_module, "PAGEINFO_OUTPUTS")
        assert hasattr(get_cart_module, "cursorify")
        assert hasattr(get_cart_module, "ShopifyGetCartSlots")
        assert hasattr(get_cart_module, "ShopifyOutputs")
        assert hasattr(get_cart_module, "register_tool")
        assert hasattr(get_cart_module, "ToolExecutionError")
        assert hasattr(get_cart_module, "LogContext")

    def test_get_cart_constants(self) -> None:
        """Test that the module constants are properly defined."""
        import arklex.env.tools.shopify.get_cart as get_cart_module

        # Verify constants are defined
        assert hasattr(get_cart_module, "description")
        assert hasattr(get_cart_module, "slots")
        assert hasattr(get_cart_module, "outputs")
        assert hasattr(get_cart_module, "GetCartParams")

        # Verify constant values
        assert "Get cart information" in get_cart_module.description
        assert isinstance(get_cart_module.slots, list)
        assert isinstance(get_cart_module.outputs, list)
