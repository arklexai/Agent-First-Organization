"""Tests for arklex.env.tools.shopify.get_web_product module."""

import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.tools.shopify.get_web_product import get_web_product
from arklex.utils.exceptions import ToolExecutionError


class TestGetWebProduct:
    """Test cases for get_web_product function."""

    def setup_method(self) -> None:
        """Set up test environment."""
        os.environ["ARKLEX_TEST_ENV"] = "local"

    def teardown_method(self) -> None:
        """Clean up test environment."""
        if "ARKLEX_TEST_ENV" in os.environ:
            del os.environ["ARKLEX_TEST_ENV"]

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_success(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test successful web product retrieval."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "products": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Product/12345",
                            "title": "Test Product",
                            "description": "A test product description",
                            "totalInventory": 100,
                            "onlineStoreUrl": "https://test-shop.myshopify.com/products/test-product",
                            "options": [
                                {"name": "Size", "values": ["Small", "Medium", "Large"]}
                            ],
                            "category": {"name": "Clothing"},
                            "variants": {
                                "nodes": [
                                    {
                                        "displayName": "Small",
                                        "id": "gid://shopify/ProductVariant/67890",
                                        "price": "29.99",
                                        "inventoryQuantity": 30,
                                    },
                                    {
                                        "displayName": "Medium",
                                        "id": "gid://shopify/ProductVariant/67891",
                                        "price": "29.99",
                                        "inventoryQuantity": 40,
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

        # Execute function
        result = get_web_product().func(
            web_product_id="gid://shopify/Product/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result
        assert "Product ID: gid://shopify/Product/12345" in result
        assert "Title: Test Product" in result
        assert "Description: A test product description" in result
        assert "Total Inventory: 100" in result
        assert "Options:" in result
        assert "Category: Clothing" in result
        assert "Variant name: Small" in result
        assert "Variant ID: gid://shopify/ProductVariant/67890" in result
        assert "Price: 29.99" in result
        assert "Inventory Quantity: 30" in result

        # Verify GraphQL query was called with correct parameters
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "products" in call_args
        assert "id:12345" in call_args  # Should extract numeric part from full ID

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_no_products_found(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval when no products are found."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "products": {
                    "nodes": [],
                    "pageInfo": {
                        "endCursor": None,
                        "hasNextPage": False,
                        "hasPreviousPage": False,
                        "startCursor": None,
                    },
                }
            }
        }
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            get_web_product().func(
                web_product_id="gid://shopify/Product/99999",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_web_product execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_graphql_exception(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval when GraphQL execution fails."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = Exception("GraphQL API Error")
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            get_web_product().func(
                web_product_id="gid://shopify/Product/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_web_product execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_session_exception(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval when session creation fails."""
        # Setup mocks to raise exception during session creation
        mock_session_temp.side_effect = Exception("Session creation failed")

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            get_web_product().func(
                web_product_id="gid://shopify/Product/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_web_product execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_with_pagination_parameters(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval with pagination parameters."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "products": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Product/12345",
                            "title": "Test Product",
                            "description": "A test product",
                            "totalInventory": 100,
                            "onlineStoreUrl": "https://test-shop.myshopify.com/products/test-product",
                            "options": [],
                            "category": {"name": "Clothing"},
                            "variants": {
                                "nodes": [
                                    {
                                        "displayName": "Default",
                                        "id": "gid://shopify/ProductVariant/67890",
                                        "price": "29.99",
                                        "inventoryQuantity": 100,
                                    }
                                ]
                            },
                        }
                    ],
                    "pageInfo": {
                        "endCursor": "cursor1",
                        "hasNextPage": True,
                        "hasPreviousPage": False,
                        "startCursor": "cursor1",
                    },
                }
            }
        }
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function with pagination parameters
        result = get_web_product().func(
            web_product_id="gid://shopify/Product/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
            limit="10",
            navigate="next",
            pageInfo='{"endCursor": "cursor1", "hasNextPage": true}',
        )

        # Verify result
        assert "Product ID: gid://shopify/Product/12345" in result
        assert "Title: Test Product" in result

        # Verify GraphQL query contains pagination parameters
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "first: 10" in call_args
        assert 'after: "cursor1"' in call_args

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_with_navigation_return_early(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval when navigation returns early."""
        # This test simulates the case where cursorify returns early
        with patch(
            "arklex.env.tools.shopify.get_web_product.cursorify"
        ) as mock_cursorify:
            mock_cursorify.return_value = ("first: 10", False)

            # Execute function
            result = get_web_product().func(
                web_product_id="gid://shopify/Product/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
                limit="10",
            )

            # Verify result is the navigation string
            assert result == "first: 10"

            # Verify no GraphQL execution was made
            mock_graphql_instance = MagicMock()
            mock_graphql.return_value = mock_graphql_instance
            mock_graphql_instance.execute.assert_not_called()

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_with_missing_fields(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval when product has missing fields."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "products": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Product/12345",
                            "title": "Test Product",
                            # Missing description, totalInventory, options, category
                            "variants": {
                                "nodes": [
                                    {
                                        "displayName": "Default",
                                        "id": "gid://shopify/ProductVariant/67890",
                                        "price": "29.99",
                                        "inventoryQuantity": 100,
                                    }
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

        # Execute function
        result = get_web_product().func(
            web_product_id="gid://shopify/Product/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result handles missing fields gracefully
        assert "Product ID: gid://shopify/Product/12345" in result
        assert "Title: Test Product" in result
        assert "Description: None" in result
        assert "Total Inventory: None" in result
        assert "Options: None" in result
        assert "Category: None" in result

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_with_empty_variants(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval when product has no variants."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "products": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Product/12345",
                            "title": "Test Product",
                            "description": "A test product",
                            "totalInventory": 100,
                            "options": [],
                            "category": {"name": "Clothing"},
                            "variants": {"nodes": []},
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

        # Execute function
        result = get_web_product().func(
            web_product_id="gid://shopify/Product/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result handles empty variants gracefully
        assert "Product ID: gid://shopify/Product/12345" in result
        assert "Title: Test Product" in result
        assert "The following are several variants of the product:" in result

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_with_missing_variants_key(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval when product is missing variants key."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "products": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Product/12345",
                            "title": "Test Product",
                            "description": "A test product",
                            "totalInventory": 100,
                            "options": [],
                            "category": {"name": "Clothing"},
                            # Missing variants key
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

        # Execute function
        result = get_web_product().func(
            web_product_id="gid://shopify/Product/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result handles missing variants key gracefully
        assert "Product ID: gid://shopify/Product/12345" in result
        assert "Title: Test Product" in result
        assert "The following are several variants of the product:" in result

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_with_numeric_id(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval with numeric product ID."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "products": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Product/12345",
                            "title": "Test Product",
                            "description": "A test product",
                            "totalInventory": 100,
                            "options": [],
                            "category": {"name": "Clothing"},
                            "variants": {
                                "nodes": [
                                    {
                                        "displayName": "Default",
                                        "id": "gid://shopify/ProductVariant/67890",
                                        "price": "29.99",
                                        "inventoryQuantity": 100,
                                    }
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

        # Execute function with numeric ID
        result = get_web_product().func(
            web_product_id="12345",  # Numeric ID instead of full GID
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result
        assert "Product ID: gid://shopify/Product/12345" in result
        assert "Title: Test Product" in result

        # Verify GraphQL query was called with the numeric ID
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "id:12345" in call_args

    def test_get_web_product_function_registration(self) -> None:
        """Test that the get_web_product function is properly registered as a tool."""
        # Create a tool instance to access the attributes
        tool_instance = get_web_product()

        # Verify the function has the expected attributes from register_tool decorator
        assert hasattr(tool_instance, "func")
        assert hasattr(tool_instance, "description")
        assert hasattr(tool_instance, "slots")
        assert hasattr(tool_instance, "output")

        # Verify the description matches expected value
        assert (
            "inventory information and description details of a product"
            in tool_instance.description
        )

        # Verify the function signature
        import inspect

        sig = inspect.signature(tool_instance.func)
        assert "web_product_id" in sig.parameters
        assert "kwargs" in sig.parameters

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_with_json_decode_error(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval when JSON response is malformed."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = "invalid json"
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            get_web_product().func(
                web_product_id="gid://shopify/Product/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_web_product execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_with_missing_data_key(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval when response is missing 'data' key."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"products": {"nodes": []}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            get_web_product().func(
                web_product_id="gid://shopify/Product/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_web_product execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_with_missing_products_key(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval when response is missing 'products' key."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"data": {"someOtherKey": {"nodes": []}}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            get_web_product().func(
                web_product_id="gid://shopify/Product/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_web_product execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_with_missing_nodes_key(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test web product retrieval when response is missing 'nodes' key."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"data": {"products": {"someOtherKey": []}}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            get_web_product().func(
                web_product_id="gid://shopify/Product/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_web_product execution failed" in str(exc_info.value)
