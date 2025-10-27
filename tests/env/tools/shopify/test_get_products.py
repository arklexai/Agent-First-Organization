"""Tests for arklex.resources.tools.shopify.get_products module."""

import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.resources.tools.shopify.get_products import get_products
from arklex.utils.logging.exceptions import ToolExecutionError


class TestGetProducts:
    """Test cases for get_products function."""

    def setup_method(self) -> None:
        """Set up test environment."""
        os.environ["ARKLEX_TEST_ENV"] = "local"

    def get_original_function(self) -> callable:
        """Get the original function from the decorated function."""
        # Access the original function from the Tool instance
        return get_products.func

    def teardown_method(self) -> None:
        """Clean up test environment."""
        if "ARKLEX_TEST_ENV" in os.environ:
            del os.environ["ARKLEX_TEST_ENV"]

    @patch("arklex.resources.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_success(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test successful product retrieval."""
        # Setup mocks
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
                            "description": "Test Description",
                            "totalInventory": 10,
                            "options": [{"name": "Size", "values": ["S", "M", "L"]}],
                            "variants": {
                                "nodes": [
                                    {
                                        "id": "gid://shopify/ProductVariant/456",
                                        "title": "S",
                                        "price": "19.99",
                                        "inventoryQuantity": 5,
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
        }
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function
        result = self.get_original_function()(
            product_ids=["gid://shopify/Product/123"],
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "admin_token": "test_admin_token",
                "api_version": "2024-10",
            },
        )

        # Verify result
        assert "Test Product" in result.message_flow
        assert "Test Description" in result.message_flow

    @patch("arklex.resources.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_no_products_found(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test product retrieval when no products are found."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"data": {"products": {"nodes": []}}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                product_ids=["gid://shopify/Product/99999"],
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "admin_token": "test_admin_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_products execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_graphql_exception(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test product retrieval when GraphQL execution fails."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = Exception("GraphQL API Error")
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                product_ids=["gid://shopify/Product/12345"],
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "admin_token": "test_admin_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_products execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_session_exception(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test product retrieval when session creation fails."""
        # Setup mocks to raise exception during session creation
        mock_session_temp.side_effect = Exception("Session creation failed")

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                product_ids=["gid://shopify/Product/12345"],
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "admin_token": "test_admin_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_products execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_with_multiple_ids(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test product retrieval with multiple product IDs."""
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
                            "title": "Product 1",
                            "description": "First product",
                            "totalInventory": 50,
                            "options": [],
                            "variants": {
                                "nodes": [
                                    {
                                        "displayName": "Default",
                                        "id": "gid://shopify/ProductVariant/67890",
                                        "price": "19.99",
                                        "inventoryQuantity": 50,
                                    }
                                ]
                            },
                        },
                        {
                            "id": "gid://shopify/Product/12346",
                            "title": "Product 2",
                            "description": "Second product",
                            "totalInventory": 75,
                            "options": [],
                            "variants": {
                                "nodes": [
                                    {
                                        "displayName": "Default",
                                        "id": "gid://shopify/ProductVariant/67891",
                                        "price": "29.99",
                                        "inventoryQuantity": 75,
                                    }
                                ]
                            },
                        },
                    ],
                    "pageInfo": {
                        "endCursor": "cursor2",
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
        result = self.get_original_function()(
            product_ids=["gid://shopify/Product/12345", "gid://shopify/Product/12346"],
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "admin_token": "test_admin_token",
                "api_version": "2024-10",
            },
        )

        # Verify result contains both products
        assert "Product ID: gid://shopify/Product/12345" in result.message_flow
        assert "Title: Product 1" in result.message_flow
        assert "Product ID: gid://shopify/Product/12346" in result.message_flow
        assert "Title: Product 2" in result.message_flow

        # Verify GraphQL query contains OR condition
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "id:12345 OR id:12346" in call_args

    @patch("arklex.resources.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_with_pagination_parameters(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test product retrieval with pagination parameters."""
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
        result = self.get_original_function()(
            product_ids=["gid://shopify/Product/12345"],
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "admin_token": "test_admin_token",
                "api_version": "2024-10",
            },
            limit="10",
            navigate="next",
            pageInfo='{"endCursor": "cursor1", "hasNextPage": true}',
        )

        # Verify result
        assert "Product ID: gid://shopify/Product/12345" in result.message_flow
        assert "Title: Test Product" in result.message_flow

        # Verify GraphQL query contains pagination parameters
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "first: 10" in call_args
        assert 'after: "cursor1"' in call_args

    @patch("arklex.resources.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_with_missing_fields(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test product retrieval when products have missing fields."""
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
        result = self.get_original_function()(
            product_ids=["gid://shopify/Product/12345"],
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "admin_token": "test_admin_token",
                "api_version": "2024-10",
            },
        )

        # Verify result handles missing fields gracefully
        assert "Product ID: gid://shopify/Product/12345" in result.message_flow
        assert "Title: Test Product" in result.message_flow
        assert "Description: None" in result.message_flow
        assert "Total Inventory: None" in result.message_flow
        assert "Options: None" in result.message_flow

    @patch("arklex.resources.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_with_empty_variants(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test product retrieval when products have no variants."""
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
        result = self.get_original_function()(
            product_ids=["gid://shopify/Product/12345"],
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "admin_token": "test_admin_token",
                "api_version": "2024-10",
            },
        )

        # Verify result handles empty variants gracefully
        assert "Product ID: gid://shopify/Product/12345" in result.message_flow
        assert "Title: Test Product" in result.message_flow
        assert (
            "The following are several variants of the product:" in result.message_flow
        )

    @patch("arklex.resources.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_with_missing_variants_key(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test product retrieval when products are missing variants key."""
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
        result = self.get_original_function()(
            product_ids=["gid://shopify/Product/12345"],
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "admin_token": "test_admin_token",
                "api_version": "2024-10",
            },
        )

        # Verify result handles missing variants key gracefully
        assert "Product ID: gid://shopify/Product/12345" in result.message_flow
        assert "Title: Test Product" in result.message_flow
        assert (
            "The following are several variants of the product:" in result.message_flow
        )

    def test_get_products_function_registration(self) -> None:
        """Test that the get_products function is properly registered as a tool."""
        # Verify the function returns a Tool instance when called
        tool_instance = get_products
        from arklex.resources.tools.tools import Tool

        assert isinstance(tool_instance, Tool)

        # Verify the tool has the expected attributes
        assert hasattr(tool_instance, "description")
        assert hasattr(tool_instance, "slots")

        # Verify the description matches expected value
        assert (
            "inventory information and description details" in tool_instance.description
        )

    @patch("arklex.resources.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_with_json_decode_error(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test product retrieval when JSON response is malformed."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = "invalid json"
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                product_ids=["gid://shopify/Product/12345"],
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "admin_token": "test_admin_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_products execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_with_missing_data_key(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test product retrieval when response is missing 'data' key."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"products": {"nodes": []}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                product_ids=["gid://shopify/Product/12345"],
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "admin_token": "test_admin_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_products execution failed" in str(exc_info.value)
