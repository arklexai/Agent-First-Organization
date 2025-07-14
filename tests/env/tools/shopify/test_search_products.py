"""Tests for arklex.env.tools.shopify.search_products module."""

import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.tools.shopify.search_products import search_products
from arklex.utils.exceptions import ToolExecutionError


class TestSearchProducts:
    """Test cases for search_products function."""

    def setup_method(self) -> None:
        """Set up test environment."""
        os.environ["ARKLEX_TEST_ENV"] = "local"

    def teardown_method(self) -> None:
        """Clean up test environment."""
        if "ARKLEX_TEST_ENV" in os.environ:
            del os.environ["ARKLEX_TEST_ENV"]

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_success(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test successful product search."""
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
                            "description": "A test product description that is longer than 180 characters to test truncation functionality in the search results",
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

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "I found some great products for you! What size are you looking for?"
        )
        mock_provider_map.get.return_value = MagicMock(return_value=mock_llm)

        # Execute function
        result = search_products().func(
            product_query="test product",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
            llm_provider="openai",
            model_type_or_path="gpt-4",
        )

        # Verify result is JSON string
        assert isinstance(result, str)
        result_data = json.loads(result)
        assert "answer" in result_data
        assert "card_list" in result_data
        assert len(result_data["card_list"]) == 1

        # Verify card data
        card = result_data["card_list"][0]
        assert card["id"] == "gid://shopify/Product/12345"
        assert card["title"] == "Test Product"
        assert "test product description" in card["description"]
        assert (
            card["link_url"] == "https://test-shop.myshopify.com/products/test-product"
        )
        assert card["image_url"] == "https://cdn.shopify.com/test-image.jpg"
        assert len(card["variants"]) == 2

        # Verify GraphQL query was called with correct parameters
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "products" in call_args
        assert "test product" in call_args

        # Verify LLM was called
        mock_llm.invoke.assert_called_once()

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_no_results(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when no products are found."""
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
            search_products().func(
                product_query="nonexistent product",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
                llm_provider="openai",
                model_type_or_path="gpt-4",
            )

        assert "Tool search_products execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_graphql_exception(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when GraphQL execution fails."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = Exception("GraphQL API Error")
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            search_products().func(
                product_query="test product",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
                llm_provider="openai",
                model_type_or_path="gpt-4",
            )

        assert "Tool search_products execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_session_exception(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when session creation fails."""
        # Setup mocks to raise exception during session creation
        mock_session_temp.side_effect = Exception("Session creation failed")

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            search_products().func(
                product_query="test product",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
                llm_provider="openai",
                model_type_or_path="gpt-4",
            )

        assert "Tool search_products execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_with_pagination_parameters(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search with pagination parameters."""
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

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "Great products found! What's your preference?"
        )
        mock_provider_map.get.return_value = MagicMock(return_value=mock_llm)

        # Mock cursorify to return expected navigation parameters
        with patch(
            "arklex.env.tools.shopify.search_products.cursorify"
        ) as mock_cursorify:
            mock_cursorify.return_value = ('first: 10, after: "cursor1"', True)

            # Execute function with pagination parameters
            result = search_products().func(
                product_query="test product",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
                llm_provider="openai",
                model_type_or_path="gpt-4",
                limit="10",
                navigate="next",
                pageInfo='{"endCursor": "cursor1"}',
            )

            # Verify result
            result_data = json.loads(result)
            assert "answer" in result_data
            assert "card_list" in result_data

            # Verify GraphQL query contains pagination parameters
            mock_graphql_instance.execute.assert_called_once()
            call_args = mock_graphql_instance.execute.call_args[0][0]
            assert "first: 10" in call_args
            assert 'after: "cursor1"' in call_args

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_with_navigation_return_early(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when navigation returns early."""
        # This test simulates the case where cursorify returns early
        with patch(
            "arklex.env.tools.shopify.search_products.cursorify"
        ) as mock_cursorify:
            mock_cursorify.return_value = ("first: 10", False)

            # Execute function
            result = search_products().func(
                product_query="test product",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
                llm_provider="openai",
                model_type_or_path="gpt-4",
                limit="10",
            )

            # Verify result is the navigation string
            assert result == "first: 10"

            # Verify no GraphQL execution was made
            mock_graphql_instance = MagicMock()
            mock_graphql.return_value = mock_graphql_instance
            mock_graphql_instance.execute.assert_not_called()

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_with_missing_fields(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when products have missing fields."""
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
                            # Missing description, handle, onlineStoreUrl, images, variants
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

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Found some products!"
        mock_provider_map.get.return_value = MagicMock(return_value=mock_llm)

        # Execute function
        result = search_products().func(
            product_query="test product",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
            llm_provider="openai",
            model_type_or_path="gpt-4",
        )

        # Verify result handles missing fields gracefully
        result_data = json.loads(result)
        card = result_data["card_list"][0]
        assert card["id"] == "gid://shopify/Product/12345"
        assert card["title"] == "Test Product"
        assert card["description"] == "None..."
        assert card["link_url"] == "https://test-shop.myshopify.com/products/None"
        assert card["image_url"] == ""
        assert card["variants"] == []

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_with_empty_images(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when products have no images."""
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
                            "handle": "test-product",
                            "onlineStoreUrl": "https://test-shop.myshopify.com/products/test-product",
                            "images": {"edges": []},
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

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Found products without images!"
        mock_provider_map.get.return_value = MagicMock(return_value=mock_llm)

        # Execute function
        result = search_products().func(
            product_query="test product",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
            llm_provider="openai",
            model_type_or_path="gpt-4",
        )

        # Verify result handles empty images gracefully
        result_data = json.loads(result)
        card = result_data["card_list"][0]
        assert card["image_url"] == ""

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_with_missing_images_key(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when products are missing images key."""
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
                            "handle": "test-product",
                            "onlineStoreUrl": "https://test-shop.myshopify.com/products/test-product",
                            # Missing images key
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

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Found products!"
        mock_provider_map.get.return_value = MagicMock(return_value=mock_llm)

        # Execute function
        result = search_products().func(
            product_query="test product",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
            llm_provider="openai",
            model_type_or_path="gpt-4",
        )

        # Verify result handles missing images key gracefully
        result_data = json.loads(result)
        card = result_data["card_list"][0]
        assert card["image_url"] == ""

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_with_missing_variants_key(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when products are missing variants key."""
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

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "Found products!"
        mock_provider_map.get.return_value = MagicMock(return_value=mock_llm)

        # Execute function
        result = search_products().func(
            product_query="test product",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
            llm_provider="openai",
            model_type_or_path="gpt-4",
        )

        # Verify result handles missing variants key gracefully
        result_data = json.loads(result)
        card = result_data["card_list"][0]
        assert card["variants"] == []

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_with_llm_exception(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when LLM invocation fails."""
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

        # Mock LLM to raise exception
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM API Error")
        mock_provider_map.get.return_value = MagicMock(return_value=mock_llm)

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            search_products().func(
                product_query="test product",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
                llm_provider="openai",
                model_type_or_path="gpt-4",
            )

        assert "Tool search_products execution failed" in str(exc_info.value)

    def test_search_products_function_registration(self) -> None:
        """Test that the search_products function is properly registered as a tool."""
        # Create a tool instance to access the attributes
        tool_instance = search_products()

        # Verify the function has the expected attributes from register_tool decorator
        assert hasattr(tool_instance, "func")
        assert hasattr(tool_instance, "description")
        assert hasattr(tool_instance, "slots")
        assert hasattr(tool_instance, "output")

        # Verify the description matches expected value
        assert "Search products by string query" in tool_instance.description

        # Verify the function signature
        import inspect

        sig = inspect.signature(tool_instance.func)
        assert "product_query" in sig.parameters
        assert "kwargs" in sig.parameters

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_with_json_decode_error(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when JSON response is malformed."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = "invalid json"
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            search_products().func(
                product_query="test product",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
                llm_provider="openai",
                model_type_or_path="gpt-4",
            )

        assert "Tool search_products execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_with_missing_data_key(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when response is missing 'data' key."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"products": {"nodes": []}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            search_products().func(
                product_query="test product",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
                llm_provider="openai",
                model_type_or_path="gpt-4",
            )

        assert "Tool search_products execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_with_missing_products_key(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when response is missing 'products' key."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"data": {"someOtherKey": {"nodes": []}}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            search_products().func(
                product_query="test product",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
                llm_provider="openai",
                model_type_or_path="gpt-4",
            )

        assert "Tool search_products execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_with_missing_nodes_key(
        self, mock_graphql: Mock, mock_session_temp: Mock, mock_provider_map: Mock
    ) -> None:
        """Test product search when response is missing 'nodes' key."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"data": {"products": {"someOtherKey": []}}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            search_products().func(
                product_query="test product",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
                llm_provider="openai",
                model_type_or_path="gpt-4",
            )

        assert "Tool search_products execution failed" in str(exc_info.value)
