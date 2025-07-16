"""
Integration tests for Shopify tools.

This module contains comprehensive integration tests for all Shopify tools,
including proper mocking of external services and edge case testing.
These tests validate the complete Shopify integration workflow from API
calls to response processing and error handling.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.tools.shopify.cancel_order import cancel_order
from arklex.env.tools.shopify.cart_add_items import cart_add_items
from arklex.env.tools.shopify.get_cart import get_cart
from arklex.env.tools.shopify.get_order_details import get_order_details
from arklex.env.tools.shopify.get_products import get_products
from arklex.env.tools.shopify.get_user_details_admin import get_user_details_admin
from arklex.env.tools.shopify.get_web_product import get_web_product
from arklex.env.tools.shopify.return_products import return_products
from arklex.env.tools.shopify.search_products import search_products
from arklex.utils.exceptions import ShopifyError, ToolExecutionError

# Extract underlying functions from decorated tool functions for direct testing
# This allows us to test the core functionality without the decorator wrapper
search_products_func = search_products().func
get_user_details_admin_func = get_user_details_admin().func
get_products_func = get_products().func
get_order_details_func = get_order_details().func
get_cart_func = get_cart().func
cart_add_items_func = cart_add_items().func
return_products_func = return_products().func
cancel_order_func = cancel_order().func
get_web_product_func = get_web_product().func


class TestShopifySearchProducts:
    """
    Integration tests for search_products tool.

    This test class validates the product search functionality, including
    successful searches, empty results, API errors, and edge cases.
    """

    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.PROVIDER_MAP")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_success(
        self,
        mock_graphql: Mock,
        mock_provider_map: Mock,
        mock_session_temp: Mock,
        sample_shopify_product_data: dict,
    ) -> None:
        """
        Test successful product search with LLM response generation.

        This test validates that the search_products tool can successfully
        query the Shopify GraphQL API, process the results, and generate
        a natural language response using the configured LLM provider.
        """
        # Mock Shopify session for API authentication
        # This simulates the Shopify session management
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock GraphQL response with comprehensive product data
        # This simulates a successful API response with product information
        mock_response = sample_shopify_product_data

        # Set up GraphQL client mock to return the test data
        # This simulates the GraphQL API client behavior
        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Mock LLM response for natural language processing
        # This simulates the LLM provider generating a human-readable response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "I found some great products for you! What size are you looking for?"
        )
        mock_provider_map.get.return_value.return_value = mock_llm

        # Execute the search_products function with test parameters
        # This tests the complete workflow from input to output
        result = search_products_func(
            product_query="test product",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
            llm_provider="openai",
            model_type_or_path="gpt-3.5-turbo",
        )

        # Parse the JSON result and validate the structure
        # This ensures the response format is correct
        result_data = json.loads(result)
        assert "answer" in result_data, "Response should contain 'answer' field"
        assert "card_list" in result_data, "Response should contain 'card_list' field"
        assert len(result_data["card_list"]) == 1, "Should return exactly one product"
        assert result_data["card_list"][0]["title"] == "Test Product", (
            "Product title should match"
        )

        # Verify that the LLM-generated answer contains expected content
        # This ensures the natural language processing worked correctly
        assert "found" in result_data["answer"].lower(), (
            "LLM response should mention finding products"
        )

    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_no_results(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """
        Test product search when no products are found.

        This test validates that the search_products tool properly handles
        cases where the search query returns no matching products and
        raises an appropriate error with a meaningful message.
        """
        # Mock Shopify session for API authentication
        # This simulates the Shopify session management
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock empty search results with proper pagination info
        # This simulates a valid API response with no matching products
        mock_response = {
            "data": {
                "products": {
                    "nodes": [],  # Empty product list
                    "pageInfo": {
                        "endCursor": None,
                        "hasNextPage": False,
                        "hasPreviousPage": False,
                        "startCursor": None,
                    },
                }
            }
        }

        # Set up GraphQL client mock to return empty results
        # This simulates the GraphQL API returning no products
        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Test that the function raises an appropriate error
        # This validates the error handling for no results
        with pytest.raises(ToolExecutionError) as exc_info:
            search_products_func(
                product_query="nonexistent product",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        # Verify error message contains expected pattern
        # This ensures the error message is informative
        assert "Tool search_products execution failed" in str(exc_info.value), (
            "Error message should indicate tool execution failure"
        )

    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_api_exception(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """
        Test product search when Shopify API throws an exception.

        This test validates that the search_products tool properly handles
        API errors and exceptions, ensuring that failures are caught and
        appropriate error messages are provided to the user.
        """
        # Mock Shopify session for API authentication
        # This simulates the Shopify session management
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Simulate API error by making execute method raise an exception
        # This simulates a network error, authentication failure, or other API issue
        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = Exception("API Error")
        mock_graphql.return_value = mock_graphql_instance

        # Test that the function raises an appropriate error
        # This validates the error handling for API exceptions
        with pytest.raises(ToolExecutionError) as exc_info:
            search_products_func(
                product_query="test product",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        # Verify error message contains expected pattern
        # This ensures the error message is informative
        assert "Tool search_products execution failed" in str(exc_info.value), (
            "Error message should indicate tool execution failure"
        )


class TestShopifyGetUserDetailsAdmin:
    """
    Integration tests for get_user_details_admin tool.

    This test class validates the user details retrieval functionality,
    including successful retrievals, user not found scenarios, and API errors.
    """

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_success(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """
        Test successful user details retrieval with comprehensive customer data.

        This test validates that the get_user_details_admin tool can successfully
        retrieve comprehensive customer information from Shopify, including
        personal details, order history, and account metadata.
        """
        # Mock Shopify session for API authentication
        # This simulates the Shopify session management
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock comprehensive GraphQL response with all customer fields
        # This simulates a complete customer profile with all available data
        mock_response = {
            "data": {
                "customer": {
                    "firstName": "John",
                    "lastName": "Doe",
                    "email": "john.doe@example.com",
                    "phone": "+1234567890",
                    "numberOfOrders": 5,
                    "amountSpent": {"amount": "299.95", "currencyCode": "USD"},
                    "createdAt": "2023-01-15T10:00:00Z",
                    "updatedAt": "2024-01-15T10:00:00Z",
                    "note": "VIP customer",
                    "verifiedEmail": True,
                    "validEmailAddress": True,
                    "tags": ["vip", "returning"],
                    "lifetimeDuration": 365,
                    "addresses": [{"address1": "123 Main St, City, State 12345"}],
                    "orders": {
                        "nodes": [
                            {"id": "gid://shopify/Order/12345"},
                            {"id": "gid://shopify/Order/12346"},
                        ]
                    },
                }
            }
        }

        # Set up GraphQL client mock to return the customer data
        # This simulates the GraphQL API returning customer information
        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute the get_user_details_admin function with test parameters
        # This tests the complete workflow from customer ID to detailed profile
        result = get_user_details_admin_func(
            user_id="gid://shopify/Customer/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        user_data = json.loads(result)
        # Verify all key customer fields are present and correct
        assert user_data["firstName"] == "John"
        assert user_data["lastName"] == "Doe"
        assert user_data["email"] == "john.doe@example.com"
        assert user_data["numberOfOrders"] == 5
        assert user_data["amountSpent"]["amount"] == "299.95"

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_user_not_found(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test user details retrieval when user is not found."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock response indicating no customer found
        mock_response = {"data": {"customer": None}}

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        with pytest.raises(ToolExecutionError) as exc_info:
            get_user_details_admin_func(
                user_id="gid://shopify/Customer/99999",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )
        # Verify error message contains expected pattern
        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_api_exception(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test user details retrieval when Shopify API throws an exception."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Simulate API error by making execute method raise an exception
        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = Exception("API Error")
        mock_graphql.return_value = mock_graphql_instance

        with pytest.raises(ToolExecutionError) as exc_info:
            get_user_details_admin_func(
                user_id="gid://shopify/Customer/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )
        # Verify error message contains expected pattern
        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)


class TestShopifyGetProducts:
    """Integration tests for get_products tool."""

    @patch("arklex.env.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_success(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test successful product details retrieval with complete product information."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock comprehensive GraphQL response with product details
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
                                        "displayName": "Small",
                                        "id": "gid://shopify/ProductVariant/67890",
                                        "price": "29.99",
                                        "inventoryQuantity": 10,
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = get_products_func(
            product_ids=["gid://shopify/Product/12345"],
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify product information is included in the result
        assert "Test Product" in result
        assert "gid://shopify/Product/12345" in result

    @patch("arklex.env.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_no_products_found(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test product details retrieval when no products are found."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock empty products response
        mock_response = {"data": {"products": {"nodes": []}}}

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        with pytest.raises(ToolExecutionError) as exc_info:
            get_products_func(
                product_ids=["gid://shopify/Product/99999"],
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )
        # Verify error message contains expected pattern
        assert "Tool get_products execution failed" in str(exc_info.value)


class TestShopifyGetOrderDetails:
    """Integration tests for get_order_details tool."""

    @patch("arklex.env.tools.shopify.get_order_details.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_order_details.shopify.GraphQL")
    def test_get_order_details_success(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test successful order details retrieval with complete order information."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock comprehensive GraphQL response with order details
        mock_response = {
            "data": {
                "orders": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Order/12345",
                            "name": "#1001",
                            "createdAt": "2024-01-15T10:00:00Z",
                            "cancelledAt": None,
                            "returnStatus": "RETURNED",
                            "statusPageUrl": "https://test-shop.myshopify.com/orders/1001",
                            "totalPriceSet": {"presentmentMoney": {"amount": "99.99"}},
                            "fulfillments": [
                                {
                                    "displayStatus": "FULFILLED",
                                    "trackingInfo": [
                                        {
                                            "number": "TRACK123",
                                            "url": "https://tracking.example.com/TRACK123",
                                        }
                                    ],
                                }
                            ],
                            "lineItems": {
                                "edges": [
                                    {
                                        "node": {
                                            "id": "gid://shopify/LineItem/67890",
                                            "title": "Test Product",
                                            "quantity": 2,
                                            "variant": {
                                                "id": "gid://shopify/ProductVariant/11111",
                                                "product": {
                                                    "id": "gid://shopify/Product/22222"
                                                },
                                            },
                                        }
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = get_order_details_func(
            order_ids=["gid://shopify/Order/12345"],
            order_names=[],
            user_id="gid://shopify/Customer/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify order information is included in the result
        assert "#1001" in result
        assert "gid://shopify/Order/12345" in result
        assert "Created At: 2024-01-15T10:00:00Z" in result
        assert "Return Status: RETURNED" in result
        assert "Total Price: 99.99" in result
        assert "Fulfillment Status: [{'displayStatus': 'FULFILLED'" in result
        assert "Title: Test Product" in result
        assert "Quantity: 2" in result

    @patch("arklex.env.tools.shopify.get_order_details.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_order_details.shopify.GraphQL")
    def test_get_order_details_no_orders_found(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test order details retrieval when no orders are found."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock empty orders response
        mock_response = {"data": {"orders": {"nodes": []}}}

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # This test should not raise an exception, it should return a message
        result = get_order_details_func(
            order_ids=["gid://shopify/Order/99999"],
            order_names=[],
            user_id="gid://shopify/Customer/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )
        # Verify appropriate message is returned when no orders exist
        assert "You have no orders placed" in result

    @patch("arklex.env.tools.shopify.get_order_details.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_order_details.shopify.GraphQL")
    def test_get_order_details_with_order_names_filter(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test order details retrieval with order names filter."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_response = {
            "data": {
                "orders": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Order/12345",
                            "name": "#1001",
                            "createdAt": "2024-01-15T10:00:00Z",
                            "cancelledAt": None,
                            "returnStatus": "RETURNED",
                            "statusPageUrl": "https://test-shop.myshopify.com/orders/1001",
                            "totalPriceSet": {"presentmentMoney": {"amount": "99.99"}},
                            "fulfillments": [],
                            "lineItems": {"edges": []},
                        }
                    ]
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = get_order_details_func(
            order_ids=[],
            order_names=["#1001", "#1002"],
            user_id="gid://shopify/Customer/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify the GraphQL query was called with order names filter
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "name:#1001" in call_args
        assert "name:#1002" in call_args
        assert "#1001" in result

    @patch("arklex.env.tools.shopify.get_order_details.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_order_details.shopify.GraphQL")
    def test_get_order_details_with_limit_parameter(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test order details retrieval with custom limit parameter."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_response = {
            "data": {
                "orders": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Order/12345",
                            "name": "#1001",
                            "createdAt": "2024-01-15T10:00:00Z",
                            "cancelledAt": None,
                            "returnStatus": "RETURNED",
                            "statusPageUrl": "https://test-shop.myshopify.com/orders/1001",
                            "totalPriceSet": {"presentmentMoney": {"amount": "99.99"}},
                            "fulfillments": [],
                            "lineItems": {"edges": []},
                        }
                    ]
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = get_order_details_func(
            order_ids=[],
            order_names=[],
            user_id="gid://shopify/Customer/12345",
            limit=5,
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify the GraphQL query was called with custom limit
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "first: 5" in call_args
        assert "#1001" in result

    @patch("arklex.env.tools.shopify.get_order_details.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_order_details.shopify.GraphQL")
    def test_get_order_details_with_cancelled_order(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test order details retrieval for a cancelled order."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_response = {
            "data": {
                "orders": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Order/12345",
                            "name": "#1001",
                            "createdAt": "2024-01-15T10:00:00Z",
                            "cancelledAt": "2024-01-16T10:00:00Z",
                            "returnStatus": "RETURNED",
                            "statusPageUrl": "https://test-shop.myshopify.com/orders/1001",
                            "totalPriceSet": {"presentmentMoney": {"amount": "99.99"}},
                            "fulfillments": [],
                            "lineItems": {"edges": []},
                        }
                    ]
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = get_order_details_func(
            order_ids=["gid://shopify/Order/12345"],
            order_names=[],
            user_id="gid://shopify/Customer/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify cancelled order information is included
        assert "Cancelled At: 2024-01-16T10:00:00Z" in result
        assert "#1001" in result

    @patch("arklex.env.tools.shopify.get_order_details.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_order_details.shopify.GraphQL")
    def test_get_order_details_with_multiple_line_items(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test order details retrieval with multiple line items."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_response = {
            "data": {
                "orders": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Order/12345",
                            "name": "#1001",
                            "createdAt": "2024-01-15T10:00:00Z",
                            "cancelledAt": None,
                            "returnStatus": "RETURNED",
                            "statusPageUrl": "https://test-shop.myshopify.com/orders/1001",
                            "totalPriceSet": {"presentmentMoney": {"amount": "199.98"}},
                            "fulfillments": [],
                            "lineItems": {
                                "edges": [
                                    {
                                        "node": {
                                            "id": "gid://shopify/LineItem/67890",
                                            "title": "Product 1",
                                            "quantity": 2,
                                            "variant": {
                                                "id": "gid://shopify/ProductVariant/11111",
                                                "product": {
                                                    "id": "gid://shopify/Product/22222"
                                                },
                                            },
                                        }
                                    },
                                    {
                                        "node": {
                                            "id": "gid://shopify/LineItem/67891",
                                            "title": "Product 2",
                                            "quantity": 1,
                                            "variant": {
                                                "id": "gid://shopify/ProductVariant/11112",
                                                "product": {
                                                    "id": "gid://shopify/Product/22223"
                                                },
                                            },
                                        }
                                    },
                                ]
                            },
                        }
                    ]
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = get_order_details_func(
            order_ids=["gid://shopify/Order/12345"],
            order_names=[],
            user_id="gid://shopify/Customer/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify multiple line items are included
        assert "Title: Product 1" in result
        assert "Title: Product 2" in result
        assert "Quantity: 2" in result
        assert "Quantity: 1" in result
        assert "Total Price: 199.98" in result

    @patch("arklex.env.tools.shopify.get_order_details.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_order_details.shopify.GraphQL")
    def test_get_order_details_api_exception(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test order details retrieval when API throws an exception."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Simulate API error by making execute method raise an exception
        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = Exception("API Error")
        mock_graphql.return_value = mock_graphql_instance

        # Test that the function raises an appropriate error
        with pytest.raises(ToolExecutionError) as exc_info:
            get_order_details_func(
                order_ids=["gid://shopify/Order/12345"],
                order_names=[],
                user_id="gid://shopify/Customer/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        # Verify error message contains expected pattern
        assert "Tool get_order_details execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_order_details.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_order_details.shopify.GraphQL")
    def test_get_order_details_with_string_limit(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test order details retrieval with string limit parameter (should be converted to int)."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_response = {
            "data": {
                "orders": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Order/12345",
                            "name": "#1001",
                            "createdAt": "2024-01-15T10:00:00Z",
                            "cancelledAt": None,
                            "returnStatus": "RETURNED",
                            "statusPageUrl": "https://test-shop.myshopify.com/orders/1001",
                            "totalPriceSet": {"presentmentMoney": {"amount": "99.99"}},
                            "fulfillments": [],
                            "lineItems": {"edges": []},
                        }
                    ]
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = get_order_details_func(
            order_ids=[],
            order_names=[],
            user_id="gid://shopify/Customer/12345",
            limit="3",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify the GraphQL query was called with converted limit
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "first: 3" in call_args
        assert "#1001" in result

    @patch("arklex.env.tools.shopify.get_order_details.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_order_details.shopify.GraphQL")
    def test_get_order_details_with_none_limit(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test order details retrieval with None limit parameter (should use default 10)."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_response = {
            "data": {
                "orders": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Order/12345",
                            "name": "#1001",
                            "createdAt": "2024-01-15T10:00:00Z",
                            "cancelledAt": None,
                            "returnStatus": "RETURNED",
                            "statusPageUrl": "https://test-shop.myshopify.com/orders/1001",
                            "totalPriceSet": {"presentmentMoney": {"amount": "99.99"}},
                            "fulfillments": [],
                            "lineItems": {"edges": []},
                        }
                    ]
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = get_order_details_func(
            order_ids=[],
            order_names=[],
            user_id="gid://shopify/Customer/12345",
            limit=None,
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify the GraphQL query was called with default limit
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "first: 10" in call_args
        assert "#1001" in result

    @patch("arklex.env.tools.shopify.get_order_details.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_order_details.shopify.GraphQL")
    def test_get_order_details_with_complex_user_id(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test order details retrieval with complex user ID that needs parsing."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_response = {
            "data": {
                "orders": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Order/12345",
                            "name": "#1001",
                            "createdAt": "2024-01-15T10:00:00Z",
                            "cancelledAt": None,
                            "returnStatus": "RETURNED",
                            "statusPageUrl": "https://test-shop.myshopify.com/orders/1001",
                            "totalPriceSet": {"presentmentMoney": {"amount": "99.99"}},
                            "fulfillments": [],
                            "lineItems": {"edges": []},
                        }
                    ]
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = get_order_details_func(
            order_ids=[],
            order_names=[],
            user_id="gid://shopify/Customer/67890",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify the GraphQL query was called with parsed user ID
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "customer_id:67890" in call_args
        assert "#1001" in result


class TestShopifyGetCart:
    """Integration tests for get_cart tool."""

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_success(self, mock_post: Mock) -> None:
        """Test successful cart retrieval with items."""
        # Mock successful API response with cart containing items
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "id": "gid://shopify/Cart/12345",
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout/12345",
                    "lines": {
                        "nodes": [
                            {
                                "id": "gid://shopify/CartLine/67890",
                                "quantity": 2,
                                "merchandise": {
                                    "id": "gid://shopify/ProductVariant/11111",
                                    "title": "Small",
                                    "product": {
                                        "id": "gid://shopify/Product/22222",
                                        "title": "Test Product",
                                    },
                                },
                            }
                        ]
                    },
                }
            }
        }
        mock_post.return_value = mock_response

        result = get_cart_func(
            cart_id="gid://shopify/Cart/12345",
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
        )

        # Verify cart information and product details are included
        assert "Checkout URL: https://test-shop.myshopify.com/checkout/12345" in result
        assert "Product ID: gid://shopify/Product/22222" in result
        assert "Product Title: Test Product" in result

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_not_found(self, mock_post: Mock) -> None:
        """Test cart retrieval when cart is not found."""
        # Mock API response with no cart found
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"cart": None}}
        mock_post.return_value = mock_response

        with pytest.raises(ToolExecutionError) as exc_info:
            get_cart_func(
                cart_id="gid://shopify/Cart/99999",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )
        # Verify error message contains expected pattern
        assert "Tool get_cart execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_api_error(self, mock_post: Mock) -> None:
        """Test cart retrieval when API returns an error."""
        # Mock API error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        with pytest.raises(ToolExecutionError) as exc_info:
            get_cart_func(
                cart_id="gid://shopify/Cart/12345",
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )
        # Verify error message contains expected pattern
        assert "Tool get_cart execution failed" in str(exc_info.value)


class TestShopifyCartAddItems:
    """Integration tests for cart_add_items tool."""

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_success(self, mock_post: Mock) -> None:
        """Test successful cart item addition."""
        # Mock successful API response with updated cart
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cartLinesAdd": {
                    "cart": {
                        "id": "gid://shopify/Cart/12345",
                        "lines": {
                            "nodes": [
                                {
                                    "id": "gid://shopify/CartLine/67890",
                                    "quantity": 2,
                                    "merchandise": {
                                        "id": "gid://shopify/ProductVariant/11111",
                                        "title": "Small",
                                        "product": {
                                            "id": "gid://shopify/Product/22222",
                                            "title": "Test Product",
                                        },
                                    },
                                }
                            ]
                        },
                    }
                }
            }
        }
        mock_post.return_value = mock_response

        result = cart_add_items_func(
            cart_id="gid://shopify/Cart/12345",
            product_variant_ids=["gid://shopify/ProductVariant/11111"],
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
        )

        # Verify product information is included in the result
        assert "Test Product" in result
        assert "Small" in result
        assert "2" in result

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_api_error(self, mock_post: Mock) -> None:
        """Test cart item addition when API returns an error."""
        # Mock API error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        with pytest.raises(ToolExecutionError) as exc_info:
            cart_add_items_func(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/11111"],
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )
        # Verify error message contains expected pattern
        assert "Tool cart_add_items execution failed" in str(exc_info.value)


class TestShopifyReturnProducts:
    """Integration tests for return_products tool."""

    @patch("arklex.env.tools.shopify.return_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.return_products.shopify.GraphQL")
    def test_return_products_success(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test successful product return with fulfillment data."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock GraphQL response for returnable fulfillments
        mock_response = {
            "data": {
                "returnableFulfillments": {
                    "edges": [
                        {
                            "node": {
                                "id": "gid://shopify/ReturnableFulfillment/12345",
                                "fulfillment": {
                                    "id": "gid://shopify/Fulfillment/67890"
                                },
                                "returnableFulfillmentLineItems": {
                                    "edges": [
                                        {
                                            "node": {
                                                "fulfillmentLineItem": {
                                                    "id": "gid://shopify/FulfillmentLineItem/11111"
                                                },
                                                "quantity": 2,
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

        # Mock successful return request response
        mock_return_response = {
            "data": {
                "returnRequest": {
                    "return": {
                        "id": "gid://shopify/Return/12345",
                        "status": "SUBMITTED",
                    },
                    "userErrors": [],
                }
            }
        }

        mock_graphql_instance = MagicMock()
        # First call returns fulfillments, second call returns return request
        mock_graphql_instance.execute.side_effect = [
            json.dumps(mock_response),
            json.dumps(mock_return_response),
        ]
        mock_graphql.return_value = mock_graphql_instance

        result = return_products_func(
            return_order_id="gid://shopify/Order/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify successful return submission message
        assert "successfully submitted" in result

    @patch("arklex.env.tools.shopify.return_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.return_products.shopify.GraphQL")
    def test_return_products_no_fulfillment_found(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test product return when no fulfillment is found."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock response with no returnable fulfillments
        mock_response = {"data": {"returnableFulfillments": {"edges": []}}}

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        with pytest.raises(ToolExecutionError) as exc_info:
            return_products_func(
                return_order_id="gid://shopify/Order/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )
        # Verify error message contains expected pattern
        assert "Tool return_products execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.return_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.return_products.shopify.GraphQL")
    def test_return_products_malformed_fulfillment_response(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test error handling when the first GraphQL response is malformed (not JSON)."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session
        # Malformed JSON response
        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = ["not a json string"]
        mock_graphql.return_value = mock_graphql_instance
        with pytest.raises(ToolExecutionError) as exc_info:
            return_products_func(
                return_order_id="gid://shopify/Order/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )
        assert "Tool return_products execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.return_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.return_products.shopify.GraphQL")
    def test_return_products_malformed_return_request_response(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test error handling when the return request mutation response is malformed (not JSON)."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session
        # First call returns valid fulfillments, second call returns malformed JSON
        mock_response = {
            "data": {
                "returnableFulfillments": {
                    "edges": [
                        {
                            "node": {
                                "id": "gid://shopify/ReturnableFulfillment/12345",
                                "fulfillment": {
                                    "id": "gid://shopify/Fulfillment/67890"
                                },
                                "returnableFulfillmentLineItems": {
                                    "edges": [
                                        {
                                            "node": {
                                                "fulfillmentLineItem": {
                                                    "id": "gid://shopify/FulfillmentLineItem/11111"
                                                },
                                                "quantity": 2,
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
        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = [
            json.dumps(mock_response),
            "not a json string",
        ]
        mock_graphql.return_value = mock_graphql_instance
        with pytest.raises(ToolExecutionError) as exc_info:
            return_products_func(
                return_order_id="gid://shopify/Order/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )
        assert "Tool return_products execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.return_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.return_products.shopify.GraphQL")
    def test_return_products_user_errors_in_return_request(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test error handling when the return request mutation returns user errors."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session
        # First call returns valid fulfillments, second call returns userErrors
        mock_response = {
            "data": {
                "returnableFulfillments": {
                    "edges": [
                        {
                            "node": {
                                "id": "gid://shopify/ReturnableFulfillment/12345",
                                "fulfillment": {
                                    "id": "gid://shopify/Fulfillment/67890"
                                },
                                "returnableFulfillmentLineItems": {
                                    "edges": [
                                        {
                                            "node": {
                                                "fulfillmentLineItem": {
                                                    "id": "gid://shopify/FulfillmentLineItem/11111"
                                                },
                                                "quantity": 2,
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
        mock_return_response = {
            "data": {
                "returnRequest": None,
                "userErrors": [
                    {"field": ["returnLineItems"], "message": "Invalid return item."}
                ],
            }
        }
        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = [
            json.dumps(mock_response),
            json.dumps(mock_return_response),
        ]
        mock_graphql.return_value = mock_graphql_instance
        with pytest.raises(ToolExecutionError) as exc_info:
            return_products_func(
                return_order_id="gid://shopify/Order/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )
        assert "Tool return_products execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.return_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.return_products.shopify.GraphQL")
    def test_return_products_graphql_api_exception(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test error handling when the Shopify GraphQL client raises an exception."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session
        # Simulate API/network error on first call
        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = Exception("API/network error")
        mock_graphql.return_value = mock_graphql_instance
        with pytest.raises(ToolExecutionError) as exc_info:
            return_products_func(
                return_order_id="gid://shopify/Order/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )
        assert "Tool return_products execution failed" in str(exc_info.value)


class TestShopifyCancelOrder:
    """Integration tests for cancel_order tool."""

    @patch("arklex.env.tools.shopify.cancel_order.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.cancel_order.shopify.GraphQL")
    def test_cancel_order_success(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test successful order cancellation."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock GraphQL response for successful cancellation
        mock_response = {
            "data": {
                "orderCancel": {
                    "order": {
                        "id": "gid://shopify/Order/12345",
                        "cancelledAt": "2024-01-15T10:00:00Z",
                    },
                    "userErrors": [],
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = cancel_order_func(
            cancel_order_id="gid://shopify/Order/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify successful cancellation message and order ID
        assert "successfully cancelled" in result
        assert "gid://shopify/Order/12345" in result

    @patch("arklex.env.tools.shopify.cancel_order.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.cancel_order.shopify.GraphQL")
    def test_cancel_order_api_exception(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test order cancellation when Shopify API throws an exception."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Simulate API error by making execute method raise an exception
        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = Exception("API Error")
        mock_graphql.return_value = mock_graphql_instance

        with pytest.raises(ShopifyError) as exc_info:
            cancel_order_func(
                cancel_order_id="gid://shopify/Order/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )
        # Verify error message contains expected pattern
        assert "Order cancellation failed" in str(exc_info.value)


class TestShopifyGetWebProduct:
    """Integration tests for get_web_product tool."""

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_success(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test successful web product retrieval with complete product data."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock comprehensive GraphQL response with web product details
        mock_response = {
            "data": {
                "products": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Product/12345",
                            "title": "Test Product",
                            "description": "A test product description",
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
                                        "inventoryQuantity": 10,
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = get_web_product_func(
            web_product_id="gid://shopify/Product/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify product information is included in the result
        assert "Test Product" in result
        assert "gid://shopify/Product/12345" in result

    @patch("arklex.env.tools.shopify.get_web_product.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_web_product.shopify.GraphQL")
    def test_get_web_product_not_found(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test web product retrieval when product is not found."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock empty product response
        mock_response = {"data": {"products": {"nodes": []}}}

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        with pytest.raises(ToolExecutionError) as exc_info:
            get_web_product_func(
                web_product_id="gid://shopify/Product/99999",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )
        # Verify error message contains expected pattern
        assert "Tool get_web_product execution failed" in str(exc_info.value)


class TestShopifyToolsEdgeCases:
    """Edge case tests for Shopify tools covering boundary conditions and error scenarios."""

    @patch("arklex.env.tools.shopify.search_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.search_products.shopify.GraphQL")
    def test_search_products_with_empty_query(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test product search with empty query string."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock empty search results for empty query
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

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        with pytest.raises(ToolExecutionError) as exc_info:
            search_products_func(
                product_query="",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )
        # Verify error message contains expected pattern
        assert "Tool search_products execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_cart.requests.post")
    def test_get_cart_with_empty_cart(self, mock_post: Mock) -> None:
        """Test cart retrieval when cart is empty."""
        # Mock API response with empty cart
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "cart": {
                    "id": "gid://shopify/Cart/12345",
                    "checkoutUrl": "https://test-shop.myshopify.com/checkout/12345",
                    "lines": {"nodes": []},
                }
            }
        }
        mock_post.return_value = mock_response

        result = get_cart_func(
            cart_id="gid://shopify/Cart/12345",
            shop_url="https://test-shop.myshopify.com",
            storefront_token="test_storefront_token",
            api_version="2024-10",
        )

        # Verify checkout URL is included but no product information
        assert "Checkout URL: https://test-shop.myshopify.com/checkout/12345" in result

    @patch("arklex.env.tools.shopify.cart_add_items.requests.post")
    def test_cart_add_items_with_zero_quantity(self, mock_post: Mock) -> None:
        """Test cart item addition with zero quantity."""
        # Mock API error response for zero quantity
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        with pytest.raises(ToolExecutionError) as exc_info:
            cart_add_items_func(
                cart_id="gid://shopify/Cart/12345",
                product_variant_ids=["gid://shopify/ProductVariant/11111"],
                shop_url="https://test-shop.myshopify.com",
                storefront_token="test_storefront_token",
                api_version="2024-10",
            )
        # Verify error message contains expected pattern
        assert "Tool cart_add_items execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_with_numeric_id(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test user details retrieval with numeric user ID instead of full GID."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock successful response
        mock_response = {
            "data": {
                "customer": {
                    "firstName": "John",
                    "lastName": "Doe",
                    "email": "john.doe@example.com",
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = get_user_details_admin_func(
            user_id="12345",  # Numeric ID instead of full GID
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        user_data = json.loads(result)
        # Verify customer information is correctly retrieved
        assert user_data["firstName"] == "John"
        assert user_data["lastName"] == "Doe"

    @patch("arklex.env.tools.shopify.get_products.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_products.shopify.GraphQL")
    def test_get_products_with_multiple_ids(
        self,
        mock_graphql: Mock,
        mock_session_temp: Mock,
    ) -> None:
        """Test product details retrieval with multiple product IDs."""
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        # Mock response with multiple products
        mock_response = {
            "data": {
                "products": {
                    "nodes": [
                        {"id": "gid://shopify/Product/12345", "title": "Product 1"},
                        {"id": "gid://shopify/Product/12346", "title": "Product 2"},
                    ]
                }
            }
        }

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        result = get_products_func(
            product_ids=["gid://shopify/Product/12345", "gid://shopify/Product/12346"],
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify all products are included in the result
        assert "Product 1" in result
        assert "Product 2" in result
        assert "gid://shopify/Product/12345" in result
        assert "gid://shopify/Product/12346" in result
