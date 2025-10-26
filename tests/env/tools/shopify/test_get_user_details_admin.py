"""Tests for arklex.resources.tools.shopify.get_user_details_admin module."""

import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.resources.tools.shopify.get_user_details_admin import get_user_details_admin
from arklex.utils.logging.exceptions import ToolExecutionError


class TestGetUserDetailsAdmin:
    """Test cases for get_user_details_admin function."""

    def setup_method(self) -> None:
        """Set up test environment."""
        os.environ["ARKLEX_TEST_ENV"] = "local"

    def get_original_function(self) -> callable:
        """Get the original function from the decorated function."""
        # Access the original function from the Tool instance
        return get_user_details_admin.func

    def teardown_method(self) -> None:
        """Clean up test environment."""
        if "ARKLEX_TEST_ENV" in os.environ:
            del os.environ["ARKLEX_TEST_ENV"]

    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_success(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test successful user details retrieval."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "customer": {
                    "id": "gid://shopify/Customer/123",
                    "firstName": "John",
                    "lastName": "Doe",
                    "email": "john.doe@example.com",
                    "phone": "+1234567890",
                    "orders": {
                        "nodes": [
                            {
                                "id": "gid://shopify/Order/456",
                                "name": "#1001",
                                "createdAt": "2024-01-01T00:00:00Z",
                            }
                        ]
                    },
                }
            }
        }
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function
        result = self.get_original_function()(
            user_id="gid://shopify/Customer/123",
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "admin_token": "test_admin_token",
                "api_version": "2024-10",
            },
        )

        # Verify result
        assert "John" in result.message_flow
        assert "john.doe@example.com" in result.message_flow

    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_user_not_found(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test user details retrieval when user is not found."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"data": {"customer": None}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                user_id="gid://shopify/Customer/99999",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "admin_token": "test_admin_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_graphql_exception(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test user details retrieval when GraphQL execution fails."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = Exception("GraphQL API Error")
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                user_id="gid://shopify/Customer/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "admin_token": "test_admin_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_session_exception(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test user details retrieval when session creation fails."""
        # Setup mocks to raise exception during session creation
        mock_session_temp.side_effect = Exception("Session creation failed")

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                user_id="gid://shopify/Customer/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "admin_token": "test_admin_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_with_pagination_parameters(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test user details retrieval with pagination parameters."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "customer": {
                    "firstName": "John",
                    "lastName": "Doe",
                    "email": "john.doe@example.com",
                    "phone": "+1234567890",
                    "numberOfOrders": 5,
                    "amountSpent": {"amount": "299.99", "currencyCode": "USD"},
                    "createdAt": "2023-01-15T10:00:00Z",
                    "updatedAt": "2024-01-15T10:00:00Z",
                    "note": "VIP customer",
                    "verifiedEmail": True,
                    "validEmailAddress": True,
                    "tags": ["VIP"],
                    "lifetimeDuration": 365,
                    "addresses": [{"address1": "123 Main St"}],
                    "orders": {
                        "nodes": [{"id": "gid://shopify/Order/12345"}],
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
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function with pagination parameters
        result = self.get_original_function()(
            user_id="gid://shopify/Customer/12345",
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
        user_data = json.loads(result.message_flow)
        assert user_data["firstName"] == "John"
        assert user_data["lastName"] == "Doe"
        assert user_data["email"] == "john.doe@example.com"

    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_with_missing_fields(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test user details retrieval when user has missing fields."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "customer": {
                    "firstName": "John",
                    "lastName": "Doe",
                    "email": "john.doe@example.com",
                    # Missing other fields
                    "orders": {"nodes": []},
                }
            }
        }
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function
        result = self.get_original_function()(
            user_id="gid://shopify/Customer/12345",
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "admin_token": "test_admin_token",
                "api_version": "2024-10",
            },
        )

        # Verify result
        user_data = json.loads(result.message_flow)
        assert user_data["firstName"] == "John"
        assert user_data["lastName"] == "Doe"
        assert user_data["email"] == "john.doe@example.com"
        assert user_data["orders"]["nodes"] == []

    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_with_empty_orders(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test user details retrieval when user has no orders."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "customer": {
                    "firstName": "John",
                    "lastName": "Doe",
                    "email": "john.doe@example.com",
                    "phone": "+1234567890",
                    "numberOfOrders": 0,
                    "amountSpent": {"amount": "0.00", "currencyCode": "USD"},
                    "createdAt": "2023-01-15T10:00:00Z",
                    "updatedAt": "2024-01-15T10:00:00Z",
                    "note": None,
                    "verifiedEmail": False,
                    "validEmailAddress": True,
                    "tags": [],
                    "lifetimeDuration": 0,
                    "addresses": [],
                    "orders": {"nodes": []},
                }
            }
        }
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function
        result = self.get_original_function()(
            user_id="gid://shopify/Customer/12345",
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "admin_token": "test_admin_token",
                "api_version": "2024-10",
            },
        )

        # Verify result
        user_data = json.loads(result.message_flow)
        assert user_data["firstName"] == "John"
        assert user_data["lastName"] == "Doe"
        assert user_data["email"] == "john.doe@example.com"
        assert user_data["numberOfOrders"] == 0
        assert user_data["amountSpent"]["amount"] == "0.00"
        assert user_data["note"] is None
        assert user_data["verifiedEmail"] is False
        assert user_data["tags"] == []
        assert user_data["lifetimeDuration"] == 0
        assert user_data["addresses"] == []
        assert user_data["orders"]["nodes"] == []

    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_with_numeric_id(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test user details retrieval with numeric user ID."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "customer": {
                    "firstName": "John",
                    "lastName": "Doe",
                    "email": "john.doe@example.com",
                    "orders": {"nodes": []},
                }
            }
        }
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function with numeric ID
        result = self.get_original_function()(
            user_id="12345",  # Numeric ID instead of full GID
            auth={
                "shop_url": "https://test-shop.myshopify.com",
                "admin_token": "test_admin_token",
                "api_version": "2024-10",
            },
        )

        # Verify result
        user_data = json.loads(result.message_flow)
        assert user_data["firstName"] == "John"
        assert user_data["lastName"] == "Doe"
        assert user_data["email"] == "john.doe@example.com"

        # Verify GraphQL query was called with numeric ID
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "customer(id:" in call_args
        assert "12345" in call_args

    def test_get_user_details_admin_function_registration(self) -> None:
        """Test that the get_user_details_admin function is properly registered as a tool."""
        # Verify the function returns a Tool instance when called
        tool_instance = get_user_details_admin
        from arklex.resources.tools.tools import Tool

        assert isinstance(tool_instance, Tool)

        # Verify the tool has the expected attributes
        assert hasattr(tool_instance, "description")
        assert hasattr(tool_instance, "slots")

    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_with_json_decode_error(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test user details retrieval when JSON response is malformed."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = "invalid json"
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                user_id="gid://shopify/Customer/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "admin_token": "test_admin_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_with_missing_data_key(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test user details retrieval when response is missing 'data' key."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"customer": {"firstName": "John", "lastName": "Doe"}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                user_id="gid://shopify/Customer/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "admin_token": "test_admin_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)

    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.resources.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_with_missing_customer_key(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test user details retrieval when response is missing 'customer' key."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"data": {"someOtherKey": {"firstName": "John"}}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            self.get_original_function()(
                user_id="gid://shopify/Customer/12345",
                auth={
                    "shop_url": "https://test-shop.myshopify.com",
                    "admin_token": "test_admin_token",
                    "api_version": "2024-10",
                },
            )

        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)
