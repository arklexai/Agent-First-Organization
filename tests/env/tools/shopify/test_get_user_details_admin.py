"""Tests for arklex.env.tools.shopify.get_user_details_admin module."""

import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.tools.shopify.get_user_details_admin import get_user_details_admin
from arklex.utils.exceptions import ToolExecutionError


class TestGetUserDetailsAdmin:
    """Test cases for get_user_details_admin function."""

    def setup_method(self) -> None:
        """Set up test environment."""
        os.environ["ARKLEX_TEST_ENV"] = "local"

    def teardown_method(self) -> None:
        """Clean up test environment."""
        if "ARKLEX_TEST_ENV" in os.environ:
            del os.environ["ARKLEX_TEST_ENV"]

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
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
                    "tags": ["VIP", "Wholesale"],
                    "lifetimeDuration": 365,
                    "addresses": [{"address1": "123 Main St"}],
                    "orders": {
                        "nodes": [
                            {"id": "gid://shopify/Order/12345"},
                            {"id": "gid://shopify/Order/12346"},
                        ]
                    },
                }
            }
        }
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function
        result = get_user_details_admin().func(
            user_id="gid://shopify/Customer/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result is JSON string
        assert isinstance(result, str)
        user_data = json.loads(result)
        assert user_data["firstName"] == "John"
        assert user_data["lastName"] == "Doe"
        assert user_data["email"] == "john.doe@example.com"
        assert user_data["phone"] == "+1234567890"
        assert user_data["numberOfOrders"] == 5
        assert user_data["amountSpent"]["amount"] == "299.99"
        assert user_data["amountSpent"]["currencyCode"] == "USD"
        assert user_data["note"] == "VIP customer"
        assert user_data["verifiedEmail"] is True
        assert user_data["validEmailAddress"] is True
        assert user_data["tags"] == ["VIP", "Wholesale"]
        assert user_data["lifetimeDuration"] == 365
        assert len(user_data["addresses"]) == 1
        assert user_data["addresses"][0]["address1"] == "123 Main St"
        assert len(user_data["orders"]["nodes"]) == 2

        # Verify GraphQL query was called with correct parameters
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "customer(id:" in call_args
        assert "gid://shopify/Customer/12345" in call_args

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
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
            get_user_details_admin().func(
                user_id="gid://shopify/Customer/99999",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
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
            get_user_details_admin().func(
                user_id="gid://shopify/Customer/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_session_exception(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test user details retrieval when session creation fails."""
        # Setup mocks to raise exception during session creation
        mock_session_temp.side_effect = Exception("Session creation failed")

        # Execute function and verify exception
        with pytest.raises(ToolExecutionError) as exc_info:
            get_user_details_admin().func(
                user_id="gid://shopify/Customer/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
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
        result = get_user_details_admin().func(
            user_id="gid://shopify/Customer/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
            limit="10",
            navigate="next",
            pageInfo='{"endCursor": "cursor1", "hasNextPage": true}',
        )

        # Verify result
        assert isinstance(result, str)
        user_data = json.loads(result)
        assert user_data["firstName"] == "John"
        assert user_data["lastName"] == "Doe"
        assert user_data["email"] == "john.doe@example.com"

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
    def test_get_user_details_admin_with_navigation_return_early(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test user details retrieval when navigation returns early."""
        # This test simulates the case where cursorify returns early
        with patch(
            "arklex.env.tools.shopify.get_user_details_admin.cursorify"
        ) as mock_cursorify:
            mock_cursorify.return_value = ("first: 10", False)

            # Execute function
            result = get_user_details_admin().func(
                user_id="gid://shopify/Customer/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
                limit="10",
            )

            # Verify result is the early return value
            assert result == "first: 10"

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
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
        result = get_user_details_admin().func(
            user_id="gid://shopify/Customer/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result
        assert isinstance(result, str)
        user_data = json.loads(result)
        assert user_data["firstName"] == "John"
        assert user_data["lastName"] == "Doe"
        assert user_data["email"] == "john.doe@example.com"
        assert user_data["orders"]["nodes"] == []

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
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
        result = get_user_details_admin().func(
            user_id="gid://shopify/Customer/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result
        assert isinstance(result, str)
        user_data = json.loads(result)
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

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
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
        result = get_user_details_admin().func(
            user_id="12345",  # Numeric ID instead of full GID
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result
        assert isinstance(result, str)
        user_data = json.loads(result)
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
        # Verify the function has the expected attributes from register_tool decorator
        tool_instance = get_user_details_admin()
        assert hasattr(tool_instance, "func")
        assert hasattr(tool_instance, "name")
        assert hasattr(tool_instance, "description")
        assert hasattr(tool_instance, "slots")
        assert hasattr(tool_instance, "output")

        # Verify the function signature
        import inspect

        sig = inspect.signature(tool_instance.func)
        assert "user_id" in sig.parameters
        assert "kwargs" in sig.parameters

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
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
            get_user_details_admin().func(
                user_id="gid://shopify/Customer/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
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
            get_user_details_admin().func(
                user_id="gid://shopify/Customer/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.get_user_details_admin.shopify.GraphQL")
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
            get_user_details_admin().func(
                user_id="gid://shopify/Customer/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Tool get_user_details_admin execution failed" in str(exc_info.value)
