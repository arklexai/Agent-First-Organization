"""Tests for arklex.env.tools.shopify.cancel_order module."""

import json
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from arklex.env.tools.shopify.cancel_order import cancel_order
from arklex.utils.exceptions import ShopifyError


class TestCancelOrder:
    """Test cases for cancel_order function."""

    def setup_method(self) -> None:
        """Set up test environment."""
        os.environ["ARKLEX_TEST_ENV"] = "local"

    def get_original_function(self) -> callable:
        """Get the original function from the decorated function."""
        # Create a Tool instance and access the original function
        tool_instance = cancel_order()
        return tool_instance.func

    def teardown_method(self) -> None:
        """Clean up test environment."""
        if "ARKLEX_TEST_ENV" in os.environ:
            del os.environ["ARKLEX_TEST_ENV"]

    @patch("arklex.env.tools.shopify.cancel_order.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.cancel_order.shopify.GraphQL")
    def test_cancel_order_success(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test successful order cancellation."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"data": {"orderCancel": {"userErrors": []}}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function
        result = self.get_original_function()(
            cancel_order_id="gid://shopify/Order/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result
        assert "successfully cancelled" in result

        # Verify GraphQL query was called with correct parameters
        mock_graphql_instance.execute.assert_called_once()
        call_args = mock_graphql_instance.execute.call_args[0][0]
        assert "orderCancel" in call_args
        assert "gid://shopify/Order/12345" in call_args
        assert "CUSTOMER" in call_args
        assert "notifyCustomer: true" in call_args
        assert "restock: true" in call_args
        assert "refund: true" in call_args

    @patch("arklex.env.tools.shopify.cancel_order.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.cancel_order.shopify.GraphQL")
    def test_cancel_order_with_user_errors(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test order cancellation when Shopify returns user errors."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {
            "data": {
                "orderCancel": {
                    "userErrors": [{"field": "orderId", "message": "Order not found"}]
                }
            }
        }
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ShopifyError) as exc_info:
            self.get_original_function()(
                cancel_order_id="gid://shopify/Order/99999",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Order cancellation failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.cancel_order.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.cancel_order.shopify.GraphQL")
    def test_cancel_order_graphql_exception(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test order cancellation when GraphQL execution fails."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.side_effect = Exception("GraphQL API Error")
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ShopifyError) as exc_info:
            self.get_original_function()(
                cancel_order_id="gid://shopify/Order/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Order cancellation failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.cancel_order.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.cancel_order.shopify.GraphQL")
    def test_cancel_order_json_decode_error(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test order cancellation when JSON response is malformed."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_graphql_instance.execute.return_value = "invalid json"
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ShopifyError) as exc_info:
            self.get_original_function()(
                cancel_order_id="gid://shopify/Order/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Order cancellation failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.cancel_order.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.cancel_order.shopify.GraphQL")
    def test_cancel_order_missing_data_key(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test order cancellation when response is missing 'data' key."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"orderCancel": {"userErrors": []}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ShopifyError) as exc_info:
            self.get_original_function()(
                cancel_order_id="gid://shopify/Order/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Order cancellation failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.cancel_order.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.cancel_order.shopify.GraphQL")
    def test_cancel_order_missing_order_cancel_key(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test order cancellation when response is missing 'orderCancel' key."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"data": {"someOtherKey": {"userErrors": []}}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function and verify exception
        with pytest.raises(ShopifyError) as exc_info:
            self.get_original_function()(
                cancel_order_id="gid://shopify/Order/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Order cancellation failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.cancel_order.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.cancel_order.shopify.GraphQL")
    def test_cancel_order_session_exception(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test order cancellation when session creation fails."""
        # Setup mocks to raise exception during session creation
        mock_session_temp.side_effect = Exception("Session creation failed")

        # Execute function and verify exception
        with pytest.raises(ShopifyError) as exc_info:
            self.get_original_function()(
                cancel_order_id="gid://shopify/Order/12345",
                shop_url="https://test-shop.myshopify.com",
                admin_token="test_admin_token",
                api_version="2024-10",
            )

        assert "Order cancellation failed" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.cancel_order.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.cancel_order.shopify.GraphQL")
    def test_cancel_order_with_empty_user_errors(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test order cancellation with empty user errors array."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"data": {"orderCancel": {"userErrors": []}}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function
        result = self.get_original_function()(
            cancel_order_id="gid://shopify/Order/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result
        assert "successfully cancelled" in result

    @patch("arklex.env.tools.shopify.cancel_order.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.cancel_order.shopify.GraphQL")
    def test_cancel_order_with_none_user_errors(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test order cancellation with None user errors."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"data": {"orderCancel": {"userErrors": None}}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function
        result = self.get_original_function()(
            cancel_order_id="gid://shopify/Order/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result
        assert "successfully cancelled" in result

    @patch("arklex.env.tools.shopify.cancel_order.shopify.Session.temp")
    @patch("arklex.env.tools.shopify.cancel_order.shopify.GraphQL")
    def test_cancel_order_with_missing_user_errors_key(
        self, mock_graphql: Mock, mock_session_temp: Mock
    ) -> None:
        """Test order cancellation when userErrors key is missing."""
        # Setup mocks
        mock_session = MagicMock()
        mock_session_temp.return_value.__enter__.return_value = mock_session

        mock_graphql_instance = MagicMock()
        mock_response = {"data": {"orderCancel": {}}}
        mock_graphql_instance.execute.return_value = json.dumps(mock_response)
        mock_graphql.return_value = mock_graphql_instance

        # Execute function
        result = self.get_original_function()(
            cancel_order_id="gid://shopify/Order/12345",
            shop_url="https://test-shop.myshopify.com",
            admin_token="test_admin_token",
            api_version="2024-10",
        )

        # Verify result
        assert "successfully cancelled" in result

    def test_cancel_order_function_registration(self) -> None:
        """Test that the cancel_order function is properly registered as a tool."""
        # Verify the function is callable
        assert callable(cancel_order)

        # Verify the function returns a Tool instance when called
        tool_instance = cancel_order()
        from arklex.env.tools.tools import Tool

        assert isinstance(tool_instance, Tool)

        # Verify the tool has the expected attributes
        assert hasattr(tool_instance, "description")
        assert hasattr(tool_instance, "slots")
        assert hasattr(tool_instance, "output")

        # Verify the description matches expected value
        assert "Cancel order by order id" in tool_instance.description
