"""Tests for arklex.env.tools.shopify.utils module."""

from unittest.mock import Mock, patch

import pytest
import requests

from arklex.env.tools.shopify.utils import (
    SHOPIFY_ADMIN_AUTH_ERROR_MSG,
    SHOPIFY_STOREFRONT_AUTH_ERROR_MSG,
    authorify_admin,
    authorify_storefront,
    make_query,
)
from arklex.utils.exceptions import AuthenticationError


class TestAuthorifyAdmin:
    """Test cases for authorify_admin function."""

    def test_authorify_admin_success(self) -> None:
        """Test successful admin authentication."""
        kwargs = {
            "shop_url": "test.myshopify.com",
            "api_version": "2024-01",
            "admin_token": "test-token",
        }

        result = authorify_admin(kwargs)

        expected = {
            "domain": "test.myshopify.com",
            "version": "2024-01",
            "token": "test-token",
        }
        assert result == expected

    def test_authorify_admin_missing_shop_url(self) -> None:
        """Test admin authentication with missing shop_url."""
        kwargs = {
            "api_version": "2024-01",
            "admin_token": "test-token",
        }

        with pytest.raises(AuthenticationError) as exc_info:
            authorify_admin(kwargs)

        assert "Shopify admin authentication failed" in str(exc_info.value)
        assert SHOPIFY_ADMIN_AUTH_ERROR_MSG in str(exc_info.value)

    def test_authorify_admin_missing_api_version(self) -> None:
        """Test admin authentication with missing api_version."""
        kwargs = {
            "shop_url": "test.myshopify.com",
            "admin_token": "test-token",
        }

        with pytest.raises(AuthenticationError) as exc_info:
            authorify_admin(kwargs)

        assert "Shopify admin authentication failed" in str(exc_info.value)
        assert SHOPIFY_ADMIN_AUTH_ERROR_MSG in str(exc_info.value)

    def test_authorify_admin_missing_admin_token(self) -> None:
        """Test admin authentication with missing admin_token."""
        kwargs = {
            "shop_url": "test.myshopify.com",
            "api_version": "2024-01",
        }

        with pytest.raises(AuthenticationError) as exc_info:
            authorify_admin(kwargs)

        assert "Shopify admin authentication failed" in str(exc_info.value)
        assert SHOPIFY_ADMIN_AUTH_ERROR_MSG in str(exc_info.value)

    def test_authorify_admin_empty_values(self) -> None:
        """Test admin authentication with empty values."""
        kwargs = {
            "shop_url": "",
            "api_version": "",
            "admin_token": "",
        }

        with pytest.raises(AuthenticationError) as exc_info:
            authorify_admin(kwargs)

        assert "Shopify admin authentication failed" in str(exc_info.value)
        assert SHOPIFY_ADMIN_AUTH_ERROR_MSG in str(exc_info.value)

    def test_authorify_admin_none_values(self) -> None:
        """Test admin authentication with None values."""
        kwargs = {
            "shop_url": None,
            "api_version": None,
            "admin_token": None,
        }

        with pytest.raises(AuthenticationError) as exc_info:
            authorify_admin(kwargs)

        assert "Shopify admin authentication failed" in str(exc_info.value)
        assert SHOPIFY_ADMIN_AUTH_ERROR_MSG in str(exc_info.value)


class TestAuthorifyStorefront:
    """Test cases for authorify_storefront function."""

    def test_authorify_storefront_success(self) -> None:
        """Test successful storefront authentication."""
        kwargs = {
            "shop_url": "test.myshopify.com",
            "api_version": "2024-01",
            "storefront_token": "test-token",
        }

        result = authorify_storefront(kwargs)

        expected = {
            "storefront_token": "test-token",
            "storefront_url": "test.myshopify.com/api/2024-01/graphql.json",
        }
        assert result == expected

    def test_authorify_storefront_missing_shop_url(self) -> None:
        """Test storefront authentication with missing shop_url."""
        kwargs = {
            "api_version": "2024-01",
            "storefront_token": "test-token",
        }

        with pytest.raises(AuthenticationError) as exc_info:
            authorify_storefront(kwargs)

        assert "Shopify storefront authentication failed" in str(exc_info.value)
        assert SHOPIFY_STOREFRONT_AUTH_ERROR_MSG in str(exc_info.value)

    def test_authorify_storefront_missing_api_version(self) -> None:
        """Test storefront authentication with missing api_version."""
        kwargs = {
            "shop_url": "test.myshopify.com",
            "storefront_token": "test-token",
        }

        with pytest.raises(AuthenticationError) as exc_info:
            authorify_storefront(kwargs)

        assert "Shopify storefront authentication failed" in str(exc_info.value)
        assert SHOPIFY_STOREFRONT_AUTH_ERROR_MSG in str(exc_info.value)

    def test_authorify_storefront_missing_storefront_token(self) -> None:
        """Test storefront authentication with missing storefront_token."""
        kwargs = {
            "shop_url": "test.myshopify.com",
            "api_version": "2024-01",
        }

        with pytest.raises(AuthenticationError) as exc_info:
            authorify_storefront(kwargs)

        assert "Shopify storefront authentication failed" in str(exc_info.value)
        assert SHOPIFY_STOREFRONT_AUTH_ERROR_MSG in str(exc_info.value)

    def test_authorify_storefront_empty_values(self) -> None:
        """Test storefront authentication with empty values."""
        kwargs = {
            "shop_url": "",
            "api_version": "",
            "storefront_token": "",
        }

        with pytest.raises(AuthenticationError) as exc_info:
            authorify_storefront(kwargs)

        assert "Shopify storefront authentication failed" in str(exc_info.value)
        assert SHOPIFY_STOREFRONT_AUTH_ERROR_MSG in str(exc_info.value)

    def test_authorify_storefront_none_values(self) -> None:
        """Test storefront authentication with None values."""
        kwargs = {
            "shop_url": None,
            "api_version": None,
            "storefront_token": None,
        }

        with pytest.raises(AuthenticationError) as exc_info:
            authorify_storefront(kwargs)

        assert "Shopify storefront authentication failed" in str(exc_info.value)
        assert SHOPIFY_STOREFRONT_AUTH_ERROR_MSG in str(exc_info.value)


class TestMakeQuery:
    """Test cases for make_query function."""

    @patch("arklex.env.tools.shopify.utils.requests.post")
    def test_make_query_success(self, mock_post: Mock) -> None:
        """Test successful query execution."""
        url = "https://test.myshopify.com/api/2024-01/graphql.json"
        query = "query { products { nodes { id title } } }"
        variables = {"limit": 10}
        headers = {"Authorization": "Bearer test-token"}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"products": {"nodes": []}}}
        mock_post.return_value = mock_response

        result = make_query(url, query, variables, headers)

        expected = {"data": {"products": {"nodes": []}}}
        assert result == expected

        mock_post.assert_called_once_with(
            url,
            json={"query": query, "variables": variables},
            headers=headers,
        )

    @patch("arklex.env.tools.shopify.utils.requests.post")
    def test_make_query_http_error(self, mock_post: Mock) -> None:
        """Test query execution with HTTP error."""
        url = "https://test.myshopify.com/api/2024-01/graphql.json"
        query = "query { products { nodes { id title } } }"
        variables = {"limit": 10}
        headers = {"Authorization": "Bearer test-token"}

        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            make_query(url, query, variables, headers)

        assert "Query failed to run by returning code of 400" in str(exc_info.value)
        assert query in str(exc_info.value)

    @patch("arklex.env.tools.shopify.utils.requests.post")
    def test_make_query_server_error(self, mock_post: Mock) -> None:
        """Test query execution with server error."""
        url = "https://test.myshopify.com/api/2024-01/graphql.json"
        query = "query { products { nodes { id title } } }"
        variables = {"limit": 10}
        headers = {"Authorization": "Bearer test-token"}

        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            make_query(url, query, variables, headers)

        assert "Query failed to run by returning code of 500" in str(exc_info.value)
        assert query in str(exc_info.value)

    @patch("arklex.env.tools.shopify.utils.requests.post")
    def test_make_query_network_error(self, mock_post: Mock) -> None:
        """Test query execution with network error."""
        url = "https://test.myshopify.com/api/2024-01/graphql.json"
        query = "query { products { nodes { id title } } }"
        variables = {"limit": 10}
        headers = {"Authorization": "Bearer test-token"}

        mock_post.side_effect = requests.RequestException("Network error")

        with pytest.raises(requests.RequestException) as exc_info:
            make_query(url, query, variables, headers)

        assert "Network error" in str(exc_info.value)

    @patch("arklex.env.tools.shopify.utils.requests.post")
    def test_make_query_with_empty_variables(self, mock_post: Mock) -> None:
        """Test query execution with empty variables."""
        url = "https://test.myshopify.com/api/2024-01/graphql.json"
        query = "query { products { nodes { id title } } }"
        variables = {}
        headers = {"Authorization": "Bearer test-token"}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"products": {"nodes": []}}}
        mock_post.return_value = mock_response

        result = make_query(url, query, variables, headers)

        expected = {"data": {"products": {"nodes": []}}}
        assert result == expected

        mock_post.assert_called_once_with(
            url,
            json={"query": query, "variables": variables},
            headers=headers,
        )

    @patch("arklex.env.tools.shopify.utils.requests.post")
    def test_make_query_with_empty_headers(self, mock_post: Mock) -> None:
        """Test query execution with empty headers."""
        url = "https://test.myshopify.com/api/2024-01/graphql.json"
        query = "query { products { nodes { id title } } }"
        variables = {"limit": 10}
        headers = {}

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"products": {"nodes": []}}}
        mock_post.return_value = mock_response

        result = make_query(url, query, variables, headers)

        expected = {"data": {"products": {"nodes": []}}}
        assert result == expected

        mock_post.assert_called_once_with(
            url,
            json={"query": query, "variables": variables},
            headers=headers,
        )
