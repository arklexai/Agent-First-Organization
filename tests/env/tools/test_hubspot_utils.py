"""
Tests for the HubSpot utils module.

This module contains comprehensive test cases for HubSpot utility functions,
including authentication, token refresh, and error handling.
"""

import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
import requests

# Import the types and exceptions that don't depend on mysql_pool
from arklex.env.tools.types import ResourceAuthGroup
from arklex.utils.exceptions import AuthenticationError

# Mock the mysql_pool import to prevent initialization issues
with patch("arklex.env.tools.hubspot.utils.mysql_pool") as mock_mysql_pool:
    from arklex.env.tools.hubspot.utils import (
        HUBSPOT_AUTH_ERROR,
        HubspotAuthTokens,
        HubspotNotIntegratedError,
        authenticate_hubspot,
        refresh_token_if_needed,
    )


class TestAuthenticateHubspot:
    """Test cases for authenticate_hubspot function."""

    def test_authenticate_hubspot_success(self) -> None:
        """Test successful HubSpot authentication with valid access token."""
        # Setup
        kwargs = {"access_token": "test_access_token_123"}

        # Execute
        result = authenticate_hubspot(kwargs)

        # Assert
        assert result == "test_access_token_123"

    def test_authenticate_hubspot_missing_token(self) -> None:
        """Test HubSpot authentication failure when access token is missing."""
        # Setup
        kwargs = {}

        # Execute & Assert
        with pytest.raises(AuthenticationError) as exc_info:
            authenticate_hubspot(kwargs)

        assert str(exc_info.value) == HUBSPOT_AUTH_ERROR + " (AUTHENTICATION_ERROR)"

    def test_authenticate_hubspot_empty_token(self) -> None:
        """Test HubSpot authentication failure when access token is empty."""
        # Setup
        kwargs = {"access_token": ""}

        # Execute & Assert
        with pytest.raises(AuthenticationError) as exc_info:
            authenticate_hubspot(kwargs)

        assert str(exc_info.value) == HUBSPOT_AUTH_ERROR + " (AUTHENTICATION_ERROR)"

    def test_authenticate_hubspot_none_token(self) -> None:
        """Test HubSpot authentication failure when access token is None."""
        # Setup
        kwargs = {"access_token": None}

        # Execute & Assert
        with pytest.raises(AuthenticationError) as exc_info:
            authenticate_hubspot(kwargs)

        assert str(exc_info.value) == HUBSPOT_AUTH_ERROR + " (AUTHENTICATION_ERROR)"

    def test_authenticate_hubspot_with_additional_params(self) -> None:
        """Test HubSpot authentication with additional parameters in kwargs."""
        # Setup
        kwargs = {
            "access_token": "test_access_token_456",
            "refresh_token": "test_refresh_token",
            "expiry_time": "2024-01-01T00:00:00Z",
        }

        # Execute
        result = authenticate_hubspot(kwargs)

        # Assert
        assert result == "test_access_token_456"


class TestHubspotNotIntegratedError:
    """Test cases for HubspotNotIntegratedError exception class."""

    def test_hubspot_not_integrated_error_creation(self) -> None:
        """Test HubspotNotIntegratedError exception creation with parameters."""
        # Setup
        hubspot_account_id = "test_account_123"
        bot_id = "test_bot_456"
        bot_version = "v1.0"

        # Execute
        error = HubspotNotIntegratedError(hubspot_account_id, bot_id, bot_version)

        # Assert
        assert error.hubspot_account_id == hubspot_account_id
        assert error.bot_id == bot_id
        assert error.bot_version == bot_version
        assert (
            str(error)
            == f"HubSpot not integrated for bot {bot_id} version {bot_version}"
        )

    def test_hubspot_not_integrated_error_inheritance(self) -> None:
        """Test that HubspotNotIntegratedError properly inherits from Exception."""
        # Setup
        error = HubspotNotIntegratedError("account", "bot", "version")

        # Assert
        assert isinstance(error, Exception)
        assert isinstance(error, HubspotNotIntegratedError)


class TestHubspotAuthTokens:
    """Test cases for HubspotAuthTokens Pydantic model."""

    def test_hubspot_auth_tokens_creation(self) -> None:
        """Test HubspotAuthTokens model creation with valid data."""
        # Setup
        access_token = "test_access_token"
        refresh_token = "test_refresh_token"
        expiry_time_str = "2024-01-01T00:00:00Z"

        # Execute
        tokens = HubspotAuthTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expiry_time_str=expiry_time_str,
        )

        # Assert
        assert tokens.access_token == access_token
        assert tokens.refresh_token == refresh_token
        assert tokens.expiry_time_str == expiry_time_str

    def test_hubspot_auth_tokens_model_dump_json(self) -> None:
        """Test HubspotAuthTokens model_dump_json method."""
        # Setup
        tokens = HubspotAuthTokens(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expiry_time_str="2024-01-01T00:00:00Z",
        )

        # Execute
        json_str = tokens.model_dump_json()

        # Assert
        assert isinstance(json_str, str)
        assert "test_access_token" in json_str
        assert "test_refresh_token" in json_str
        assert "2024-01-01T00:00:00Z" in json_str

    def test_hubspot_auth_tokens_validation(self) -> None:
        """Test HubspotAuthTokens validation with invalid data."""
        # Setup & Execute & Assert
        with pytest.raises(ValueError):
            HubspotAuthTokens(
                access_token=123,  # Integer should fail validation for str field
                refresh_token="test_refresh_token",
                expiry_time_str="2024-01-01T00:00:00Z",
            )


class TestRefreshTokenIfNeeded:
    """Test cases for refresh_token_if_needed function."""

    @patch.dict(
        os.environ,
        {
            "HUBSPOT_CLIENT_ID": "test_client_id",
            "HUBSPOT_CLIENT_SECRET": "test_client_secret",
        },
    )
    @patch("arklex.env.tools.hubspot.utils.mysql_pool")
    @patch("arklex.env.tools.hubspot.utils.requests.post")
    @patch("arklex.env.tools.hubspot.utils.log_context")
    def test_refresh_token_if_needed_token_not_expired(
        self, mock_log_context: Mock, mock_requests_post: Mock, mock_mysql_pool: Mock
    ) -> None:
        """Test refresh_token_if_needed when token is not expired."""
        # Setup
        bot_id = "test_bot_123"
        bot_version = "v1.0"
        # Set expiry time to be more than 15 minutes in the future to ensure early return
        # Use a properly formatted timezone-aware datetime
        from datetime import timezone

        future_time = datetime.now(timezone.utc) + timedelta(hours=2)
        hubspot_auth_tokens = HubspotAuthTokens(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expiry_time_str=future_time.isoformat(),
        )

        # Execute
        result = refresh_token_if_needed(bot_id, bot_version, hubspot_auth_tokens)

        # Assert - should return early without calling refresh
        assert result == hubspot_auth_tokens
        mock_requests_post.assert_not_called()
        mock_mysql_pool.execute.assert_not_called()

        mock_log_context.info.assert_called_once_with(
            f"Refreshing HubSpot auth tokens for bot {bot_id} version {bot_version}"
        )

    @patch.dict(
        os.environ,
        {
            "HUBSPOT_CLIENT_ID": "test_client_id",
            "HUBSPOT_CLIENT_SECRET": "test_client_secret",
        },
    )
    @patch("arklex.env.tools.hubspot.utils.mysql_pool")
    @patch("arklex.env.tools.hubspot.utils.requests.post")
    @patch("arklex.env.tools.hubspot.utils.log_context")
    def test_refresh_token_if_needed_token_expired_successful_refresh(
        self, mock_log_context: Mock, mock_requests_post: Mock, mock_mysql_pool: Mock
    ) -> None:
        """Test refresh_token_if_needed when token is expired and refresh succeeds."""
        # Setup
        bot_id = "test_bot_123"
        bot_version = "v1.0"
        past_time = datetime.now() - timedelta(hours=1)
        hubspot_auth_tokens = HubspotAuthTokens(
            access_token="old_access_token",
            refresh_token="test_refresh_token",
            expiry_time_str=past_time.isoformat() + "Z",
        )

        # Mock successful token refresh response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
        }
        mock_requests_post.return_value = mock_response

        # Execute
        result = refresh_token_if_needed(bot_id, bot_version, hubspot_auth_tokens)

        # Assert
        assert result.access_token == "new_access_token"
        assert result.refresh_token == "new_refresh_token"
        assert "Z" in result.expiry_time_str
        mock_requests_post.assert_called_once()
        mock_mysql_pool.execute.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "HUBSPOT_CLIENT_ID": "test_client_id",
            "HUBSPOT_CLIENT_SECRET": "test_client_secret",
        },
    )
    @patch("arklex.env.tools.hubspot.utils.mysql_pool")
    @patch("arklex.env.tools.hubspot.utils.requests.post")
    @patch("arklex.env.tools.hubspot.utils.log_context")
    def test_refresh_token_if_needed_token_expired_refresh_fails(
        self, mock_log_context: Mock, mock_requests_post: Mock, mock_mysql_pool: Mock
    ) -> None:
        """Test refresh_token_if_needed when token is expired but refresh fails."""
        # Setup
        bot_id = "test_bot_123"
        bot_version = "v1.0"
        past_time = datetime.now() - timedelta(hours=1)
        hubspot_auth_tokens = HubspotAuthTokens(
            access_token="old_access_token",
            refresh_token="test_refresh_token",
            expiry_time_str=past_time.isoformat() + "Z",
        )

        # Mock failed token refresh response
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.RequestException("API Error")
        )
        mock_requests_post.return_value = mock_response

        # Execute & Assert
        result = refresh_token_if_needed(bot_id, bot_version, hubspot_auth_tokens)
        assert result == hubspot_auth_tokens
        mock_requests_post.assert_called_once()
        mock_mysql_pool.execute.assert_not_called()

    @patch.dict(
        os.environ,
        {
            "HUBSPOT_CLIENT_ID": "test_client_id",
            "HUBSPOT_CLIENT_SECRET": "test_client_secret",
        },
    )
    @patch("arklex.env.tools.hubspot.utils.mysql_pool")
    @patch("arklex.env.tools.hubspot.utils.requests.post")
    @patch("arklex.env.tools.hubspot.utils.log_context")
    def test_refresh_token_if_needed_invalid_expiry_time(
        self, mock_log_context: Mock, mock_requests_post: Mock, mock_mysql_pool: Mock
    ) -> None:
        """Test refresh_token_if_needed with invalid expiry time format."""
        # Setup
        bot_id = "test_bot_123"
        bot_version = "v1.0"
        hubspot_auth_tokens = HubspotAuthTokens(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expiry_time_str="invalid_time_format",
        )

        # Mock successful token refresh response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
        }
        mock_requests_post.return_value = mock_response

        # Execute
        result = refresh_token_if_needed(bot_id, bot_version, hubspot_auth_tokens)

        # Assert
        assert result.access_token == "new_access_token"
        assert result.refresh_token == "new_refresh_token"
        mock_requests_post.assert_called_once()
        mock_mysql_pool.execute.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    def test_refresh_token_if_needed_missing_env_vars(self) -> None:
        """Test refresh_token_if_needed when environment variables are missing."""
        # Setup
        bot_id = "test_bot_123"
        bot_version = "v1.0"
        hubspot_auth_tokens = HubspotAuthTokens(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expiry_time_str="2024-01-01T00:00:00Z",
        )

        # Execute & Assert
        with pytest.raises(Exception) as exc_info:
            refresh_token_if_needed(bot_id, bot_version, hubspot_auth_tokens)

        assert "HubSpot client ID and secret not found in environment variables" in str(
            exc_info.value
        )

    @patch.dict(os.environ, {"HUBSPOT_CLIENT_ID": "test_client_id"})
    def test_refresh_token_if_needed_missing_client_secret(self) -> None:
        """Test refresh_token_if_needed when HUBSPOT_CLIENT_SECRET is missing."""
        # Setup
        bot_id = "test_bot_123"
        bot_version = "v1.0"
        hubspot_auth_tokens = HubspotAuthTokens(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expiry_time_str="2024-01-01T00:00:00Z",
        )

        # Execute & Assert
        with pytest.raises(Exception) as exc_info:
            refresh_token_if_needed(bot_id, bot_version, hubspot_auth_tokens)

        assert "HubSpot client ID and secret not found in environment variables" in str(
            exc_info.value
        )

    @patch.dict(os.environ, {"HUBSPOT_CLIENT_SECRET": "test_client_secret"})
    def test_refresh_token_if_needed_missing_client_id(self) -> None:
        """Test refresh_token_if_needed when HUBSPOT_CLIENT_ID is missing."""
        # Setup
        bot_id = "test_bot_123"
        bot_version = "v1.0"
        hubspot_auth_tokens = HubspotAuthTokens(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expiry_time_str="2024-01-01T00:00:00Z",
        )

        # Execute & Assert
        with pytest.raises(Exception) as exc_info:
            refresh_token_if_needed(bot_id, bot_version, hubspot_auth_tokens)

        assert "HubSpot client ID and secret not found in environment variables" in str(
            exc_info.value
        )

    @patch.dict(
        os.environ,
        {
            "HUBSPOT_CLIENT_ID": "test_client_id",
            "HUBSPOT_CLIENT_SECRET": "test_client_secret",
        },
    )
    @patch("arklex.env.tools.hubspot.utils.mysql_pool")
    @patch("arklex.env.tools.hubspot.utils.requests.post")
    @patch("arklex.env.tools.hubspot.utils.log_context")
    def test_refresh_token_if_needed_database_update(
        self, mock_log_context: Mock, mock_requests_post: Mock, mock_mysql_pool: Mock
    ) -> None:
        """Test refresh_token_if_needed database update with correct parameters."""
        # Setup
        bot_id = "test_bot_123"
        bot_version = "v1.0"
        past_time = datetime.now() - timedelta(hours=1)
        hubspot_auth_tokens = HubspotAuthTokens(
            access_token="old_access_token",
            refresh_token="test_refresh_token",
            expiry_time_str=past_time.isoformat() + "Z",
        )

        # Mock successful token refresh response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
        }
        mock_requests_post.return_value = mock_response

        # Execute
        refresh_token_if_needed(bot_id, bot_version, hubspot_auth_tokens)

        # Assert
        mock_mysql_pool.execute.assert_called_once()
        call_args = mock_mysql_pool.execute.call_args
        assert (
            call_args[0][0]
            == """
            UPDATE qa_bot_resource_permission 
            SET auth = %s 
            WHERE qa_bot_id = %s 
            AND qa_bot_version = %s 
            AND qa_bot_resource_auth_group_id = %s
            """
        )
        assert call_args[0][1][1] == bot_id
        assert call_args[0][1][2] == bot_version
        assert call_args[0][1][3] == ResourceAuthGroup.HUBSPOT.value

    @patch.dict(
        os.environ,
        {
            "HUBSPOT_CLIENT_ID": "test_client_id",
            "HUBSPOT_CLIENT_SECRET": "test_client_secret",
        },
    )
    @patch("arklex.env.tools.hubspot.utils.mysql_pool")
    @patch("arklex.env.tools.hubspot.utils.requests.post")
    @patch("arklex.env.tools.hubspot.utils.log_context")
    def test_refresh_token_if_needed_general_exception(
        self, mock_log_context: Mock, mock_requests_post: Mock, mock_mysql_pool: Mock
    ) -> None:
        """Test refresh_token_if_needed when a general exception occurs."""
        # Setup
        bot_id = "test_bot_123"
        bot_version = "v1.0"
        hubspot_auth_tokens = HubspotAuthTokens(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expiry_time_str="2024-01-01T00:00:00Z",
        )

        # Mock successful token refresh response to avoid validation errors
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
        }
        mock_requests_post.return_value = mock_response

        # Mock mysql_pool to raise an exception
        mock_mysql_pool.execute.side_effect = Exception("Database error")

        # Execute & Assert
        with pytest.raises(Exception) as exc_info:
            refresh_token_if_needed(bot_id, bot_version, hubspot_auth_tokens)

        assert "Failed to get HubSpot access token: Database error" in str(
            exc_info.value
        )

    @patch.dict(
        os.environ,
        {
            "HUBSPOT_CLIENT_ID": "test_client_id",
            "HUBSPOT_CLIENT_SECRET": "test_client_secret",
        },
    )
    @patch("arklex.env.tools.hubspot.utils.mysql_pool")
    @patch("arklex.env.tools.hubspot.utils.requests.post")
    @patch("arklex.env.tools.hubspot.utils.log_context")
    def test_refresh_token_if_needed_hubspot_not_integrated_error(
        self, mock_log_context: Mock, mock_requests_post: Mock, mock_mysql_pool: Mock
    ) -> None:
        """Test refresh_token_if_needed when HubspotNotIntegratedError is raised."""
        # Setup
        bot_id = "test_bot_123"
        bot_version = "v1.0"
        hubspot_auth_tokens = HubspotAuthTokens(
            access_token="test_access_token",
            refresh_token="test_refresh_token",
            expiry_time_str="2024-01-01T00:00:00Z",
        )

        # Mock successful token refresh response to avoid validation errors
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
        }
        mock_requests_post.return_value = mock_response

        # Mock mysql_pool to raise HubspotNotIntegratedError
        mock_mysql_pool.execute.side_effect = HubspotNotIntegratedError(
            "account", bot_id, bot_version
        )

        # Execute & Assert
        with pytest.raises(HubspotNotIntegratedError) as exc_info:
            refresh_token_if_needed(bot_id, bot_version, hubspot_auth_tokens)

        assert exc_info.value.bot_id == bot_id
        assert exc_info.value.bot_version == bot_version


class TestConstants:
    """Test cases for module constants."""

    def test_hubspot_auth_error_constant(self) -> None:
        """Test HUBSPOT_AUTH_ERROR constant value."""
        expected_error = "Missing some or all required hubspot authentication parameters: access_token. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'access_token': <hubspot_access_token>}"
        assert expected_error == HUBSPOT_AUTH_ERROR
