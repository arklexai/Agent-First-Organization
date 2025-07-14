"""Comprehensive tests for the APIClientService class.

This module provides comprehensive test coverage for the APIClientService class,
ensuring all functionality is properly tested including error handling and edge cases.
"""

from unittest.mock import Mock, patch

import httpx
import pytest

from arklex.orchestrator.NLU.entities.slot_entities import Slot
from arklex.orchestrator.NLU.services.api_service import (
    DEFAULT_TIMEOUT,
    HTTP_METHOD_POST,
    APIClientService,
)
from arklex.utils.exceptions import APIError, ValidationError


class TestAPIClientServiceInitialization:
    """Test APIClientService initialization."""

    def test_api_client_service_initialization_success(self) -> None:
        """Test successful initialization of APIClientService."""
        with patch("httpx.Client") as mock_client:
            service = APIClientService("https://api.example.com")

            assert service.base_url == "https://api.example.com"
            assert service.timeout == DEFAULT_TIMEOUT
            mock_client.assert_called_once_with(timeout=DEFAULT_TIMEOUT)

    def test_api_client_service_initialization_with_custom_timeout(self) -> None:
        """Test initialization with custom timeout."""
        with patch("httpx.Client") as mock_client:
            service = APIClientService("https://api.example.com", timeout=60)

            assert service.base_url == "https://api.example.com"
            assert service.timeout == 60
            mock_client.assert_called_once_with(timeout=60)

    def test_api_client_service_initialization_strips_trailing_slash(self) -> None:
        """Test that base_url trailing slash is stripped."""
        with patch("httpx.Client"):
            service = APIClientService("https://api.example.com/")

            assert service.base_url == "https://api.example.com"

    def test_api_client_service_initialization_empty_url(self) -> None:
        """Test initialization with empty URL raises ValidationError."""
        with pytest.raises(ValidationError, match="Base URL cannot be empty"):
            APIClientService("")

    def test_api_client_service_initialization_none_url(self) -> None:
        """Test initialization with None URL raises ValidationError."""
        with pytest.raises(ValidationError, match="Base URL cannot be empty"):
            APIClientService(None)

    def test_api_client_service_initialization_client_error(self) -> None:
        """Test initialization when httpx.Client fails."""
        with (
            patch("httpx.Client", side_effect=Exception("Client error")),
            pytest.raises(APIError, match="Failed to initialize API client"),
        ):
            APIClientService("https://api.example.com")


class TestAPIClientServiceMakeRequest:
    """Test the _make_request method."""

    @pytest.fixture
    def api_service(self) -> APIClientService:
        """Create an APIClientService instance for testing."""
        with patch("httpx.Client"):
            return APIClientService("https://api.example.com")

    def test_make_request_success(self, api_service: APIClientService) -> None:
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        api_service.client.request.return_value = mock_response

        result = api_service._make_request("/test", "POST", {"data": "test"})

        assert result == {"result": "success"}
        api_service.client.request.assert_called_once_with(
            "POST", "https://api.example.com/test", json={"data": "test"}
        )

    def test_make_request_without_data(self, api_service: APIClientService) -> None:
        """Test API request without data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        api_service.client.request.return_value = mock_response

        result = api_service._make_request("/test", "GET")

        assert result == {"result": "success"}
        api_service.client.request.assert_called_once_with(
            "GET", "https://api.example.com/test", json=None
        )

    def test_make_request_strips_endpoint_slash(
        self, api_service: APIClientService
    ) -> None:
        """Test that endpoint leading slash is stripped."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        api_service.client.request.return_value = mock_response

        api_service._make_request("//test", "GET")

        api_service.client.request.assert_called_once_with(
            "GET", "https://api.example.com/test", json=None
        )

    def test_make_request_http_error(self, api_service: APIClientService) -> None:
        """Test API request with HTTP error."""
        http_error = httpx.HTTPError("HTTP Error")
        http_error.response = Mock()
        http_error.response.status_code = 404

        api_service.client.request.side_effect = http_error

        with pytest.raises(APIError, match="API request failed"):
            api_service._make_request("/test", "GET")

    def test_make_request_http_error_no_response(
        self, api_service: APIClientService
    ) -> None:
        """Test API request with HTTP error but no response."""
        http_error = httpx.HTTPError("HTTP Error")
        http_error.response = None

        api_service.client.request.side_effect = http_error

        with pytest.raises(APIError, match="API request failed"):
            api_service._make_request("/test", "GET")

    def test_make_request_json_error(self, api_service: APIClientService) -> None:
        """Test API request with JSON parsing error."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")

        api_service.client.request.return_value = mock_response

        with pytest.raises(APIError, match="API request failed"):
            api_service._make_request("/test", "GET")

    def test_make_request_general_exception(
        self, api_service: APIClientService
    ) -> None:
        """Test API request with general exception."""
        api_service.client.request.side_effect = Exception("General error")

        with pytest.raises(APIError, match="API request failed"):
            api_service._make_request("/test", "GET")


class TestAPIClientServicePredictIntent:
    """Test the predict_intent method."""

    @pytest.fixture
    def api_service(self) -> APIClientService:
        """Create an APIClientService instance for testing."""
        with patch("httpx.Client"):
            return APIClientService("https://api.example.com")

    @patch("arklex.orchestrator.NLU.services.api_service.validate_intent_response")
    def test_predict_intent_success(
        self, mock_validate: Mock, api_service: APIClientService
    ) -> None:
        """Test successful intent prediction."""
        mock_validate.return_value = "booking_intent"

        with patch.object(api_service, "_make_request") as mock_request:
            mock_request.return_value = {
                "intent": "booking_intent",
                "idx2intents_mapping": {"0": "booking_intent"},
            }

            result = api_service.predict_intent(
                "I want to book a flight",
                {"booking_intent": [{"definition": "Book a flight"}]},
                "Previous conversation",
                {"model": "gpt-4"},
            )

            assert result == "booking_intent"
            mock_request.assert_called_once_with(
                "/nlu/predict",
                HTTP_METHOD_POST,
                {
                    "text": "I want to book a flight",
                    "intents": {"booking_intent": [{"definition": "Book a flight"}]},
                    "chat_history_str": "Previous conversation",
                    "model_config": {"model": "gpt-4"},
                },
            )

    @patch("arklex.orchestrator.NLU.services.api_service.validate_intent_response")
    def test_predict_intent_validation_error(
        self, mock_validate: Mock, api_service: APIClientService
    ) -> None:
        """Test intent prediction with validation error."""
        mock_validate.side_effect = ValidationError("Invalid response")

        with patch.object(api_service, "_make_request") as mock_request:
            mock_request.return_value = {"intent": "invalid", "idx2intents_mapping": {}}

            with pytest.raises(ValidationError, match="Invalid response"):
                api_service.predict_intent(
                    "test text", {"test_intent": []}, "chat history", {"model": "test"}
                )

    def test_predict_intent_api_error(self, api_service: APIClientService) -> None:
        """Test intent prediction with API error."""
        with patch.object(api_service, "_make_request") as mock_request:
            mock_request.side_effect = APIError("API Error")

            with pytest.raises(APIError, match="API Error"):
                api_service.predict_intent(
                    "test text", {"test_intent": []}, "chat history", {"model": "test"}
                )


class TestAPIClientServicePredictSlots:
    """Test the predict_slots method."""

    @pytest.fixture
    def api_service(self) -> APIClientService:
        """Create an APIClientService instance for testing."""
        with patch("httpx.Client"):
            return APIClientService("https://api.example.com")

    def test_predict_slots_success(self, api_service: APIClientService) -> None:
        """Test successful slot prediction."""
        slots = [Slot(name="date", type="string", description="Travel date")]

        with patch.object(api_service, "_make_request") as mock_request:
            mock_request.return_value = {
                "slots": [
                    {
                        "name": "date",
                        "type": "string",
                        "description": "Travel date",
                        "value": "2024-01-15",
                    }
                ]
            }

            result = api_service.predict_slots(
                "I want to travel on 2024-01-15", slots, {"model": "gpt-4"}
            )

            assert len(result) == 1
            assert result[0].name == "date"
            assert result[0].value == "2024-01-15"

            mock_request.assert_called_once_with(
                "/slotfill/predict",
                HTTP_METHOD_POST,
                {
                    "text": "I want to travel on 2024-01-15",
                    "slots": [slot.model_dump() for slot in slots],
                    "model_config": {"model": "gpt-4"},
                },
            )

    def test_predict_slots_empty_response(self, api_service: APIClientService) -> None:
        """Test slot prediction with empty response."""
        slots = [Slot(name="date", type="string", description="Travel date")]

        with patch.object(api_service, "_make_request") as mock_request:
            mock_request.return_value = {"slots": []}

            result = api_service.predict_slots("test text", slots, {"model": "test"})

            assert result == []

    def test_predict_slots_missing_slots_key(
        self, api_service: APIClientService
    ) -> None:
        """Test slot prediction with missing slots key in response."""
        slots = [Slot(name="date", type="string", description="Travel date")]

        with patch.object(api_service, "_make_request") as mock_request:
            mock_request.return_value = {}

            result = api_service.predict_slots("test text", slots, {"model": "test"})

            assert result == []

    def test_predict_slots_api_error(self, api_service: APIClientService) -> None:
        """Test slot prediction with API error."""
        slots = [Slot(name="date", type="string", description="Travel date")]

        with patch.object(api_service, "_make_request") as mock_request:
            mock_request.side_effect = APIError("API Error")

            with pytest.raises(APIError, match="API Error"):
                api_service.predict_slots("test text", slots, {"model": "test"})


class TestAPIClientServiceVerifySlots:
    """Test the verify_slots method."""

    @pytest.fixture
    def api_service(self) -> APIClientService:
        """Create an APIClientService instance for testing."""
        with patch("httpx.Client"):
            return APIClientService("https://api.example.com")

    def test_verify_slots_success(self, api_service: APIClientService) -> None:
        """Test successful slot verification."""
        slots = [
            Slot(
                name="date",
                type="string",
                description="Travel date",
                value="2024-01-15",
            )
        ]

        with patch.object(api_service, "_make_request") as mock_request:
            mock_request.return_value = {
                "verification_needed": True,
                "thought": "Date seems ambiguous",
            }

            result = api_service.verify_slots(
                "I want to travel on 2024-01-15", slots, {"model": "gpt-4"}
            )

            assert result == (True, "Date seems ambiguous")

            mock_request.assert_called_once_with(
                "/slotfill/verify",
                HTTP_METHOD_POST,
                {
                    "text": "I want to travel on 2024-01-15",
                    "slots": [slot.model_dump() for slot in slots],
                    "model_config": {"model": "gpt-4"},
                },
            )

    def test_verify_slots_no_verification_needed(
        self, api_service: APIClientService
    ) -> None:
        """Test slot verification when no verification is needed."""
        slots = [
            Slot(
                name="date",
                type="string",
                description="Travel date",
                value="2024-01-15",
            )
        ]

        with patch.object(api_service, "_make_request") as mock_request:
            mock_request.return_value = {
                "verification_needed": False,
                "thought": "Date is clear",
            }

            result = api_service.verify_slots(
                "I want to travel on 2024-01-15", slots, {"model": "gpt-4"}
            )

            assert result == (False, "Date is clear")

    def test_verify_slots_missing_fields(self, api_service: APIClientService) -> None:
        """Test slot verification with missing fields in response."""
        slots = [
            Slot(
                name="date",
                type="string",
                description="Travel date",
                value="2024-01-15",
            )
        ]

        with patch.object(api_service, "_make_request") as mock_request:
            mock_request.return_value = {}

            result = api_service.verify_slots("test text", slots, {"model": "test"})

            assert result == (False, "No need to verify")

    def test_verify_slots_api_error(self, api_service: APIClientService) -> None:
        """Test slot verification with API error."""
        slots = [
            Slot(
                name="date",
                type="string",
                description="Travel date",
                value="2024-01-15",
            )
        ]

        with patch.object(api_service, "_make_request") as mock_request:
            mock_request.side_effect = APIError("API Error")

            with pytest.raises(APIError, match="API Error"):
                api_service.verify_slots("test text", slots, {"model": "test"})


class TestAPIClientServiceClose:
    """Test the close method."""

    @pytest.fixture
    def api_service(self) -> APIClientService:
        """Create an APIClientService instance for testing."""
        with patch("httpx.Client"):
            return APIClientService("https://api.example.com")

    def test_close_success(self, api_service: APIClientService) -> None:
        """Test successful client closure."""
        api_service.close()
        api_service.client.close.assert_called_once()

    def test_close_with_exception(self, api_service: APIClientService) -> None:
        """Test client closure with exception."""
        api_service.client.close.side_effect = Exception("Close error")

        # Should not raise exception, just log error
        api_service.close()
        api_service.client.close.assert_called_once()


class TestAPIClientServiceConstants:
    """Test API service constants."""

    def test_default_timeout_constant(self) -> None:
        """Test DEFAULT_TIMEOUT constant."""
        from arklex.orchestrator.NLU.services.api_service import DEFAULT_TIMEOUT

        assert DEFAULT_TIMEOUT == 30

    def test_http_method_post_constant(self) -> None:
        """Test HTTP_METHOD_POST constant."""
        from arklex.orchestrator.NLU.services.api_service import HTTP_METHOD_POST

        assert HTTP_METHOD_POST == "POST"
