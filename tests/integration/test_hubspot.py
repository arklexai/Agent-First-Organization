"""
Integration tests for HubSpot tools.

This module contains comprehensive integration tests for all HubSpot tools,
including proper mocking of external services and edge case testing.
These tests validate the complete HubSpot integration workflow from API
calls to response processing and error handling.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from hubspot.crm.objects.emails import ApiException as EmailsApiException
from hubspot.crm.objects.meetings import ApiException as MeetingsApiException

from arklex.env.tools.hubspot._exception_prompt import HubspotExceptionPrompt
from arklex.env.tools.hubspot.check_available import check_available
from arklex.env.tools.hubspot.create_meeting import create_meeting
from arklex.env.tools.hubspot.create_ticket import create_ticket
from arklex.env.tools.hubspot.find_contact_by_email import find_contact_by_email
from arklex.env.tools.hubspot.find_owner_id_by_contact_id import (
    find_owner_id_by_contact_id,
)
from arklex.utils.exceptions import AuthenticationError, ToolExecutionError

# Extract underlying functions from decorated tool functions for direct testing
# This allows us to test the core functionality without the decorator wrapper
find_contact_by_email_func = find_contact_by_email().func
create_ticket_func = create_ticket().func
find_owner_id_by_contact_id_func = find_owner_id_by_contact_id().func
check_available_func = check_available().func
create_meeting_func = create_meeting().func


class TestHubSpotFindContactByEmail:
    """
    Integration tests for find_contact_by_email tool.

    This test class validates the contact search functionality, including
    successful searches, contact not found scenarios, API errors, and
    authentication failures.
    """

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_success(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test successful contact search by email with communication tracking.

        This test validates that the find_contact_by_email tool can successfully
        search for contacts by email address, create communication records,
        and associate them with the found contact.
        """
        # Mock authentication to return a valid token
        # This simulates successful HubSpot authentication
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Execute the find_contact_by_email function with test parameters
        # This tests the complete workflow from email to contact details
        result = find_contact_by_email_func(
            email="john.doe@example.com",
            chat="Hello, I have a question about your products",
            access_token="test_token",
        )

        # Parse the JSON result and validate the structure
        # This ensures the response format is correct
        expected_result = {
            "contact_id": "12345",
            "contact_email": "john.doe@example.com",
            "contact_first_name": "John",
            "contact_last_name": "Doe",
        }
        assert json.loads(result) == expected_result, (
            "Contact details should match expected format"
        )

        # Verify all expected API calls were made
        # This ensures the tool performs all required operations
        mock_hubspot_client.crm.contacts.search_api.do_search.assert_called_once()
        mock_hubspot_client.crm.objects.communications.basic_api.create.assert_called_once()
        mock_hubspot_client.crm.associations.v4.basic_api.create.assert_called_once()

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_not_found(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test contact search when no matching contact exists.

        This test validates that the find_contact_by_email tool properly handles
        cases where the email address doesn't match any existing contacts
        and raises an appropriate error with a meaningful message.
        """
        # Mock authentication to return a valid token
        # This simulates successful HubSpot authentication
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock empty search results
        # This simulates a valid API response with no matching contacts
        mock_search_response = MagicMock()
        mock_search_response.to_dict.return_value = {"total": 0, "results": []}
        mock_hubspot_client.crm.contacts.search_api.do_search.return_value = (
            mock_search_response
        )

        # Test that the function raises an appropriate error
        # This validates the error handling for no matching contacts
        with pytest.raises(ToolExecutionError) as exc_info:
            find_contact_by_email_func(
                email="nonexistent@example.com", chat="Hello", access_token="test_token"
            )

        # Verify the error message contains the expected prompt
        # This ensures the error message is informative and consistent
        assert HubspotExceptionPrompt.USER_NOT_FOUND_PROMPT in str(exc_info.value), (
            "Error message should contain user not found prompt"
        )

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_api_exception(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test contact search when HubSpot API throws an exception.

        This test validates that the find_contact_by_email tool properly handles
        API errors and exceptions, ensuring that failures are caught and
        appropriate error messages are provided to the user.
        """
        # Mock authentication to return a valid token
        # This simulates successful HubSpot authentication
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Simulate API error by making search method raise an exception
        # This simulates a network error, rate limiting, or other API issue
        mock_hubspot_client.crm.contacts.search_api.do_search.side_effect = (
            EmailsApiException(status=400)
        )

        # Test that the function raises an appropriate error
        # This validates the error handling for API exceptions
        with pytest.raises(ToolExecutionError) as exc_info:
            find_contact_by_email_func(
                email="test@example.com", chat="Hello", access_token="test_token"
            )

        # Verify the error message contains the expected prompt
        # This ensures the error message is informative and consistent
        assert HubspotExceptionPrompt.USER_NOT_FOUND_PROMPT in str(exc_info.value), (
            "Error message should contain user not found prompt"
        )

    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_authentication_error(
        self, mock_authenticate: Mock
    ) -> None:
        """
        Test contact search with invalid or missing access token.

        This test validates that the find_contact_by_email tool properly handles
        authentication failures and provides appropriate error messages
        when the access token is invalid or missing.
        """
        # Mock authentication to raise an authentication error
        # This simulates an invalid or missing access token
        mock_authenticate.side_effect = AuthenticationError("Missing access token")

        # Test that the function raises an authentication error
        # This validates the authentication error handling
        with pytest.raises(AuthenticationError):
            find_contact_by_email_func(
                email="test@example.com", chat="Hello", access_token=""
            )

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_multiple_contacts_found(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test contact search when multiple contacts are found with the same email.

        This test validates that the find_contact_by_email tool properly handles
        cases where multiple contacts exist with the same email address,
        which should be treated as an error since we expect unique email addresses.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock search results with multiple contacts
        mock_search_response = MagicMock()
        mock_search_response.to_dict.return_value = {
            "total": 2,
            "results": [
                {"id": "12345", "properties": {"firstname": "John", "lastname": "Doe"}},
                {"id": "67890", "properties": {"firstname": "Jane", "lastname": "Doe"}},
            ],
        }
        mock_hubspot_client.crm.contacts.search_api.do_search.return_value = (
            mock_search_response
        )

        # Test that the function raises an appropriate error
        with pytest.raises(ToolExecutionError) as exc_info:
            find_contact_by_email_func(
                email="duplicate@example.com", chat="Hello", access_token="test_token"
            )

        # Verify the error message contains the expected prompt
        assert HubspotExceptionPrompt.USER_NOT_FOUND_PROMPT in str(exc_info.value), (
            "Error message should contain user not found prompt for multiple contacts"
        )

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_communication_creation_failure(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test contact search when communication creation fails but contact is found.

        This test validates that the find_contact_by_email tool gracefully handles
        communication creation failures and still returns contact information
        even when the communication tracking fails.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful contact search
        mock_search_response = MagicMock()
        mock_search_response.to_dict.return_value = {
            "total": 1,
            "results": [
                {
                    "id": "12345",
                    "properties": {"firstname": "John", "lastname": "Doe"},
                }
            ],
        }
        mock_hubspot_client.crm.contacts.search_api.do_search.return_value = (
            mock_search_response
        )

        # Mock communication creation to fail
        mock_hubspot_client.crm.objects.communications.basic_api.create.side_effect = (
            EmailsApiException(status=400)
        )

        # Execute the function - should still succeed despite communication failure
        result = find_contact_by_email_func(
            email="john.doe@example.com",
            chat="Hello, I have a question",
            access_token="test_token",
        )

        # Verify contact information is still returned
        expected_result = {
            "contact_id": "12345",
            "contact_email": "john.doe@example.com",
            "contact_first_name": "John",
            "contact_last_name": "Doe",
        }
        assert json.loads(result) == expected_result, (
            "Contact details should be returned even when communication creation fails"
        )

        # Verify search was called but communication creation failed
        mock_hubspot_client.crm.contacts.search_api.do_search.assert_called_once()
        mock_hubspot_client.crm.objects.communications.basic_api.create.assert_called_once()

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_association_creation_failure(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test contact search when association creation fails but contact and communication are created.

        This test validates that the find_contact_by_email tool gracefully handles
        association creation failures and still returns contact information
        even when the association between contact and communication fails.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful contact search
        mock_search_response = MagicMock()
        mock_search_response.to_dict.return_value = {
            "total": 1,
            "results": [
                {
                    "id": "12345",
                    "properties": {"firstname": "John", "lastname": "Doe"},
                }
            ],
        }
        mock_hubspot_client.crm.contacts.search_api.do_search.return_value = (
            mock_search_response
        )

        # Mock successful communication creation
        mock_communication_response = MagicMock()
        mock_communication_response.to_dict.return_value = {"id": "comm_123"}
        mock_hubspot_client.crm.objects.communications.basic_api.create.return_value = (
            mock_communication_response
        )

        # Mock association creation to fail
        mock_hubspot_client.crm.associations.v4.basic_api.create.side_effect = (
            EmailsApiException(status=400)
        )

        # Execute the function - should still succeed despite association failure
        result = find_contact_by_email_func(
            email="john.doe@example.com",
            chat="Hello, I have a question",
            access_token="test_token",
        )

        # Verify contact information is still returned
        expected_result = {
            "contact_id": "12345",
            "contact_email": "john.doe@example.com",
            "contact_first_name": "John",
            "contact_last_name": "Doe",
        }
        assert json.loads(result) == expected_result, (
            "Contact details should be returned even when association creation fails"
        )

        # Verify all API calls were made
        mock_hubspot_client.crm.contacts.search_api.do_search.assert_called_once()
        mock_hubspot_client.crm.objects.communications.basic_api.create.assert_called_once()
        mock_hubspot_client.crm.associations.v4.basic_api.create.assert_called_once()

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_missing_contact_properties(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test contact search when contact properties are missing or null.

        This test validates that the find_contact_by_email tool properly handles
        cases where contact properties like firstname or lastname are missing,
        ensuring the function doesn't fail and returns None for missing properties.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful contact search with missing properties
        mock_search_response = MagicMock()
        mock_search_response.to_dict.return_value = {
            "total": 1,
            "results": [
                {
                    "id": "12345",
                    "properties": {},  # Empty properties
                }
            ],
        }
        mock_hubspot_client.crm.contacts.search_api.do_search.return_value = (
            mock_search_response
        )

        # Mock successful communication creation
        mock_communication_response = MagicMock()
        mock_communication_response.to_dict.return_value = {"id": "comm_123"}
        mock_hubspot_client.crm.objects.communications.basic_api.create.return_value = (
            mock_communication_response
        )

        # Mock successful association creation
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        # Execute the function
        result = find_contact_by_email_func(
            email="john.doe@example.com",
            chat="Hello, I have a question",
            access_token="test_token",
        )

        # Verify contact information is returned with None for missing properties
        expected_result = {
            "contact_id": "12345",
            "contact_email": "john.doe@example.com",
            "contact_first_name": None,
            "contact_last_name": None,
        }
        assert json.loads(result) == expected_result, (
            "Contact details should be returned with None for missing properties"
        )

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_empty_chat_message(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test contact search with empty chat message.

        This test validates that the find_contact_by_email tool properly handles
        empty chat messages and still creates communication records with empty content.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful contact search
        mock_search_response = MagicMock()
        mock_search_response.to_dict.return_value = {
            "total": 1,
            "results": [
                {
                    "id": "12345",
                    "properties": {"firstname": "John", "lastname": "Doe"},
                }
            ],
        }
        mock_hubspot_client.crm.contacts.search_api.do_search.return_value = (
            mock_search_response
        )

        # Mock successful communication creation
        mock_communication_response = MagicMock()
        mock_communication_response.to_dict.return_value = {"id": "comm_123"}
        mock_hubspot_client.crm.objects.communications.basic_api.create.return_value = (
            mock_communication_response
        )

        # Mock successful association creation
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        # Execute the function with empty chat message
        result = find_contact_by_email_func(
            email="john.doe@example.com",
            chat="",  # Empty chat message
            access_token="test_token",
        )

        # Verify contact information is still returned
        expected_result = {
            "contact_id": "12345",
            "contact_email": "john.doe@example.com",
            "contact_first_name": "John",
            "contact_last_name": "Doe",
        }
        assert json.loads(result) == expected_result, (
            "Contact details should be returned even with empty chat message"
        )

        # Verify communication was created with empty body
        mock_hubspot_client.crm.objects.communications.basic_api.create.assert_called_once()
        call_args = (
            mock_hubspot_client.crm.objects.communications.basic_api.create.call_args
        )
        communication_data = call_args[0][0]
        assert communication_data.properties["hs_communication_body"] == "", (
            "Communication should be created with empty body"
        )

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_special_characters_in_email(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test contact search with special characters in email address.

        This test validates that the find_contact_by_email tool properly handles
        email addresses containing special characters like plus signs, dots, etc.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful contact search
        mock_search_response = MagicMock()
        mock_search_response.to_dict.return_value = {
            "total": 1,
            "results": [
                {
                    "id": "12345",
                    "properties": {"firstname": "John", "lastname": "Doe"},
                }
            ],
        }
        mock_hubspot_client.crm.contacts.search_api.do_search.return_value = (
            mock_search_response
        )

        # Mock successful communication creation
        mock_communication_response = MagicMock()
        mock_communication_response.to_dict.return_value = {"id": "comm_123"}
        mock_hubspot_client.crm.objects.communications.basic_api.create.return_value = (
            mock_communication_response
        )

        # Mock successful association creation
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        # Execute the function with special characters in email
        special_email = "john.doe+test@example.com"
        result = find_contact_by_email_func(
            email=special_email,
            chat="Hello, I have a question",
            access_token="test_token",
        )

        # Verify contact information is returned
        expected_result = {
            "contact_id": "12345",
            "contact_email": special_email,
            "contact_first_name": "John",
            "contact_last_name": "Doe",
        }
        assert json.loads(result) == expected_result, (
            "Contact details should be returned for email with special characters"
        )

        # Verify search was called with the exact email
        mock_hubspot_client.crm.contacts.search_api.do_search.assert_called_once()
        call_args = mock_hubspot_client.crm.contacts.search_api.do_search.call_args
        search_request = call_args[1]["public_object_search_request"]
        filter_value = search_request.filter_groups[0]["filters"][0]["value"]
        assert filter_value == special_email, (
            "Search should be called with exact email including special characters"
        )

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_long_chat_message(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test contact search with a very long chat message.

        This test validates that the find_contact_by_email tool properly handles
        long chat messages and creates communication records with the full content.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful contact search
        mock_search_response = MagicMock()
        mock_search_response.to_dict.return_value = {
            "total": 1,
            "results": [
                {
                    "id": "12345",
                    "properties": {"firstname": "John", "lastname": "Doe"},
                }
            ],
        }
        mock_hubspot_client.crm.contacts.search_api.do_search.return_value = (
            mock_search_response
        )

        # Mock successful communication creation
        mock_communication_response = MagicMock()
        mock_communication_response.to_dict.return_value = {"id": "comm_123"}
        mock_hubspot_client.crm.objects.communications.basic_api.create.return_value = (
            mock_communication_response
        )

        # Mock successful association creation
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        # Create a long chat message
        long_chat = (
            "This is a very long chat message that contains a lot of text. " * 50
        )

        # Execute the function with long chat message
        result = find_contact_by_email_func(
            email="john.doe@example.com",
            chat=long_chat,
            access_token="test_token",
        )

        # Verify contact information is returned
        expected_result = {
            "contact_id": "12345",
            "contact_email": "john.doe@example.com",
            "contact_first_name": "John",
            "contact_last_name": "Doe",
        }
        assert json.loads(result) == expected_result, (
            "Contact details should be returned even with long chat message"
        )

        # Verify communication was created with the full long message
        mock_hubspot_client.crm.objects.communications.basic_api.create.assert_called_once()
        call_args = (
            mock_hubspot_client.crm.objects.communications.basic_api.create.call_args
        )
        communication_data = call_args[0][0]
        assert communication_data.properties["hs_communication_body"] == long_chat, (
            "Communication should be created with the full long message"
        )

    @patch("arklex.env.tools.hubspot.find_contact_by_email.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_unicode_characters(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test contact search with unicode characters in chat message.

        This test validates that the find_contact_by_email tool properly handles
        unicode characters in chat messages and creates communication records correctly.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful contact search
        mock_search_response = MagicMock()
        mock_search_response.to_dict.return_value = {
            "total": 1,
            "results": [
                {
                    "id": "12345",
                    "properties": {"firstname": "John", "lastname": "Doe"},
                }
            ],
        }
        mock_hubspot_client.crm.contacts.search_api.do_search.return_value = (
            mock_search_response
        )

        # Mock successful communication creation
        mock_communication_response = MagicMock()
        mock_communication_response.to_dict.return_value = {"id": "comm_123"}
        mock_hubspot_client.crm.objects.communications.basic_api.create.return_value = (
            mock_communication_response
        )

        # Mock successful association creation
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        # Create chat message with unicode characters
        unicode_chat = "Hello! I have a question about your products. 🚀 你好！"

        # Execute the function with unicode characters
        result = find_contact_by_email_func(
            email="john.doe@example.com",
            chat=unicode_chat,
            access_token="test_token",
        )

        # Verify contact information is returned
        expected_result = {
            "contact_id": "12345",
            "contact_email": "john.doe@example.com",
            "contact_first_name": "John",
            "contact_last_name": "Doe",
        }
        assert json.loads(result) == expected_result, (
            "Contact details should be returned even with unicode characters"
        )

        # Verify communication was created with unicode content
        mock_hubspot_client.crm.objects.communications.basic_api.create.assert_called_once()
        call_args = (
            mock_hubspot_client.crm.objects.communications.basic_api.create.call_args
        )
        communication_data = call_args[0][0]
        assert communication_data.properties["hs_communication_body"] == unicode_chat, (
            "Communication should be created with unicode characters preserved"
        )


class TestHubSpotCreateTicket:
    """
    Integration tests for create_ticket tool.

    This test class validates the ticket creation functionality, including
    successful ticket creation, contact association, and error handling.
    """

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_success(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_ticket_client: MagicMock,
    ) -> None:
        """
        Test successful ticket creation with contact association.

        This test validates that the create_ticket tool can successfully
        create a support ticket in HubSpot and associate it with the
        specified customer contact.
        """
        # Mock authentication to return a valid token
        # This simulates successful HubSpot authentication
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_ticket_client

        # Execute the create_ticket function with test parameters
        # This tests the complete workflow from issue description to ticket creation
        result = create_ticket_func(
            cus_cid="12345",
            issue="I need help with my order",
            access_token="test_token",
        )

        # Verify the ticket ID is returned
        # This ensures the ticket was created successfully
        expected_result = "ticket_123"
        assert result == expected_result, "Should return the created ticket ID"

        # Verify ticket creation and association calls were made
        # This ensures the tool performs all required operations
        mock_hubspot_ticket_client.crm.tickets.basic_api.create.assert_called_once()
        mock_hubspot_ticket_client.crm.associations.v4.basic_api.create.assert_called_once()

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_api_exception_during_creation(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """
        Test ticket creation when HubSpot API throws an exception during ticket creation.

        This test validates that the create_ticket tool properly handles
        API errors during ticket creation and provides appropriate error messages.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Simulate API error during ticket creation
        from hubspot.crm.objects.emails import ApiException

        mock_hubspot_client.crm.tickets.basic_api.create.side_effect = ApiException(
            status=400, reason="Bad Request"
        )

        # Test that the function raises an appropriate error
        with pytest.raises(ToolExecutionError) as exc_info:
            create_ticket_func(
                cus_cid="12345",
                issue="I need help with my order",
                access_token="test_token",
            )

        # Verify the error message contains expected content
        assert HubspotExceptionPrompt.TICKET_CREATION_ERROR_PROMPT in str(
            exc_info.value
        ), "Error message should contain ticket creation error prompt"

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_api_exception_during_association(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """
        Test ticket creation when HubSpot API throws an exception during association.

        This test validates that the create_ticket tool properly handles
        API errors during contact-ticket association and provides appropriate error messages.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful ticket creation
        mock_ticket_response = MagicMock()
        mock_ticket_response.to_dict.return_value = {"id": "ticket_123"}
        mock_hubspot_client.crm.tickets.basic_api.create.return_value = (
            mock_ticket_response
        )

        # Simulate API error during association
        from hubspot.crm.objects.emails import ApiException

        mock_hubspot_client.crm.associations.v4.basic_api.create.side_effect = (
            ApiException(status=400, reason="Association Failed")
        )

        # Test that the function raises an appropriate error
        with pytest.raises(ToolExecutionError) as exc_info:
            create_ticket_func(
                cus_cid="12345",
                issue="I need help with my order",
                access_token="test_token",
            )

        # Verify the error message contains expected content
        assert HubspotExceptionPrompt.TICKET_CREATION_ERROR_PROMPT in str(
            exc_info.value
        ), "Error message should contain ticket creation error prompt"

    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_authentication_error(self, mock_authenticate: Mock) -> None:
        """
        Test ticket creation with invalid or missing access token.

        This test validates that the create_ticket tool properly handles
        authentication failures and provides appropriate error messages.
        """
        # Mock authentication to raise an authentication error
        mock_authenticate.side_effect = AuthenticationError("Invalid access token")

        # Test that the function raises an authentication error
        with pytest.raises(AuthenticationError):
            create_ticket_func(
                cus_cid="12345",
                issue="I need help with my order",
                access_token="invalid_token",
            )

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_with_long_issue_description(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """
        Test ticket creation with a very long issue description.

        This test validates that the create_ticket tool can handle
        lengthy issue descriptions without truncation or errors.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful ticket creation
        mock_ticket_response = MagicMock()
        mock_ticket_response.to_dict.return_value = {"id": "ticket_456"}
        mock_hubspot_client.crm.tickets.basic_api.create.return_value = (
            mock_ticket_response
        )
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        # Create a very long issue description
        long_issue = (
            "I have been experiencing problems with my order for the past three weeks. "
            * 10
        )

        result = create_ticket_func(
            cus_cid="12345",
            issue=long_issue,
            access_token="test_token",
        )

        assert result == "ticket_456", "Should return the created ticket ID"

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_with_special_characters_in_issue(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """
        Test ticket creation with special characters in issue description.

        This test validates that the create_ticket tool can handle
        special characters, emojis, and unicode in issue descriptions.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful ticket creation
        mock_ticket_response = MagicMock()
        mock_ticket_response.to_dict.return_value = {"id": "ticket_789"}
        mock_hubspot_client.crm.tickets.basic_api.create.return_value = (
            mock_ticket_response
        )
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        # Issue with special characters, emojis, and unicode
        special_issue = "I need help with my order! 🚀 The product has issues: émojis & symbols @#$%^&*()"

        result = create_ticket_func(
            cus_cid="12345",
            issue=special_issue,
            access_token="test_token",
        )

        assert result == "ticket_789", "Should return the created ticket ID"

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_with_numeric_customer_id(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """
        Test ticket creation with numeric customer ID as specified in documentation.

        This test validates that the create_ticket tool works correctly
        with numeric customer IDs as described in the slots documentation.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful ticket creation
        mock_ticket_response = MagicMock()
        mock_ticket_response.to_dict.return_value = {"id": "ticket_999"}
        mock_hubspot_client.crm.tickets.basic_api.create.return_value = (
            mock_ticket_response
        )
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        # Use numeric customer ID as specified in documentation
        numeric_customer_id = "97530152525"

        result = create_ticket_func(
            cus_cid=numeric_customer_id,
            issue="I need help with my order",
            access_token="test_token",
        )

        assert result == "ticket_999", "Should return the created ticket ID"

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_timestamp_format_validation(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """
        Test that ticket creation generates proper timestamp format in subject.

        This test validates that the timestamp in the subject line
        follows the expected ISO 8601 format with microsecond precision.
        """
        # Mock authentication to return a valid token
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful ticket creation
        mock_ticket_response = MagicMock()
        mock_ticket_response.to_dict.return_value = {"id": "ticket_timestamp"}
        mock_hubspot_client.crm.tickets.basic_api.create.return_value = (
            mock_ticket_response
        )
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        create_ticket_func(
            cus_cid="12345",
            issue="I need help with my order",
            access_token="test_token",
        )

        # Verify that the ticket creation was called with proper subject format
        create_call = mock_hubspot_client.crm.tickets.basic_api.create.call_args
        ticket_input = create_call[1]["simple_public_object_input_for_create"]
        properties = ticket_input.properties

        # Check that subject contains customer ID and timestamp
        assert "subject" in properties, "Subject should be set in ticket properties"
        subject = properties["subject"]
        assert "Issue of 12345 at" in subject, (
            "Subject should contain customer ID and timestamp"
        )
        assert "T" in subject and "Z" in subject, (
            "Subject should contain ISO 8601 timestamp"
        )


class TestHubSpotFindOwnerIdByContactId:
    """
    Integration tests for find_owner_id_by_contact_id tool.

    This test class validates the owner ID lookup functionality, including
    successful lookups, API errors, and edge cases.
    """

    @pytest.fixture
    def mock_hubspot_client(self) -> MagicMock:
        """
        Create a mock HubSpot client with owner ID lookup responses.

        Returns:
            MagicMock: A mock HubSpot client configured for owner ID testing.

        This fixture provides a mock HubSpot client that simulates owner ID
        lookup operations with realistic responses.
        """
        mock_client = MagicMock()

        # Mock successful owner ID retrieval
        # This simulates a successful API response with owner information
        mock_response = MagicMock()
        mock_response.json.return_value = {"properties": {"hubspot_owner_id": "67890"}}
        mock_client.api_request.return_value = mock_response

        return mock_client

    @patch("arklex.env.tools.hubspot.find_owner_id_by_contact_id.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_owner_id_by_contact_id.authenticate_hubspot")
    def test_find_owner_id_by_contact_id_success(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test successful owner ID retrieval for a contact.

        This test validates that the find_owner_id_by_contact_id tool can
        successfully retrieve the owner ID associated with a specific contact
        from the HubSpot CRM.
        """
        # Mock authentication to return a valid token
        # This simulates successful HubSpot authentication
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Execute the find_owner_id_by_contact_id function with test parameters
        # This tests the complete workflow from contact ID to owner ID
        result = find_owner_id_by_contact_id_func(
            cus_cid="12345", access_token="test_token"
        )

        # Verify the owner ID is returned correctly
        # This ensures the owner lookup worked correctly
        assert result == "67890", "Should return the correct owner ID"
        mock_hubspot_client.api_request.assert_called_once()

    @patch("arklex.env.tools.hubspot.find_owner_id_by_contact_id.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.find_owner_id_by_contact_id.authenticate_hubspot")
    def test_find_owner_id_by_contact_id_api_exception(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """
        Test owner ID retrieval when HubSpot API throws an exception.

        This test validates that the find_owner_id_by_contact_id tool properly
        handles API errors and exceptions, ensuring that failures are caught
        and appropriate error messages are provided.
        """
        # Mock authentication to return a valid token
        # This simulates successful HubSpot authentication
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Simulate API error by making api_request method raise an exception
        # This simulates a network error, rate limiting, or other API issue
        from hubspot.crm.objects.emails import ApiException

        mock_hubspot_client.api_request.side_effect = ApiException("API Error")

        # Test that the function raises an appropriate error
        # This validates the error handling for API exceptions
        with pytest.raises(ToolExecutionError) as exc_info:
            find_owner_id_by_contact_id_func(cus_cid="12345", access_token="test_token")

        # Verify the error message contains expected content
        # This ensures the error message is informative
        assert "Tool find_owner_id_by_contact_id execution failed" in str(
            exc_info.value
        ), "Error message should indicate tool execution failure"


class TestHubSpotCheckAvailable:
    """Integration tests for check_available tool."""

    @pytest.fixture
    def mock_hubspot_client(self) -> MagicMock:
        """Create a mock HubSpot client with meeting availability responses."""
        mock_client = MagicMock()

        # Mock meeting links response
        mock_links_response = MagicMock()
        mock_links_response.json.return_value = {
            "status": "success",
            "results": [{"organizerUserId": "67890", "slug": "veronica-chen"}],
        }
        mock_client.api_request.return_value = mock_links_response

        return mock_client

    @patch("arklex.env.tools.hubspot.check_available.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.check_available.authenticate_hubspot")
    def test_check_available_success(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test successful availability check with meeting links and time slots."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock availability response with 15-minute slots
        mock_availability_response = MagicMock()
        mock_availability_response.json.return_value = {
            "linkAvailability": {
                "linkAvailabilityByDuration": {
                    "900000": {  # 15 minutes in milliseconds
                        "availabilities": [
                            {
                                "startMillisUtc": 1640995200000,  # Example timestamp
                                "endMillisUtc": 1640996100000,
                            }
                        ]
                    }
                }
            }
        }

        # Configure mock to return different responses for different calls
        mock_hubspot_client.api_request.side_effect = [
            mock_hubspot_client.api_request.return_value,  # First call (meeting links)
            mock_availability_response,  # Second call (availability)
        ]

        result = check_available_func(
            owner_id=67890,
            time_zone="America/New_York",
            meeting_date="tomorrow",
            duration=15,
            access_token="test_token",
        )

        assert "veronica-chen" in result
        assert "available times for other dates" in result
        assert mock_hubspot_client.api_request.call_count == 2

    @patch("arklex.env.tools.hubspot.check_available.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.check_available.authenticate_hubspot")
    def test_check_available_no_meeting_links(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test availability check when no meeting links are found for the owner."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock empty meeting links response
        mock_links_response = MagicMock()
        mock_links_response.json.return_value = {"status": "success", "results": []}
        mock_hubspot_client.api_request.return_value = mock_links_response

        with pytest.raises(
            ToolExecutionError, match=HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
        ):
            check_available_func(
                owner_id=67890,
                time_zone="America/New_York",
                meeting_date="tomorrow",
                duration=15,
                access_token="test_token",
            )

    @patch("arklex.env.tools.hubspot.check_available.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.check_available.authenticate_hubspot")
    def test_check_available_api_error(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test availability check when HubSpot API returns an error response."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock error response from API
        mock_links_response = MagicMock()
        mock_links_response.json.return_value = {
            "status": "error",
            "message": "API error",
        }
        mock_hubspot_client.api_request.return_value = mock_links_response

        with pytest.raises(
            ToolExecutionError, match=HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
        ):
            check_available_func(
                owner_id=67890,
                time_zone="America/New_York",
                meeting_date="tomorrow",
                duration=15,
                access_token="test_token",
            )


class TestHubSpotCreateMeeting:
    """Integration tests for create_meeting tool."""

    @pytest.fixture
    def mock_hubspot_client(self) -> MagicMock:
        """Create a mock HubSpot client with meeting creation responses."""
        mock_client = MagicMock()

        # Mock successful meeting creation
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
            "endTime": "2024-01-15T10:15:00Z",
            "subject": "Meeting with John Doe",
        }
        mock_client.api_request.return_value = mock_meeting_response

        return mock_client

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_success(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test successful meeting creation with contact details."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=15,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"
        assert meeting_data["subject"] == "Meeting with John Doe"
        mock_hubspot_client.api_request.assert_called_once()

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_api_exception(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation when HubSpot API throws an exception."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Simulate API error during meeting creation
        mock_hubspot_client.api_request.side_effect = MeetingsApiException(status=400)

        with pytest.raises(ToolExecutionError) as exc_info:
            create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time="10:00 AM",
                duration=15,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="test_token",
            )
        assert HubspotExceptionPrompt.MEETING_UNAVAILABLE_PROMPT in str(exc_info.value)

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_iso8601_time(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with ISO8601 formatted time string."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="2024-01-15T10:00:00Z",
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"
        mock_hubspot_client.api_request.assert_called_once()

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_different_timezones(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """Test meeting creation with various timezone configurations."""
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
            "endTime": "2024-01-15T10:30:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test with different timezones
        timezones = [
            "America/New_York",
            "America/Los_Angeles",
            "Asia/Tokyo",
            "Europe/London",
        ]

        for timezone in timezones:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time="10:00 AM",
                duration=30,
                slug="veronica-chen",
                time_zone=timezone,
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_different_durations(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with different duration values (15, 30, 60 minutes)."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
            "endTime": "2024-01-15T10:15:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test different durations
        durations = [15, 30, 60]
        for duration in durations:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time="10:00 AM",
                duration=duration,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

            # Verify the duration was converted to milliseconds correctly
            call_args = mock_hubspot_client.api_request.call_args[0][0]
            expected_duration_ms = duration * 60 * 1000  # minutes to milliseconds
            assert call_args["body"]["duration"] == expected_duration_ms

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_natural_language_dates(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with various natural language date inputs."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test different natural language date formats
        date_inputs = [
            "tomorrow",
            "today",
            "next Monday",
            "May 1st",
            "December 25th",
            "next week",
        ]

        for date_input in date_inputs:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date=date_input,
                meeting_start_time="10:00 AM",
                duration=30,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_natural_language_times(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with various natural language time inputs."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test different natural language time formats
        time_inputs = [
            "10:00 AM",
            "2:30 PM",
            "9am",
            "3pm",
            "noon",
            "midnight",
        ]

        for time_input in time_inputs:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time=time_input,
                duration=30,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_iso8601_time_with_timezone(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with ISO8601 time that includes timezone information."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T15:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="2024-01-15T10:00:00-05:00",  # EST timezone
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_iso8601_time_without_timezone(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with ISO8601 time without timezone (should use provided timezone)."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T15:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="2024-01-15T10:00:00",  # No timezone
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_authentication_error(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """Test meeting creation when authentication fails."""
        mock_authenticate.side_effect = AuthenticationError("Invalid access token")

        with pytest.raises(AuthenticationError):
            create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time="10:00 AM",
                duration=30,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="invalid_token",
            )

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_special_characters_in_names(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with special characters in customer names."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        result = create_meeting_func(
            cus_fname="José",
            cus_lname="O'Connor",
            cus_email="jose.oconnor@example.com",
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"

        # Verify the special characters were passed correctly
        call_args = mock_hubspot_client.api_request.call_args[0][0]
        assert call_args["body"]["firstName"] == "José"
        assert call_args["body"]["lastName"] == "O'Connor"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_long_email(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with a very long email address."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        long_email = (
            "very.long.email.address.with.many.subdomains@very.long.domain.name.com"
        )

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email=long_email,
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"

        # Verify the long email was passed correctly
        call_args = mock_hubspot_client.api_request.call_args[0][0]
        assert call_args["body"]["email"] == long_email

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_complex_slug(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with complex slug containing special characters."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        complex_slug = "dr-maria-garcia-phd"

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=30,
            slug=complex_slug,
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"

        # Verify the complex slug was passed correctly
        call_args = mock_hubspot_client.api_request.call_args[0][0]
        assert call_args["body"]["slug"] == complex_slug

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_api_request_structure(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test that the API request structure is correct."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        # Verify the API request structure
        call_args = mock_hubspot_client.api_request.call_args[0][0]

        # Check path and method
        assert call_args["path"] == "/scheduler/v3/meetings/meeting-links/book"
        assert call_args["method"] == "POST"

        # Check body structure
        body = call_args["body"]
        assert "slug" in body
        assert "duration" in body
        assert "startTime" in body
        assert "email" in body
        assert "firstName" in body
        assert "lastName" in body
        assert "timezone" in body
        assert "locale" in body

        # Check query string
        assert call_args["qs"]["timezone"] == "America/New_York"

        # Check specific values
        assert body["email"] == "john.doe@example.com"
        assert body["firstName"] == "John"
        assert body["lastName"] == "Doe"
        assert body["timezone"] == "America/New_York"
        assert body["locale"] == "en-us"
        assert body["slug"] == "veronica-chen"
        assert body["duration"] == 30 * 60 * 1000  # 30 minutes in milliseconds

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_response_handling(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test that the API response is properly handled and returned as JSON string."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response with detailed data
        mock_meeting_response = MagicMock()
        expected_response = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
            "endTime": "2024-01-15T10:30:00Z",
            "subject": "Meeting with John Doe",
            "description": "Scheduled meeting",
            "attendees": [{"email": "john.doe@example.com", "name": "John Doe"}],
            "organizer": {"email": "organizer@company.com", "name": "Veronica Chen"},
            "meetingUrl": "https://meet.google.com/abc-defg-hij",
            "status": "scheduled",
        }
        mock_meeting_response.json.return_value = expected_response
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        # Verify the response is properly JSON serialized
        result_data = json.loads(result)
        assert result_data == expected_response
        assert result_data["id"] == "meeting_123"
        assert result_data["meetingUrl"] == "https://meet.google.com/abc-defg-hij"
        assert result_data["status"] == "scheduled"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_edge_case_time_formats(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with edge case time formats."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test edge case time formats
        edge_case_times = [
            "12:00 AM",  # Midnight
            "11:59 PM",  # Just before midnight
            "12:00 PM",  # Noon
            "12:30 PM",  # Half past noon
            "1:00 AM",  # Early morning
            "11:00 PM",  # Late evening
        ]

        for time_str in edge_case_times:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time=time_str,
                duration=30,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_boundary_duration_values(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with boundary duration values."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test boundary duration values
        boundary_durations = [15, 30, 60]  # Only valid enum values

        for duration in boundary_durations:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time="10:00 AM",
                duration=duration,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

            # Verify duration conversion to milliseconds
            call_args = mock_hubspot_client.api_request.call_args[0][0]
            expected_duration_ms = duration * 60 * 1000
            assert call_args["body"]["duration"] == expected_duration_ms

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_iso8601_time_no_timezone_localization(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with ISO8601 time that needs timezone localization."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test ISO8601 time without timezone (should be localized)
        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="2024-01-15T10:00:00",  # No timezone
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_natural_language_time_parsing(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with natural language time parsing."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test natural language time formats
        natural_time_formats = [
            "10:00 AM",
            "2:30 PM",
            "noon",
            "midnight",
            "3pm",
            "9am",
        ]

        for time_str in natural_time_formats:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time=time_str,
                duration=30,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_complex_natural_language_dates(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with complex natural language date parsing."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test complex natural language date formats
        complex_date_formats = [
            "next Monday",
            "next Tuesday",
            "next Wednesday",
            "next Thursday",
            "next Friday",
            "next Saturday",
            "next Sunday",
            "this Friday",
            "this Monday",
            "next week",
            "next month",
        ]

        for date_str in complex_date_formats:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date=date_str,
                meeting_start_time="10:00 AM",
                duration=30,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_json_response_error_handling(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with JSON response error handling."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response with complex JSON
        mock_meeting_response = MagicMock()
        complex_response = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
            "endTime": "2024-01-15T10:30:00Z",
            "subject": "Meeting with John Doe",
            "description": "Scheduled meeting",
            "attendees": [
                {"email": "john.doe@example.com", "name": "John Doe"},
                {"email": "organizer@company.com", "name": "Veronica Chen"},
            ],
            "organizer": {"email": "organizer@company.com", "name": "Veronica Chen"},
            "meetingUrl": "https://meet.google.com/abc-defg-hij",
            "status": "scheduled",
            "metadata": {
                "created_by": "system",
                "source": "hubspot_api",
                "tags": ["customer_meeting", "sales"],
            },
        }
        mock_meeting_response.json.return_value = complex_response
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        # Verify the complex JSON response is properly handled
        result_data = json.loads(result)
        assert result_data == complex_response
        assert result_data["id"] == "meeting_123"
        assert len(result_data["attendees"]) == 2
        assert result_data["metadata"]["tags"] == ["customer_meeting", "sales"]

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_all_timezone_variants(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with all supported timezone variants."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test all supported timezones
        supported_timezones = [
            "America/New_York",
            "America/Los_Angeles",
            "Asia/Tokyo",
            "Europe/London",
        ]

        for timezone in supported_timezones:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time="10:00 AM",
                duration=30,
                slug="veronica-chen",
                time_zone=timezone,
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

            # Verify timezone is passed correctly in API request
            call_args = mock_hubspot_client.api_request.call_args[0][0]
            assert call_args["body"]["timezone"] == timezone
            assert call_args["qs"]["timezone"] == timezone


class TestHubSpotToolsEdgeCases:
    """Edge case tests for HubSpot tools covering boundary conditions and error scenarios."""

    @patch("arklex.env.tools.hubspot.find_contact_by_email.authenticate_hubspot")
    def test_find_contact_by_email_missing_access_token(
        self, mock_authenticate: Mock
    ) -> None:
        """Test contact search with empty access token."""
        mock_authenticate.side_effect = AuthenticationError("Missing access token")

        with pytest.raises(AuthenticationError):
            find_contact_by_email_func(
                email="test@example.com", chat="Hello", access_token=""
            )

    @patch("arklex.env.tools.hubspot.create_ticket.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_ticket.authenticate_hubspot")
    def test_create_ticket_with_empty_issue(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """Test ticket creation with empty issue description."""
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock successful ticket creation even with empty issue
        mock_ticket_response = MagicMock()
        mock_ticket_response.to_dict.return_value = {"id": "ticket_123"}
        mock_hubspot_client.crm.tickets.basic_api.create.return_value = (
            mock_ticket_response
        )
        mock_hubspot_client.crm.associations.v4.basic_api.create.return_value = None

        result = create_ticket_func(
            cus_cid="12345",
            issue="",  # Empty issue
            access_token="test_token",
        )

        assert result == "ticket_123"

    @patch("arklex.env.tools.hubspot.check_available.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.check_available.authenticate_hubspot")
    def test_check_available_with_different_durations(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """Test availability check with different meeting durations (15, 30, 60 minutes)."""
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting links response
        mock_links_response = MagicMock()
        mock_links_response.json.return_value = {
            "status": "success",
            "results": [{"organizerUserId": "67890", "slug": "veronica-chen"}],
        }

        # Mock availability response for 30 minutes
        mock_availability_response = MagicMock()
        mock_availability_response.json.return_value = {
            "linkAvailability": {
                "linkAvailabilityByDuration": {
                    "1800000": {  # 30 minutes in milliseconds
                        "availabilities": [
                            {
                                "startMillisUtc": 1640995200000,
                                "endMillisUtc": 1640997000000,
                            }
                        ]
                    }
                }
            }
        }

        mock_hubspot_client.api_request.side_effect = [
            mock_links_response,
            mock_availability_response,
        ]

        result = check_available_func(
            owner_id=67890,
            time_zone="America/Los_Angeles",
            meeting_date="next Monday",
            duration=30,
            access_token="test_token",
        )

        assert "veronica-chen" in result
        assert "available times for other dates" in result

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_different_timezones(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """Test meeting creation with various timezone configurations."""
        mock_authenticate.return_value = "test_token"
        mock_hubspot_client = MagicMock()
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
            "endTime": "2024-01-15T10:30:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test with different timezones
        timezones = [
            "America/New_York",
            "America/Los_Angeles",
            "Asia/Tokyo",
            "Europe/London",
        ]

        for timezone in timezones:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time="10:00 AM",
                duration=30,
                slug="veronica-chen",
                time_zone=timezone,
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_different_durations(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with different duration values (15, 30, 60 minutes)."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
            "endTime": "2024-01-15T10:15:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test different durations
        durations = [15, 30, 60]
        for duration in durations:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time="10:00 AM",
                duration=duration,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

            # Verify the duration was converted to milliseconds correctly
            call_args = mock_hubspot_client.api_request.call_args[0][0]
            expected_duration_ms = duration * 60 * 1000  # minutes to milliseconds
            assert call_args["body"]["duration"] == expected_duration_ms

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_natural_language_dates(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with various natural language date inputs."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test different natural language date formats
        date_inputs = [
            "tomorrow",
            "today",
            "next Monday",
            "May 1st",
            "December 25th",
            "next week",
        ]

        for date_input in date_inputs:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date=date_input,
                meeting_start_time="10:00 AM",
                duration=30,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_natural_language_times(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with various natural language time inputs."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        # Test different natural language time formats
        time_inputs = [
            "10:00 AM",
            "2:30 PM",
            "9am",
            "3pm",
            "noon",
            "midnight",
        ]

        for time_input in time_inputs:
            result = create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time=time_input,
                duration=30,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="test_token",
            )

            meeting_data = json.loads(result)
            assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_iso8601_time_with_timezone(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with ISO8601 time that includes timezone information."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T15:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="2024-01-15T10:00:00-05:00",  # EST timezone
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_iso8601_time_without_timezone(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with ISO8601 time without timezone (should use provided timezone)."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T15:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="2024-01-15T10:00:00",  # No timezone
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_authentication_error(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
    ) -> None:
        """Test meeting creation when authentication fails."""
        mock_authenticate.side_effect = AuthenticationError("Invalid access token")

        with pytest.raises(AuthenticationError):
            create_meeting_func(
                cus_fname="John",
                cus_lname="Doe",
                cus_email="john.doe@example.com",
                meeting_date="tomorrow",
                meeting_start_time="10:00 AM",
                duration=30,
                slug="veronica-chen",
                time_zone="America/New_York",
                access_token="invalid_token",
            )

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_special_characters_in_names(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with special characters in customer names."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        result = create_meeting_func(
            cus_fname="José",
            cus_lname="O'Connor",
            cus_email="jose.oconnor@example.com",
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"

        # Verify the special characters were passed correctly
        call_args = mock_hubspot_client.api_request.call_args[0][0]
        assert call_args["body"]["firstName"] == "José"
        assert call_args["body"]["lastName"] == "O'Connor"

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_long_email(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with a very long email address."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        long_email = (
            "very.long.email.address.with.many.subdomains@very.long.domain.name.com"
        )

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email=long_email,
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"

        # Verify the long email was passed correctly
        call_args = mock_hubspot_client.api_request.call_args[0][0]
        assert call_args["body"]["email"] == long_email

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_with_complex_slug(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test meeting creation with complex slug containing special characters."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        complex_slug = "dr-maria-garcia-phd"

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=30,
            slug=complex_slug,
            time_zone="America/New_York",
            access_token="test_token",
        )

        meeting_data = json.loads(result)
        assert meeting_data["id"] == "meeting_123"

        # Verify the complex slug was passed correctly
        call_args = mock_hubspot_client.api_request.call_args[0][0]
        assert call_args["body"]["slug"] == complex_slug

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_api_request_structure(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test that the API request structure is correct."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response
        mock_meeting_response = MagicMock()
        mock_meeting_response.json.return_value = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
        }
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        # Verify the API request structure
        call_args = mock_hubspot_client.api_request.call_args[0][0]

        # Check path and method
        assert call_args["path"] == "/scheduler/v3/meetings/meeting-links/book"
        assert call_args["method"] == "POST"

        # Check body structure
        body = call_args["body"]
        assert "slug" in body
        assert "duration" in body
        assert "startTime" in body
        assert "email" in body
        assert "firstName" in body
        assert "lastName" in body
        assert "timezone" in body
        assert "locale" in body

        # Check query string
        assert call_args["qs"]["timezone"] == "America/New_York"

        # Check specific values
        assert body["email"] == "john.doe@example.com"
        assert body["firstName"] == "John"
        assert body["lastName"] == "Doe"
        assert body["timezone"] == "America/New_York"
        assert body["locale"] == "en-us"
        assert body["slug"] == "veronica-chen"
        assert body["duration"] == 30 * 60 * 1000  # 30 minutes in milliseconds

    @patch("arklex.env.tools.hubspot.create_meeting.hubspot.Client.create")
    @patch("arklex.env.tools.hubspot.create_meeting.authenticate_hubspot")
    def test_create_meeting_response_handling(
        self,
        mock_authenticate: Mock,
        mock_client_create: Mock,
        mock_hubspot_client: MagicMock,
    ) -> None:
        """Test that the API response is properly handled and returned as JSON string."""
        mock_authenticate.return_value = "test_token"
        mock_client_create.return_value = mock_hubspot_client

        # Mock meeting creation response with detailed data
        mock_meeting_response = MagicMock()
        expected_response = {
            "id": "meeting_123",
            "startTime": "2024-01-15T10:00:00Z",
            "endTime": "2024-01-15T10:30:00Z",
            "subject": "Meeting with John Doe",
            "description": "Scheduled meeting",
            "attendees": [{"email": "john.doe@example.com", "name": "John Doe"}],
            "organizer": {"email": "organizer@company.com", "name": "Veronica Chen"},
            "meetingUrl": "https://meet.google.com/abc-defg-hij",
            "status": "scheduled",
        }
        mock_meeting_response.json.return_value = expected_response
        mock_hubspot_client.api_request.return_value = mock_meeting_response

        result = create_meeting_func(
            cus_fname="John",
            cus_lname="Doe",
            cus_email="john.doe@example.com",
            meeting_date="tomorrow",
            meeting_start_time="10:00 AM",
            duration=30,
            slug="veronica-chen",
            time_zone="America/New_York",
            access_token="test_token",
        )

        # Verify the response is properly JSON serialized
        result_data = json.loads(result)
        assert result_data == expected_response
        assert result_data["id"] == "meeting_123"
        assert result_data["meetingUrl"] == "https://meet.google.com/abc-defg-hij"
        assert result_data["status"] == "scheduled"
