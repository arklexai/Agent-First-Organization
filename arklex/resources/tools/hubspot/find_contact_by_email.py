"""
Tool for finding contacts by email via HubSpot in the Arklex framework.

This module provides a tool for searching and retrieving contact information by email using the HubSpot API. It is designed for use within the Arklex tool system and supports updating communication history.
"""

import inspect
import json
from datetime import datetime, timezone
from typing import Any

import hubspot
from hubspot.crm.associations.v4 import AssociationSpec
from hubspot.crm.objects.communications.models import SimplePublicObjectInputForCreate
from hubspot.crm.objects.emails import ApiException, PublicObjectSearchRequest

from arklex.resources.tools.hubspot._exception_prompt import HubspotExceptionPrompt
from arklex.resources.tools.hubspot.base.entities import HubspotAuthTokens
from arklex.resources.tools.hubspot.utils import authenticate_hubspot
from arklex.resources.tools.tools import register_tool
from arklex.utils.logging.exceptions import ToolExecutionError
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)

description: str = "Find the contacts record by email. If the record is found, the lastmodifieddate of the contact will be updated. If the correspodning record is not found, the function will return an error message."

slots: list[dict[str, Any]] = [
    {
        "name": "email",
        "type": "str",
        "description": "The email of the user, such as 'something@example.com'.",
        "prompt": "Thanks for your interest in our products! Could you please provide your email?",
        "required": True,
    },
    {
        "name": "chat",
        "type": "str",
        "description": "This occurs when user communicates with the chatbot",
        "prompt": "",
        "required": True,
    },
]


@register_tool(description, slots)
def find_contact_by_email(
    email: str, chat: str, auth: HubspotAuthTokens, **kwargs: dict[str, Any]
) -> str:
    """
    Find a contact in HubSpot by email and update their communication history.

    Args:
        email (str): Email address of the contact to find
        chat (str): Chat message to record in communication history
        **kwargs (Dict[str, Any]): Additional keyword arguments

    Returns:
        str: JSON string containing contact information

    Raises:
        ToolExecutionError: If contact is not found or API calls fail
    """
    func_name: str = inspect.currentframe().f_code.co_name
    access_token: str = authenticate_hubspot(auth)

    api_client: hubspot.Client = hubspot.Client.create(access_token=access_token)
    public_object_search_request: PublicObjectSearchRequest = PublicObjectSearchRequest(
        filter_groups=[
            {"filters": [{"propertyName": "email", "operator": "EQ", "value": email}]}
        ]
    )

    try:
        contact_search_response: Any = api_client.crm.contacts.search_api.do_search(
            public_object_search_request=public_object_search_request
        )
        log_context.info(f"Found contact by email: {email}")
        contact_search_response: dict[str, Any] = contact_search_response.to_dict()
        if contact_search_response["total"] == 1:
            contact_id: str = contact_search_response["results"][0]["id"]
            communication_data: SimplePublicObjectInputForCreate = (
                SimplePublicObjectInputForCreate(
                    properties={
                        "hs_communication_channel_type": "CUSTOM_CHANNEL_CONVERSATION",
                        "hs_communication_body": chat,
                        "hs_communication_logged_from": "CRM",
                        "hs_timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            )
            contact_info_properties: dict[str, Any] = {
                "contact_id": contact_id,
                "contact_email": email,
                "contact_first_name": contact_search_response["results"][0][
                    "properties"
                ].get("firstname"),
                "contact_last_name": contact_search_response["results"][0][
                    "properties"
                ].get("lastname"),
            }
            try:
                communication_creation_response: Any = (
                    api_client.crm.objects.communications.basic_api.create(
                        communication_data
                    )
                )
                communication_creation_response: dict[str, Any] = (
                    communication_creation_response.to_dict()
                )
                communication_id: str = communication_creation_response["id"]
                association_spec: list[AssociationSpec] = [
                    AssociationSpec(
                        association_category="HUBSPOT_DEFINED", association_type_id=82
                    )
                ]
                try:
                    api_client.crm.associations.v4.basic_api.create(
                        object_type="contact",
                        object_id=contact_id,
                        to_object_type="communication",
                        to_object_id=communication_id,
                        association_spec=association_spec,
                    )
                except ApiException as e:
                    log_context.info(f"Exception when calling AssociationV4: {e}\n")
            except ApiException as e:
                log_context.info(f"Exception when calling basic_api: {e}\n")

            return json.dumps(contact_info_properties)
        else:
            raise ToolExecutionError(
                func_name, HubspotExceptionPrompt.USER_NOT_FOUND_PROMPT
            )
    except ApiException as e:
        log_context.info(f"Exception when calling search_api: {e}\n")
        raise ToolExecutionError(
            func_name, HubspotExceptionPrompt.USER_NOT_FOUND_PROMPT
        ) from e
