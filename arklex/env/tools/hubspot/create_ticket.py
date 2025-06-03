"""
Tool for creating support tickets via HubSpot in the Arklex framework.

This module implements a tool for creating support tickets for customers using the HubSpot API. It handles ticket creation, association with contacts, and is designed for integration with the Arklex tool system.
"""

from datetime import datetime
import inspect
import hubspot
from hubspot.crm.objects.emails import ApiException
from hubspot.crm.associations.v4 import AssociationSpec
from hubspot.crm.tickets.models import SimplePublicObjectInputForCreate
from typing import Dict, Any, List

from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.hubspot.utils import authenticate_hubspot
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.hubspot._exception_prompt import HubspotExceptionPrompt


# Tool description for creating support tickets
description: str = "Create a ticket for the existing customer when the customer has some problem about the specific product."

# List of required parameters for the tool
slots: List[Dict[str, Any]] = [
    {
        "name": "cus_cid",
        "type": "str",
        "description": "The id of the customer contact. It consists of all numbers. e.g. 97530152525",
        "prompt": "",
        "required": True,
        "verified": True,
    },
    {
        "name": "issue",
        "type": "str",
        "description": "The question that the customer has for the specific product",
        "prompt": "",
        "required": True,
    },
]

# List of output parameters for the tool
outputs: List[Dict[str, Any]] = [
    {
        "name": "ticket_id",
        "type": "str",
        "description": "The id of the ticket for the existing customer and the specific issue",
    }
]


@register_tool(description, slots, outputs)
def create_ticket(cus_cid: str, issue: str, **kwargs: Dict[str, Any]) -> str:
    """
    Create a support ticket for a customer and associate it with their contact record.

    Args:
        cus_cid (str): Customer contact ID
        issue (str): Description of the customer's issue
        **kwargs (Dict[str, Any]): Additional keyword arguments

    Returns:
        str: ID of the created ticket

    Raises:
        ToolExecutionError: If ticket creation or association fails
    """
    func_name: str = inspect.currentframe().f_code.co_name
    access_token: str = authenticate_hubspot(kwargs)

    api_client: hubspot.Client = hubspot.Client.create(access_token=access_token)

    timestamp: str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3] + "Z"
    subject_name: str = "Issue of " + cus_cid + " at " + timestamp
    ticket_properties: Dict[str, Any] = {
        "hs_pipeline_stage": 1,
        "content": issue,
        "subject": subject_name,
    }
    ticket_for_create: SimplePublicObjectInputForCreate = (
        SimplePublicObjectInputForCreate(properties=ticket_properties)
    )
    try:
        ticket_creation_response: Any = api_client.crm.tickets.basic_api.create(
            simple_public_object_input_for_create=ticket_for_create
        )
        ticket_creation_response: Dict[str, Any] = ticket_creation_response.to_dict()
        ticket_id: str = ticket_creation_response["id"]
        association_spec: List[AssociationSpec] = [
            AssociationSpec(
                association_category="HUBSPOT_DEFINED", association_type_id=15
            )
        ]
        try:
            association_creation_response: Any = (
                api_client.crm.associations.v4.basic_api.create(
                    object_type="contact",
                    object_id=cus_cid,
                    to_object_type="ticket",
                    to_object_id=ticket_id,
                    association_spec=association_spec,
                )
            )
            return ticket_id
        except ApiException as e:
            logger.info("Exception when calling AssociationV4: %s\n" % e)
            raise ToolExecutionError(
                func_name, HubspotExceptionPrompt.TICKET_CREATION_ERROR_PROMPT
            )
    except ApiException as e:
        logger.info("Exception when calling Crm.tickets.create: %s\n" % e)
        raise ToolExecutionError(
            func_name, HubspotExceptionPrompt.TICKET_CREATION_ERROR_PROMPT
        )
