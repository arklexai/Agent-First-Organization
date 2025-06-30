"""
Tool for finding the owner ID of a contact via HubSpot in the Arklex framework.

This module implements a tool for retrieving the owner ID associated with a contact using the HubSpot API. It is designed for integration with the Arklex tool system.
"""

import inspect
from typing import Any

import hubspot
from hubspot.crm.objects.emails import ApiException

from arklex.env.tools.hubspot._exception_prompt import HubspotExceptionPrompt
from arklex.env.tools.hubspot.utils import authenticate_hubspot
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

# Tool description for finding owner ID
description: str = "Find the owner id in the contact. If owner id is found, the next step is using the extracted owner id to find the information of the owner. "

# List of required parameters for the tool
slots: list[dict[str, Any]] = [
    {
        "name": "cus_cid",
        "type": "str",
        "description": "The id of the customer contact. It consists of numbers, e.g.104643732568.",
        "prompt": "",
        "required": True,
        "verified": True,
    },
]

# List of output parameters for the tool
outputs: list[dict[str, Any]] = [
    {
        "name": "owner_id",
        "type": "int",
        "description": "The id of the owner of the contact. It consists of numbers.",
    }
]


@register_tool(description, slots, outputs)
def find_owner_id_by_contact_id(cus_cid: str, **kwargs: dict[str, Any]) -> str:
    """
    Find the owner ID for a given contact ID.

    Args:
        cus_cid (str): Customer contact ID
        **kwargs (Dict[str, Any]): Additional keyword arguments

    Returns:
        str: Owner ID as string

    Raises:
        ToolExecutionError: If owner ID cannot be found
    """
    func_name: str = inspect.currentframe().f_code.co_name
    access_token: str = authenticate_hubspot(kwargs)

    api_client: hubspot.Client = hubspot.Client.create(access_token=access_token)

    try:
        get_owner_id_response: Any = api_client.api_request(
            {
                "path": f"/crm/v3/objects/contacts/{cus_cid}",
                "method": "GET",
                "headers": {"Content-Type": "application/json"},
                "qs": {"properties": "hubspot_owner_id"},
            }
        )
        get_owner_id_response: dict[str, Any] = get_owner_id_response.json()

        owner_id: str = get_owner_id_response["properties"]["hubspot_owner_id"]

        return owner_id
    except ApiException as e:
        log_context.info(f"Exception when extracting owner_id of one contact: {e}\n")
        raise ToolExecutionError(
            func_name, HubspotExceptionPrompt.OWNER_UNFOUND_PROMPT
        ) from e
