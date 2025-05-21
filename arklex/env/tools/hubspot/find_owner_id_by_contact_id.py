import inspect
import hubspot
from hubspot.crm.objects.emails import ApiException
from typing import Dict, Any, List

from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.hubspot.utils import authenticate_hubspot
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.hubspot._exception_prompt import HubspotExceptionPrompt

description: str = "Find the owner id in the contact. If owner id is found, the next step is using the extracted owner id to find the information of the owner. "


slots: List[Dict[str, Any]] = [
    {
        "name": "cus_cid",
        "type": "str",
        "description": "The id of the customer contact. It consists of numbers, e.g.104643732568.",
        "prompt": "",
        "required": True,
        "verified": True,
    },
]

outputs: List[Dict[str, Any]] = [
    {
        "name": "owner_id",
        "type": "int",
        "description": "The id of the owner of the contact. It consists of numbers.",
    }
]


@register_tool(description, slots, outputs)
def find_owner_id_by_contact_id(cus_cid: str, **kwargs: Dict[str, Any]) -> str:
    func_name: str = inspect.currentframe().f_code.co_name
    access_token: str = authenticate_hubspot(kwargs)

    api_client: Any = hubspot.Client.create(access_token=access_token)

    try:
        get_owner_id_response: Any = api_client.api_request(
            {
                "path": "/crm/v3/objects/contacts/{}".format(cus_cid),
                "method": "GET",
                "headers": {"Content-Type": "application/json"},
                "qs": {"properties": "hubspot_owner_id"},
            }
        )
        get_owner_id_response: Dict[str, Any] = get_owner_id_response.json()

        owner_id: str = get_owner_id_response["properties"]["hubspot_owner_id"]

        return owner_id
    except ApiException as e:
        logger.info("Exception when extracting owner_id of one contact: %s\n" % e)
        raise ToolExecutionError(func_name, HubspotExceptionPrompt.OWNER_UNFOUND_PROMPT)
