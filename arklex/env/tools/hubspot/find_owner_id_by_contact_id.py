import ast

import hubspot
from hubspot.crm.objects.emails import ApiException

from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.hubspot.utils import authenticate_hubspot
from arklex.exceptions import ToolExecutionError


description = "Find the owner id in the contact. If owner id is found, the next step is using the extracted owner id to find the information of the owner. "


slots = [
    {
        "name": "cus_cid",
        "type": "str",
        "description": "The id of the customer contact.",
        "prompt": "",
        "required": True,
    },
]

outputs = [
    {
        "name": "owner_id",
        "type": "str",
        "description": "The id of the owner of the contact.",
    }
]

OWNER_UNFOUND_PROMPT = 'Owner not found (not an existing customer)'


@register_tool(description, slots, outputs)
def find_owner_id_by_contact_id(cus_cid, **kwargs) -> str:
    access_token = authenticate_hubspot(kwargs)


    api_client = hubspot.Client.create(access_token=access_token)

    try:
        get_owner_id_response = api_client.api_request(
            {
                "path": "/crm/v3/objects/contacts/{}".format(cus_cid),
                "method": "GET",
                "headers": {
                    'Content-Type': 'application/json'
                },
                "qs": {
                    "properties": 'hubspot_owner_id'
                }
            }

        )
        get_owner_id_response = get_owner_id_response.json()

        owner_id = get_owner_id_response['properties']['hubspot_owner_id']

        return owner_id
    except ApiException as e:
        logger.info("Exception when extracting owner_id of one contact: %s\n" % e)
        raise ToolExecutionError(f"HubSpot find_owner_id_by_contact_id failed: {e}", OWNER_UNFOUND_PROMPT)
