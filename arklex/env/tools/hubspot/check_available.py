from datetime import datetime

import hubspot
import parsedatetime
from hubspot.crm.objects.meetings import ApiException

from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.hubspot.utils import authenticate_hubspot
from arklex.exceptions import ToolExecutionError

description = "Give the customer that the unavailable time of the specific representative and the representative's related meeting link information."

slots = [
    {
        "name": "owner_id",
        "type": "str",
        "description": "The owner id of the owner.'",
        "prompt": "",
        "required": True,
    },
    {
        "name": "time_zone",
        "type": "str",
        "enum": ["America/New_York", "America/Los_Angeles", "Asia/Tokyo", "Europe/London"],
        "description": "The timezone of the user. For example, 'America/New_York'. If you are not sure, just ask the user to confirm.",
        "prompt": "Could you please provide your timezone or where are you now?",
        "required": True
    },
    {
        "name": "meeting_date",
        "type": "str",
        "description": "The exact date the customer want to take meeting with the representative. e.g. today, Next Monday, May 1st.",
        "prompt": "Could you please give me the date of the meeting?",
        "required": True,
    }
]
outputs = [
    {
        "name": "meeting_info",
        "type": "string",
        "description": "The unavailable time slots of the representative and the corresponding slug.",
    }
]

MEETING_LINK_UNFOUND_PROMPT = "The representative does not have a meeting link."


@register_tool(description, slots, outputs)
def check_available(owner_id: str, time_zone: str, meeting_date: str, **kwargs) -> str:
    access_token = authenticate_hubspot(kwargs)
    api_client = hubspot.Client.create(access_token=access_token)
    meeting_info = {
        'busy_time_slots': [],
        'busy_time_slots_unix': []
    }
    try:
        meeting_link_response = api_client.api_request(
            {
                "path": "/scheduler/v3/meetings/meeting-links",
                "method": "GET",
                "headers": {
                    'Content-Type': 'application/json'
                },
                "qs": {
                    'organizerUserId': owner_id
                }
            }
        )
        meeting_link_response = meeting_link_response.json()

        if meeting_link_response.get('total') == 0:
            raise ToolExecutionError("HubSpot check_available failed", MEETING_LINK_UNFOUND_PROMPT)
        else:
            meeting_links = meeting_link_response['results'][0]
        meeting_slug = meeting_links['slug']
        meeting_info['slug'] = meeting_slug
        try:
            availability_response = api_client.api_request(
                {
                    "path": "/scheduler/v3/meetings/meeting-links/book/{}".format(meeting_slug),
                    "method": "GET",
                    "headers": {
                        'Content-Type': 'application/json'
                    },
                    "qs": {
                        'timezone': time_zone
                    }
                }
            )
            cal = parsedatetime.Calendar()
            time_struct, _ = cal.parse(meeting_date)
            meeting_date = datetime(*time_struct[:3])
            availability_response = availability_response.json()
            busy_times = availability_response['allUsersBusyTimes'][0]['busyTimes']
            for busy_time in busy_times:
                start_time = datetime.fromtimestamp(busy_time["start"] / 1000)
                end_time = datetime.fromtimestamp(busy_time["end"] / 1000)

                if start_time.date() == meeting_date.date():
                    meeting_info['busy_time_slots'].append({
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    })
                    meeting_info['busy_time_slots_unix'].append({
                        "start": busy_time["start"],
                        "end": busy_time["end"]
                    })
            return str(meeting_info)
        except ApiException as e:
            logger.info("Exception when extracting booking information of someone: %s\n" % e)
            raise ToolExecutionError(f"HubSpot check_available failed: {e}", MEETING_LINK_UNFOUND_PROMPT)
    except ApiException as e:
        logger.info("Exception when extracting meeting scheduler links: %s\n" % e)
        raise ToolExecutionError(f"HubSpot check_available failed: {e}", MEETING_LINK_UNFOUND_PROMPT)




