import ast
import json
from datetime import datetime, timedelta
import pytz

import hubspot
import parsedatetime
from hubspot.crm.objects.meetings import ApiException

from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.hubspot.utils import authenticate_hubspot
from arklex.exceptions import ToolExecutionError

description = "Schedule a meeting for the existing customer with the specific representative."


slots = [
    {
        "name": "cus_fname",
        "type": "str",
        "description": "The first name of the customer contact.",
        "prompt": "",
        "required": True,
    },
    {
        "name": "cus_lname",
        "type": "str",
        "description": "The last name of the customer contact.",
        "prompt": "",
        "required": True,
    },
    {
        "name": "cus_email",
        "type": "str",
        "description": "The email of the customer contact.",
        "prompt": "",
        "required": True,
    },
    {
        "name": "meeting_date",
        "type": "str",
        "description": "The exact date the customer want to take meeting with the representative. e.g. tomorrow, today, Next Monday, May 1st. If you are not sure about the input, ask the user to give you confirmation.",
        "prompt": "Could you please give me the date of the meeting?",
        "required": True,
    },
    {
        "name": "meeting_start_time",
        "type": "str",
        "description": "The exact start time the customer want to take meeting with the representative. e.g. 1pm, 1:00 PM. If you are not sure about the input, ask the user to give you confirmation.",
        "prompt": "Could you please give me the start time of the meeting? Typically, the representative will hold the meeting from 9:00 am to 4:45 pm.",
        "required": True,
    },
    {
        "name": "duration",
        "type": "int",
        "enum": [15, 30, 60],
        "description": "The exact duration of the meeting. Please ask the user to input. DO NOT AUTOMATICALLY GIVE THE SLOT ANY VALUE.",
        "prompt": "Could you please give me the duration of the meeting (e.g. 15, 30, 60 mins)?",
        "required": True,
    },
    {
        "name": "slug",
        "type": "str",
        "description": "The corresponding slug for the meeting link, which is extracted from check_available tool",
        "prompt": "",
        "required": True,
    },
    {
        "name": "bt_slots_ux",
        "type": "str",
        "description": "The busy time slots (unix form) of the representative. This is extracted from check_available tool. This is a list of dict.",
        "prompt": "",
        "required": True,
    },
    {
        "name": "time_zone",
        "type": "str",
        "enum": ["America/New_York", "America/Los_Angeles", "Asia/Tokyo", "Europe/London"],
        "description": "The timezone of the user. For example, 'America/New_York'.",
        "prompt": "Could you please provide your timezone or where are you now?",
        "required": True
    }
]
outputs = [
    {
        "name": "meeting_confirmation_info",
        "type": "string",
        "description": "The detailed information about the meeting to let the customer confirm",
    }
]

UNAVAILABLE_PROMPT = "error: the representative is not available during the required period."


@register_tool(description, slots, outputs)
def create_meeting(cus_fname: str, cus_lname: str, cus_email: str, meeting_date: str,
                   meeting_start_time: str, duration: int,
                   slug: str, bt_slots_ux: str, time_zone: str, **kwargs) -> str:
    access_token = authenticate_hubspot(kwargs)

    meeting_date = parse_natural_date(meeting_date, timezone=time_zone, date_input=True)
    meeting_start_time = parse_natural_date(meeting_start_time, meeting_date, timezone=time_zone)
    meeting_start_time = int(meeting_start_time.timestamp() * 1000)


    duration = int(duration)
    duration = int(timedelta(minutes=duration).total_seconds() * 1000)

    meeting_end_time = meeting_start_time + duration

    bt_slots_ux = json.loads(bt_slots_ux)
    for time_slot in bt_slots_ux:
        if meeting_start_time >= time_slot['start'] and meeting_start_time < time_slot['end']:
            raise ToolExecutionError("HubSpot create_meeting failed", UNAVAILABLE_PROMPT)
        elif meeting_end_time >= time_slot['start'] and meeting_end_time <= time_slot['end']:
            raise ToolExecutionError("HubSpot create_meeting failed", UNAVAILABLE_PROMPT)

    api_client = hubspot.Client.create(access_token=access_token)

    try:
        create_meeting_response = api_client.api_request(
            {
                "path": "/scheduler/v3/meetings/meeting-links/book",
                "method": "POST",
                "body": {
                    "slug": slug,
                    "duration": duration,
                    "startTime": meeting_start_time,
                    "email": cus_email,
                    "firstName": cus_fname,
                    "lastName": cus_lname,
                    "timezone": time_zone,
                    "locale": "en-us",
                },
                "qs": {
                    'timezone': time_zone
                }
            }

        )
        create_meeting_response = create_meeting_response.json()
        return json.dumps(create_meeting_response)
    except ApiException as e:
        logger.info("Exception when scheduling a meeting: %s\n" % e)
        raise ToolExecutionError(f"HubSpot create_meeting failed: {e}", UNAVAILABLE_PROMPT)


def parse_natural_date(date_str, base_date=None, timezone=None, date_input=False):
    cal = parsedatetime.Calendar()
    time_struct, _ = cal.parse(date_str, base_date)
    if date_input:
        parsed_dt = datetime(*time_struct[:3])
    else:
        parsed_dt = datetime(*time_struct[:6])

    if base_date and (parsed_dt.date() != base_date.date()):
        parsed_dt = datetime.combine(base_date.date(), parsed_dt.time())

    if timezone:
        local_timezone = pytz.timezone(timezone)
        parsed_dt = local_timezone.localize(parsed_dt)
        parsed_dt = parsed_dt.astimezone(pytz.utc)
    return parsed_dt


