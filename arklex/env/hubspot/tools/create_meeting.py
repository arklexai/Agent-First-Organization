import ast
import json
from datetime import datetime, timedelta
import pytz
import inspect

import hubspot
import parsedatetime
from dateutil.parser import isoparse
from hubspot.crm.objects.meetings import ApiException

from arklex.env.tools.tools import register_tool, logger
from arklex.env.hubspot.tools.utils import authenticate_hubspot
from arklex.exceptions import ToolExecutionError
from arklex.env.hubspot.tools._exception_prompt import HubspotExceptionPrompt


description = "Schedule a meeting for the existing customer with the specific representative. If you are not sure any information, please ask users to confirm in response."


slots = [
    {
        "name": "cus_fname",
        "type": "str",
        "description": "The first name of the customer contact.",
        "required": True,
        "verified": True,
    },
    {
        "name": "cus_lname",
        "type": "str",
        "description": "The last name of the customer contact.",
        "required": True,
        "verified": True
    },
    {
        "name": "cus_email",
        "type": "str",
        "description": "The email of the customer contact.",
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
        "prompt": "Could you please give me the start time of the meeting?",
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
        "description": "The corresponding slug for the meeting link. Typically, it consists of the organizer's name, like \'lingxiao-chen\'.",
        "required": True,
        "verified": True,
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
        "type": "dict",
        "description": "The detailed information about the meeting to let the customer confirm",
    }
]


@register_tool(description, slots, outputs)
def create_meeting(cus_fname: str, cus_lname: str, cus_email: str, meeting_date: str,
                   meeting_start_time: str, duration: int,
                   slug: str, time_zone: str, **kwargs) -> str:
    func_name = inspect.currentframe().f_code.co_name
    access_token = authenticate_hubspot(kwargs)

    meeting_date = parse_natural_date(
        meeting_date, timezone=time_zone, date_input=True)
    if is_iso8601(meeting_start_time):
        dt = isoparse(meeting_start_time)
        if dt.tzinfo is None:
            dt = pytz.timezone(time_zone).localize(dt)
        dt_utc = dt.astimezone(pytz.utc)
        meeting_start_time = int(dt_utc.timestamp() * 1000)
    else:
        dt = parse_natural_date(
            meeting_start_time, meeting_date, timezone=time_zone)
        meeting_start_time = int(dt.timestamp() * 1000)

    duration = int(duration)
    duration = int(timedelta(minutes=duration).total_seconds() * 1000)

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
        raise ToolExecutionError(
            func_name, HubspotExceptionPrompt.MEETING_UNAVAILABLE_PROMPT)


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


def is_iso8601(s):
    try:
        isoparse(s)
        return True
    except Exception:
        return False
