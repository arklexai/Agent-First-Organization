"""
Tool for scheduling meetings via HubSpot in the Arklex framework.

This module provides a tool implementation for scheduling meetings with customer representatives using the HubSpot API. It handles slot extraction, time parsing, and meeting creation, and is designed for integration with the Arklex tool system.
"""

import inspect
import json
from datetime import datetime, timedelta
from typing import Any

import hubspot
import parsedatetime
import pytz
from dateutil.parser import isoparse
from hubspot.crm.objects.meetings import ApiException

from arklex.env.tools.hubspot._exception_prompt import HubspotExceptionPrompt
from arklex.env.tools.hubspot.utils import authenticate_hubspot
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

description: str = "Schedule a meeting for the existing customer with the specific representative. If you are not sure any information, please ask users to confirm in response."


slots: list[dict[str, Any]] = [
    {
        "name": "cus_fname",
        "type": "str",
        "description": "The first name of the customer contact.",
        "prompt": "Please provide your first and last name.",
        "required": True,
        "verified": True,
    },
    {
        "name": "cus_lname",
        "type": "str",
        "description": "The last name of the customer contact.",
        "prompt": "Please provide your first and last name.",
        "required": True,
        "verified": True,
    },
    {
        "name": "cus_email",
        "type": "str",
        "description": "The email of the customer contact.",
        "prompt": "Please provide your email address.",
        "required": True,
    },
    {
        "name": "meeting_date",
        "type": "str",
        "description": "The exact date (only the month and day) the customer want to take meeting with the representative. e.g. tomorrow, today, Next Monday, May 1st. If you are not sure about the input, ask the user to give you confirmation.",
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
        "description": "The corresponding slug for the meeting link. Typically, it consists of the organizer's name, like 'lingxiao-chen'.",
        "required": True,
        "verified": True,
    },
    {
        "name": "time_zone",
        "type": "str",
        "enum": [
            "America/New_York",
            "America/Los_Angeles",
            "Asia/Tokyo",
            "Europe/London",
        ],
        "description": "The timezone of the user. For example, 'America/New_York'. It allows users to input abbreviation like nyc, NYC. If you are not sure, just ask the user to confirm in response.",
        "prompt": "Could you please provide your timezone or where are you now?",
        "required": True,
    },
]
outputs: list[dict[str, Any]] = [
    {
        "name": "meeting_confirmation_info",
        "type": "dict",
        "description": "The detailed information about the meeting to let the customer confirm",
    }
]


@register_tool(description, slots, outputs)
def create_meeting(
    cus_fname: str,
    cus_lname: str,
    cus_email: str,
    meeting_date: str,
    meeting_start_time: str,
    duration: int,
    slug: str,
    time_zone: str,
    **kwargs: dict[str, Any],
) -> str:
    """
    Schedule a meeting for a customer with a specific representative.

    Args:
        cus_fname (str): Customer's first name
        cus_lname (str): Customer's last name
        cus_email (str): Customer's email address
        meeting_date (str): Desired meeting date
        meeting_start_time (str): Desired meeting start time
        duration (int): Meeting duration in minutes
        slug (str): Meeting link slug
        time_zone (str): Timezone for the meeting
        **kwargs (Dict[str, Any]): Additional keyword arguments

    Returns:
        str: JSON string containing meeting confirmation information

    Raises:
        ToolExecutionError: If meeting scheduling fails
    """
    func_name: str = inspect.currentframe().f_code.co_name
    access_token: str = authenticate_hubspot(kwargs)

    meeting_date: datetime = parse_natural_date(
        meeting_date, timezone=time_zone, date_input=True
    )
    if is_iso8601(meeting_start_time):
        dt: datetime = isoparse(meeting_start_time)
        if dt.tzinfo is None:
            dt = pytz.timezone(time_zone).localize(dt)
        dt_utc: datetime = dt.astimezone(pytz.utc)
        meeting_start_time: int = int(dt_utc.timestamp() * 1000)
    else:
        dt: datetime = parse_natural_date(
            meeting_start_time, meeting_date, timezone=time_zone
        )
        meeting_start_time: int = int(dt.timestamp() * 1000)

    duration: int = int(duration)
    duration: int = int(timedelta(minutes=duration).total_seconds() * 1000)

    api_client: hubspot.Client = hubspot.Client.create(access_token=access_token)

    try:
        create_meeting_response: Any = api_client.api_request(
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
                "qs": {"timezone": time_zone},
            }
        )
        create_meeting_response: dict[str, Any] = create_meeting_response.json()
        return json.dumps(create_meeting_response)
    except ApiException as e:
        log_context.info(f"Exception when scheduling a meeting: {e}\n")
        raise ToolExecutionError(
            func_name, HubspotExceptionPrompt.MEETING_UNAVAILABLE_PROMPT
        ) from e


def parse_natural_date(
    date_str: str,
    base_date: datetime | None = None,
    timezone: str | None = None,
    date_input: bool = False,
) -> datetime:
    """
    Parse a natural language date string into a datetime object.

    Args:
        date_str (str): Date string to parse
        base_date (Optional[datetime]): Optional base date for relative dates
        timezone (Optional[str]): Optional timezone
        date_input (bool): Whether input is date-only

    Returns:
        datetime: Parsed datetime object
    """
    cal: parsedatetime.Calendar = parsedatetime.Calendar(
        version=parsedatetime.VERSION_CONTEXT_STYLE
    )
    time_struct: tuple = cal.parse(date_str, base_date)[0]
    if date_input:
        parsed_dt: datetime = datetime(*time_struct[:3])
    else:
        parsed_dt: datetime = datetime(*time_struct[:6])

    if base_date and (parsed_dt.date() != base_date.date()):
        parsed_dt = datetime.combine(base_date.date(), parsed_dt.time())

    if timezone:
        local_timezone: pytz.BaseTzInfo = pytz.timezone(timezone)
        # For date-only inputs or when date_input=True, ensure we start at midnight
        # in the local timezone to avoid day shifts during UTC conversion
        if date_input or (parsed_dt.hour >= 12 and parsed_dt.minute > 0):
            # Set to midnight in the local timezone
            parsed_dt = datetime.combine(parsed_dt.date(), datetime.min.time())

        parsed_dt = local_timezone.localize(parsed_dt)
        parsed_dt = parsed_dt.astimezone(pytz.utc)
    return parsed_dt


def is_iso8601(s: str) -> bool:
    """
    Check if a string is in ISO8601 format.

    Args:
        s (str): String to check

    Returns:
        bool: True if string is in ISO8601 format, False otherwise
    """
    try:
        isoparse(s)
        return True
    except Exception:
        return False
