"""
Tool for finding available meeting times for a representative via HubSpot in the Arklex framework.

This module provides a tool for retrieving available meeting slots for a specific representative using the HubSpot API. It is designed for use within the Arklex tool system and supports slot extraction and time zone management.
"""

from datetime import datetime, timezone
import inspect
import pytz
import calendar
import hubspot
import parsedatetime
from typing import Dict, Any, List, Optional
from hubspot.crm.objects.meetings import ApiException
from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.hubspot.utils import authenticate_hubspot
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.hubspot._exception_prompt import HubspotExceptionPrompt

# Tool description for checking available meeting times
description: str = "Give the customer that the unavailable time of the specific representative and the representative's related meeting link information."

# List of required parameters for the tool
slots: List[Dict[str, Any]] = [
    {
        "name": "owner_id",
        "type": "int",
        "description": "The id of the owner of the contact.",
        "prompt": "",
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
        "description": "The timezone of the user. It allows users to input abbreviation like nyc, NYC. If you are not sure, just ask the user to confirm in response.",
        "prompt": "Could you please provide your timezone or where are you now?",
        "required": True,
    },
    {
        "name": "meeting_date",
        "type": "str",
        "description": "The exact date (only the month and day) the customer want to take meeting with the representative. e.g. today, Next Monday, May 1st. If users confirm the specific date, then accept it.",
        "prompt": "Could you please give me the date of the meeting?",
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
]

# List of output parameters for the tool
outputs: List[Dict[str, Any]] = [
    {
        "name": "meeting_info",
        "type": "dict",
        "description": "The available time slots of the representative and the corresponding slug. Typically, the format is '{'slug': 'veronica-chen', 'available_time_slots': {'start':  , 'end':  }}'",
    }
]


@register_tool(description, slots, outputs)
def check_available(
    owner_id: int,
    time_zone: str,
    meeting_date: str,
    duration: int,
    **kwargs: Dict[str, Any],
) -> str:
    """
    Check available meeting times for a specific representative.

    Args:
        owner_id (int): ID of the meeting organizer
        time_zone (str): Timezone for checking availability
        meeting_date (str): Date to check availability for
        duration (int): Desired meeting duration in minutes
        **kwargs (Dict[str, Any]): Additional keyword arguments

    Returns:
        str: String containing available time slots information

    Raises:
        ToolExecutionError: If meeting link is not found or availability check fails
    """
    func_name: str = inspect.currentframe().f_code.co_name
    access_token: str = authenticate_hubspot(kwargs)
    api_client: hubspot.Client = hubspot.Client.create(access_token=access_token)

    try:
        meeting_link_response: Any = api_client.api_request(
            {
                "path": "/scheduler/v3/meetings/meeting-links",
                "method": "GET",
                "headers": {"Content-Type": "application/json"},
            }
        )
        meeting_link_response: Dict[str, Any] = meeting_link_response.json()
        if meeting_link_response.get("status") == "error" or meeting_link_response.get(
            "error"
        ):
            logger.error(
                f"The error for retrieving the meeting link happens:{meeting_link_response.get('message', 'Unknown error happens')}"
            )
            raise ToolExecutionError(
                func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
            )
        else:
            # Emphasize on the results part of the response
            if not meeting_link_response.get("results"):
                logger.error(
                    f"The error for retrieving the meeting link happens:{meeting_link_response.get('message', 'Unknown error happens')}"
                )
                raise ToolExecutionError(
                    func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
                )
            else:
                # Extract the corresponsing link of the specific user
                meeting_links_ls: List[Dict[str, Any]] = [
                    item
                    for item in meeting_link_response.get("results")
                    if item.get("organizerUserId") == str(owner_id)
                ]

                # Get the first link of someone
                if len(meeting_links_ls) != 0:
                    meeting_links: Dict[str, Any] = meeting_links_ls[0]
                else:
                    # The length is 0, then raise error
                    logger.error(
                        "There is no meeting links corresponding to the owner's id."
                    )
                    raise ToolExecutionError(
                        func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
                    )

        meeting_slug: str = meeting_links["slug"]
        cal: parsedatetime.Calendar = parsedatetime.Calendar()
        time_struct: tuple = cal.parse(meeting_date)[0]
        meeting_date: datetime = datetime(*time_struct[:3])

        last_day: int = calendar.monthrange(meeting_date.year, meeting_date.month)[1]
        is_last_day: bool = meeting_date.day == last_day

        month_offset: int = 1 if is_last_day else 0

        try:
            availability_response: Any = api_client.api_request(
                {
                    "path": f"/scheduler/v3/meetings/meeting-links/book/availability-page/{meeting_slug}",
                    "method": "GET",
                    "headers": {"Content-Type": "application/json"},
                    "qs": {"timezone": time_zone, "monthOffset": month_offset},
                }
            )
            availability_response: Dict[str, Any] = availability_response.json()
            duration_ms: str = str(duration * 60 * 1000)
            availability: Dict[str, Any] = (
                availability_response.get("linkAvailability", {})
                .get("linkAvailabilityByDuration", {})
                .get(duration_ms, {})
            )
            slots: List[Dict[str, Any]] = availability.get("availabilities", [])
            time_zone: pytz.BaseTzInfo = pytz.timezone(time_zone)

            ab_times: List[Dict[str, int]] = []

            for slot in slots:
                start_ts: int = slot["startMillisUtc"]
                end_ts: int = slot["endMillisUtc"]

                ab_times.append({"start": start_ts, "end": end_ts})
            same_dt_info: Dict[str, List[Dict[str, str]]] = {"available_time_slots": []}

            other_dt_info: Dict[str, List[Dict[str, str]]] = {
                "available_time_slots": []
            }

            for ab_time in ab_times:
                start_dt: datetime = datetime.fromtimestamp(
                    ab_time["start"] / 1000, tz=pytz.utc
                ).astimezone(time_zone)
                if meeting_date.date() == start_dt.date():
                    same_dt_info["available_time_slots"].append(
                        {
                            "start": datetime.fromtimestamp(
                                ab_time["start"] / 1000, tz=timezone.utc
                            )
                            .astimezone(time_zone)
                            .isoformat(),
                            "end": datetime.fromtimestamp(
                                ab_time["end"] / 1000, tz=timezone.utc
                            )
                            .astimezone(time_zone)
                            .isoformat(),
                        }
                    )
                else:
                    other_dt_info["available_time_slots"].append(
                        {
                            "start": datetime.fromtimestamp(
                                ab_time["start"] / 1000, tz=timezone.utc
                            )
                            .astimezone(time_zone)
                            .isoformat(),
                            "end": datetime.fromtimestamp(
                                ab_time["end"] / 1000, tz=timezone.utc
                            )
                            .astimezone(time_zone)
                            .isoformat(),
                        }
                    )

            response: str = ""
            if not len(same_dt_info["available_time_slots"]) == 0:
                response += f"The slug for your meeting is: {meeting_slug}\n"
                response += f"The alternative time for you on the same date is {same_dt_info['available_time_slots']}\n"
                response += f"Feel free to choose from it\n"
                response += f"You must give some available time slots for users as the reference to choose.\n"
            else:
                response += f"The slug for your meeting is: {meeting_slug}\n"
                response += (
                    f"I am sorry there is no available time slots on the same day.\n"
                )
                response += f"If you want to change the date, available times for other dates are {other_dt_info['available_time_slots']}\n"
                response += f"Feel free to choose from the list.\n"
                response += f"You must give some available time slots for users as the reference so that they could choose from.\n"
            return response
        except ApiException as e:
            logger.info(
                "Exception when extracting booking information of someone: %s\n" % e
            )
            raise ToolExecutionError(
                func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
            )
    except ApiException as e:
        logger.info("Exception when extracting meeting scheduler links: %s\n" % e)
        raise ToolExecutionError(
            func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
        )


def parse_natural_date(
    date_str: str,
    base_date: Optional[datetime] = None,
    timezone: Optional[str] = None,
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
    cal: parsedatetime.Calendar = parsedatetime.Calendar()
    time_struct: tuple = cal.parse(date_str, base_date)[0]
    if date_input:
        parsed_dt: datetime = datetime(*time_struct[:3])
    else:
        parsed_dt: datetime = datetime(*time_struct[:6])

    if base_date and (parsed_dt.date() != base_date.date()):
        parsed_dt = datetime.combine(base_date.date(), parsed_dt.time())

    if timezone:
        local_timezone: pytz.BaseTzInfo = pytz.timezone(timezone)
        parsed_dt = local_timezone.localize(parsed_dt)
        parsed_dt = parsed_dt.astimezone(pytz.utc)
    return parsed_dt
