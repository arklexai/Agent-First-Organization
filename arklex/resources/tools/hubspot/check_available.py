"""
Tool for finding available meeting times for a representative via HubSpot in the Arklex framework.

This module provides a tool for retrieving available meeting slots for a specific representative using the HubSpot API. It is designed for use within the Arklex tool system and supports slot extraction and time zone management.
"""

import calendar
import inspect
from datetime import datetime, timezone
from typing import Any

import hubspot
import parsedatetime
import pytz
from hubspot.crm.objects.meetings import ApiException

from arklex.resources.tools.hubspot._exception_prompt import HubspotExceptionPrompt
from arklex.resources.tools.hubspot.utils import authenticate_hubspot
from arklex.resources.tools.tools import register_tool
from arklex.utils.logging.exceptions import ToolExecutionError
from arklex.utils.logging.logging_utils import LogContext

from .base.entities import HubspotAuthTokens

log_context = LogContext(__name__)

# Tool description for checking available meeting times
description: str = "Give the customer that the unavailable time of the specific representative and the representative's related meeting link information."

# List of required parameters for the tool
slots: list[dict[str, Any]] = [
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
        "enum": pytz.common_timezones,
        "description": "The timezone of the user. For example, 'America/New_York'.",
        "prompt": "Could you please provide your timezone or where are you now?",
        "required": True,
    },
    {
        "name": "meeting_date",
        "type": "str",
        "description": "The exact date (only the month and day) the customer want to take meeting with the representative. e.g. today, Next Monday, May 1st. IMPORTANT: please TRANSFORM tmr to tomorrow.",
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


@register_tool(description, slots)
def check_available(
    owner_id: int,
    time_zone: str,
    meeting_date: str,
    duration: int,
    auth: HubspotAuthTokens,
    **kwargs: dict[str, Any],
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
    access_token: str = authenticate_hubspot(auth)
    api_client: hubspot.Client = hubspot.Client.create(access_token=access_token)

    try:
        meeting_link_response: Any = api_client.api_request(
            {
                "path": "/scheduler/v3/meetings/meeting-links",
                "method": "GET",
                "headers": {"Content-Type": "application/json"},
            }
        )
        meeting_link_response: dict[str, Any] = meeting_link_response.json()
        if meeting_link_response.get("status") == "error" or meeting_link_response.get(
            "error"
        ):
            log_context.error(
                f"The error for retrieving the meeting link happens:{meeting_link_response.get('message', 'Unknown error happens')}"
            )
            raise ToolExecutionError(
                func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
            )
        else:
            # Emphasize on the results part of the response
            if not meeting_link_response.get("results"):
                log_context.error(
                    f"The error for retrieving the meeting link happens:{meeting_link_response.get('message', 'Unknown error happens')}"
                )
                raise ToolExecutionError(
                    func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
                )
            else:
                # Extract the corresponsing link of the specific user
                meeting_links_ls: list[dict[str, Any]] = [
                    item
                    for item in meeting_link_response.get("results")
                    if item.get("organizerUserId") == str(owner_id)
                ]

                # Get the first link of someone
                if len(meeting_links_ls) != 0:
                    meeting_links: dict[str, Any] = meeting_links_ls[0]
                else:
                    # The length is 0, then raise error
                    log_context.error(
                        "There is no meeting links corresponding to the owner's id."
                    )
                    raise ToolExecutionError(
                        func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
                    )

        meeting_slug: str = meeting_links["slug"]
        cal: parsedatetime.Calendar = parsedatetime.Calendar(
            version=parsedatetime.VERSION_CONTEXT_STYLE
        )
        # Use current date as base to ensure consistent parsing
        current_date: datetime = datetime.now()
        dt: datetime
        status: int
        dt, status = cal.parseDT(meeting_date, sourceTime=current_date)
        log_context.info(f"Today is {current_date}. Then the date you ask is:{dt}")
        exact_date: datetime = datetime(dt.year, dt.month, dt.day)

        last_day: int = calendar.monthrange(exact_date.year, exact_date.month)[1]
        is_last_day: bool = exact_date.day == last_day

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
            availability_response: dict[str, Any] = availability_response.json()
            duration_ms: str = str(duration * 60 * 1000)
            availability: dict[str, Any] = (
                availability_response.get("linkAvailability", {})
                .get("linkAvailabilityByDuration", {})
                .get(duration_ms, {})
            )
            slots: list[dict[str, Any]] = availability.get("availabilities", [])
            time_zone_obj: pytz.BaseTzInfo = pytz.timezone(time_zone)

            ab_times: list[dict[str, int]] = []

            for slot in slots:
                start_ts: int = slot.get("startMillisUtc")
                end_ts: int = slot.get("endMillisUtc")

                # Skip slots with missing timestamp data
                if start_ts is None or end_ts is None:
                    continue

                ab_times.append({"start": start_ts, "end": end_ts})
            same_dt_info: dict[str, list[dict[str, str]]] = {"available_time_slots": []}

            other_dt_info: dict[str, list[dict[str, str]]] = {
                "available_time_slots": []
            }

            for ab_time in ab_times:
                start_dt: datetime = datetime.fromtimestamp(
                    ab_time["start"] / 1000, tz=pytz.utc
                ).astimezone(time_zone_obj)
                if exact_date.date() == start_dt.date():
                    same_dt_info["available_time_slots"].append(
                        {
                            "start": datetime.fromtimestamp(
                                ab_time["start"] / 1000, tz=timezone.utc
                            )
                            .astimezone(time_zone_obj)
                            .isoformat(),
                            "end": datetime.fromtimestamp(
                                ab_time["end"] / 1000, tz=timezone.utc
                            )
                            .astimezone(time_zone_obj)
                            .isoformat(),
                        }
                    )
                else:
                    other_dt_info["available_time_slots"].append(
                        {
                            "start": datetime.fromtimestamp(
                                ab_time["start"] / 1000, tz=timezone.utc
                            )
                            .astimezone(time_zone_obj)
                            .isoformat(),
                            "end": datetime.fromtimestamp(
                                ab_time["end"] / 1000, tz=timezone.utc
                            )
                            .astimezone(time_zone_obj)
                            .isoformat(),
                        }
                    )

            response: str = ""
            if len(same_dt_info["available_time_slots"]) != 0:
                response += f"The slug for your meeting is: {meeting_slug}\n"
                response += f"Here are the available time slots for your meeting about {duration} mins in {exact_date}:\n"

                slots: list[tuple[datetime, datetime]] = [
                    (
                        datetime.fromisoformat(slot["start"]),
                        datetime.fromisoformat(slot["end"]),
                    )
                    for slot in same_dt_info["available_time_slots"]
                ]
                slots.sort()

                merged_slots: list[tuple[datetime, datetime]] = []
                current_start: datetime
                current_end: datetime
                current_start, current_end = slots[0]
                for start, end in slots[1:]:
                    if start == current_end:
                        current_end = end
                    else:
                        merged_slots.append((current_start, current_end))
                        current_start, current_end = start, end
                merged_slots.append((current_start, current_end))
                for start, end in merged_slots:
                    start_str: str = start.strftime("%I:%M %p").lstrip("0")
                    end_str: str = end.strftime("%I:%M %p").lstrip("0")
                    response += f" - {start_str} – {end_str}\n"

                response += (
                    "\nPlease pick one of the time slots above as your preferred start time.\n"
                    "You should follow the instructions below to generate response.\n"
                )

                response += (
                    "\nInstruction: If the user already mentioned a specific time (e.g. 9:00 am) and only if it matches one of the available time slots above, "
                    "you should confirm that the requested time is available, and then ask for continuing."
                    "Otherwise, merge all consecutive 15-minute time slots (e.g., 1:00–1:15 and 1:15–1:30 → 1:00–1:30), and present the merged time ranges in the response. DO NOT PROVIDE A LONG LIST."
                    "Please also inform the customer that start times must be at 15-minute intervals (e.g., 9:00, 9:15, etc.)."
                )

            else:
                response += f"The slug for your meeting is: {meeting_slug}\n"
                response += (
                    "I'm sorry, there are no available time slots on the selected date.\n"
                    f"Here are the available time slots on other dates for your {duration}-minute meeting:\n"
                )
                slots: list[tuple[datetime, datetime]] = [
                    (
                        datetime.fromisoformat(slot["start"]),
                        datetime.fromisoformat(slot["end"]),
                    )
                    for slot in other_dt_info["available_time_slots"]
                ]
                slots.sort()
                merged_slots: list[tuple[datetime, datetime]] = []
                current_start: datetime
                current_end: datetime
                current_start, current_end = slots[0]
                for start, end in slots[1:]:
                    if start == current_end:
                        current_end = end
                    else:
                        merged_slots.append((current_start, current_end))
                        current_start, current_end = start, end
                merged_slots.append((current_start, current_end))
                for start, end in merged_slots:
                    start_str: str = start.strftime("%b %d, %I:%M %p").lstrip("0")
                    end_str: str = end.strftime("%I:%M %p").lstrip("0")
                    response += f" - {start_str} – {end_str}\n"

                response += (
                    "\nPlease pick one of the time slots above as your preferred start time.\n"
                    "You should follow the instructions below to generate response.\n"
                )

                response += (
                    "\nInstruction: If the user already mentioned a specific time (e.g. 9:00 am) and only if it matches one of the available time slots above, "
                    "you should confirm that the requested time is available, and then ask for continuing. "
                    "Otherwise, merge all consecutive 15-minute time slots (e.g., 1:00–1:15 and 1:15–1:30 → 1:00–1:30), and present the merged time ranges in the response. DO NOT PROVIDE A LONG LIST."
                    "Please also inform the customer that start times must be at 15-minute intervals (e.g., 9:00, 9:15, etc.)."
                )
            return response
        except ApiException as e:
            log_context.info(
                f"Exception when extracting booking information of someone: {e}\n"
            )
            raise ToolExecutionError(
                func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
            ) from e
    except ApiException as e:
        log_context.info(f"Exception when extracting meeting scheduler links: {e}\n")
        raise ToolExecutionError(
            func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
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
