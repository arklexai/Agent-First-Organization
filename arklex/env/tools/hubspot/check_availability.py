"""
Tool for checking representative availability via HubSpot in the Arklex framework.

This module implements a tool for checking the availability of representatives using the HubSpot API. It supports slot extraction, time zone handling, and provides available meeting times for integration with the Arklex tool system.
"""

import json
from datetime import datetime
from typing import Any

import hubspot
import pytz
from hubspot import HubSpot

from arklex.env.tools.hubspot.utils import authenticate_hubspot
from arklex.env.tools.tools import register_tool
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

description: str = "Check the availability of any representative from Husbspot calendar. If you are not sure any information, please ask users to confirm in response."

slots: list[dict[str, Any]] = [
    {
        "name": "start_time",
        "type": "str",
        "required": True,
        "prompt": "Could you please provide when you want to meet with the representative?",
        "description": f"The start time that the meeting will take place. The meeting's start time includes the hour, as the date alone is not sufficient. The format should be 'YYYY-MM-DDTHH:MM:SS'. Today is {datetime.now().isoformat()}.",
    },
    {
        "name": "duration",
        "type": "int",
        "enum": [15, 30, 60],
        "description": "The duration of the meeting in minutes. Ask the user how long he wants the meeting to be.",
        "prompt": "Could you please provide the duration of the meeting in minutes? It can be 15, 30, or 60 minutes.",
        "required": True,
    },
    {
        "name": "timezone",
        "type": "str",
        "enum": pytz.common_timezones,
        "description": "The timezone of the user. For example, 'America/New_York'.",
        "prompt": "Could you please provide your timezone or where are you now?",
        "required": True,
    },
]


outputs: list[dict[str, Any]] = [
    {
        "name": "meeting_info",
        "type": "dict",
        "description": "The time and date of the meeting if available. If not, the function will return a list of available time slots to choose from. If no time slots are available, the function will say no times available.",
    }
]

errors: list[str] = []


@register_tool(description, slots, outputs)
def check_availability(
    timezone: str, duration: int, start_time: str, **kwargs: dict[str, Any]
) -> str:
    access_token: str = authenticate_hubspot(kwargs)
    api_client: hubspot.Client = hubspot.Client.create(access_token=access_token)

    if duration not in [15, 30, 60]:
        return "error: invalid meeting duration. Please choose 15, 30, or 60 minutes."
    duration_ms: int = duration * 60 * 1000
    log_context.info(f"duration: {duration_ms} ms")

    tz: pytz.BaseTzInfo = pytz.timezone(timezone)
    log_context.info(f"timezone: {timezone}")

    start_time_dt: datetime = datetime.fromisoformat(start_time)
    start_time_dt: datetime = tz.localize(start_time_dt)
    log_context.info(f"start_time_dt: {start_time_dt}")

    if not (slugs := get_all_slugs(api_client)):
        return "error: no meeting links found. there are no representatives available for meetings."

    all_alternate_times: list[tuple[datetime, str]] = []

    for slug in slugs:
        is_available: bool
        alternates: list[datetime] | None
        is_available, alternates = check_slug_availability(
            api_client, slug, start_time_dt, duration_ms, timezone
        )
        if is_available:
            return json.dumps(
                {
                    "status": "available",
                    "duration": duration,
                    "timezone": timezone,
                    "time": [
                        {
                            "slug": slug,
                            "start_time": start_time_dt.isoformat(),
                        }
                    ],
                }
            )
        if alternates:
            for alternate_time in alternates:
                all_alternate_times.append((alternate_time, slug))

    if all_alternate_times:
        unique_times: set[tuple[datetime, str]] = sorted(set(all_alternate_times))
        return json.dumps(
            {
                "status": "alternate times available",
                "duration": duration,
                "timezone": timezone,
                "time": summarize_time_slots(unique_times),
            }
        )
    return json.dumps(
        {
            "status": "no available times on the same day",
            "times": [],
        }
    )


def summarize_time_slots(times: list[tuple[datetime, str]]) -> list[dict[str, str]]:
    """Summarize a list of time slots by grouping consecutive slots together.

    Args:
        times: List of (datetime, slug) tuples representing available time slots

    Returns:
        A list of dicts with slot info
    """
    if not times:
        return []

    # Sort times to ensure proper grouping
    sorted_times: list[tuple[datetime, str]] = sorted(times, key=lambda x: x[0])
    ranges: list[dict[str, str]] = []
    current_start: datetime
    current_slug: str
    current_start, current_slug = sorted_times[0]
    current_end: datetime = current_start

    for i in range(1, len(sorted_times)):
        slot_time: datetime
        slot_slug: str
        slot_time, slot_slug = sorted_times[i]
        # Check if this slot is 15 minutes after the previous slot and has the same slug
        if (
            slot_time - current_end
        ).total_seconds() == 900 and slot_slug == current_slug:
            current_end = slot_time
        else:
            ranges.append(
                {
                    "slug": current_slug,
                    "start_time": current_start.isoformat(),
                }
            )
            current_start, current_slug = slot_time, slot_slug
            current_end = slot_time

    # Add the last range
    ranges.append(
        {
            "slug": current_slug,
            "start_time": current_start.isoformat(),
        }
    )

    return ranges


def get_all_slugs(api_client: HubSpot) -> list[str]:
    """Get all slugs from the HubSpot API."""
    try:
        response: Any = api_client.api_request(
            {
                "path": "/scheduler/v3/meetings/meeting-links",
                "method": "GET",
                "headers": {"Content-Type": "application/json"},
            }
        )
        response: dict[str, Any] = response.json()
        return [link["slug"] for link in response["results"]]
    except Exception as e:
        log_context.error(f"Error getting slugs: {e}")
        return []


def check_slug_availability(
    api_client: HubSpot,
    meeting_slug: str,
    start_time: datetime,
    duration: int,
    timezone: str,
) -> tuple[bool, list[datetime] | None]:
    alternate_times_on_same_day: list[datetime] = []
    month_offset: int = 0
    has_more: bool = True

    while has_more:
        try:
            res: Any = api_client.api_request(
                {
                    "path": f"/scheduler/v3/meetings/meeting-links/book/availability-page/{meeting_slug}",
                    "method": "GET",
                    "headers": {"Content-Type": "application/json"},
                    "qs": {"timezone": timezone, "monthOffset": month_offset},
                }
            )
            res: dict[str, Any] = res.json()

            if res.get("status") == "error":
                log_context.error(f"Error getting availability: {res}")
                return False, []

            availabilities: list[dict[str, Any]] = res["linkAvailability"][
                "linkAvailabilityByDuration"
            ][str(duration)]["availabilities"]
            has_more: bool = res["linkAvailability"].get("hasMore", False)

            for avail_time in availabilities:
                avail_time_utc: datetime = datetime.fromtimestamp(
                    avail_time["startMillisUtc"] / 1000, tz=pytz.utc
                )
                avail_time_local: datetime = avail_time_utc.astimezone(
                    pytz.timezone(timezone)
                )

                if avail_time_local == start_time:
                    return True, None
                elif avail_time_local.date() == start_time.date():
                    alternate_times_on_same_day.append(avail_time_local)

            month_offset += 1

        except Exception as e:
            log_context.error(f"Error getting availability: {e}")
            log_context.exception(e)
            return False, []

    return False, alternate_times_on_same_day
