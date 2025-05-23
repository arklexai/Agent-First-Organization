import inspect
import logging
from datetime import datetime

import hubspot
import pytz
from hubspot import HubSpot

from arklex.env.tools.hubspot.utils import authenticate_hubspot
from arklex.env.tools.tools import logger, register_tool

logger = logging.getLogger(__name__)

description = "Check the availability of any representative from Husbspot calendar"

slots = [
    {
        "name": "timezone",
        "type": "str",
        "enum": pytz.common_timezones,
        "description": "The timezone of the user. For example, 'America/New_York'.",
        "prompt": "Could you please provide your timezone or where are you now?",
        "required": True,
    },
    {
        "name": "duration",
        "type": "int",
        "enum": [15, 30, 60],
        "description": "The duration of the meeting in minutes. Ask the user how long he wants the meeting to be.",
        "required": True,
    },
    {
        "name": "start_time",
        "type": "str",
        "required": True,
        "description": "The start time that the meeting will take place. The meeting's start time includes the hour, as the date alone is not sufficient. The format should be 'YYYY-MM-DDTHH:MM:SS'. Today is {today}.".format(
            today=datetime.now().isoformat()
        ),
    },
]

outputs = [
    {
        "name": "meeting_info",
        "type": "dict",
        "decription": "The time and date of the meeting if available. If not, the function will return a list of available time slots to choose from. If no time slots are available, the function will return an error message.",
    }
]

errors = []


@register_tool(description, slots, outputs)
def check_availability(timezone: str, duration: int, start_time: str, **kwargs) -> str:

    access_token = authenticate_hubspot(kwargs)
    api_client = hubspot.Client.create(access_token=access_token)

    if duration not in [15, 30, 60]:
        return "error: invalid meeting duration. Please choose 15, 30, or 60 minutes."
    duration_ms = duration * 60 * 1000
    logger.info(f"duration: {duration_ms} ms")

    tz = pytz.timezone(timezone)
    logger.info(f"timezone: {timezone}")

    start_time_dt = datetime.fromisoformat(start_time)
    start_time_dt = tz.localize(start_time_dt)
    logger.info(f"start_time_dt: {start_time_dt}")

    if not (slugs := get_all_slugs(api_client)):
        return "error: no meeting links found. there are no representatives available for meetings."

    all_alternate_times = []

    for slug in slugs:
        is_available, alternates = check_slug_availability(
            api_client, slug, start_time_dt, duration_ms, timezone
        )
        if is_available:
            return f"The representative is available at {start_time_dt.strftime('%I:%M %p')} on {start_time_dt.strftime('%B %d, %Y')}."
        if alternates:
            all_alternate_times.extend(alternates)

    if all_alternate_times:
        unique_times = sorted(set(all_alternate_times))
        return (
            "The representative is not available at that time. Here are some alternate times on the same day across all representatives:\n"
            + summarize_time_slots(unique_times)
        )
    return "The representative is not available at that time and there are no alternate times on the same day."


def format_time_range(start_time: datetime, end_time: datetime) -> str:
    """Format a time range in a user-friendly way."""
    date_str = start_time.strftime("%B %d, %Y")
    return f"{date_str} {start_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}"


def summarize_time_slots(times: list[datetime]) -> str:
    """Summarize a list of time slots by grouping consecutive slots together.

    Args:
        times: List of datetime objects representing available time slots

    Returns:
        A formatted string with time ranges
    """
    if not times:
        return ""

    # Sort times to ensure proper grouping
    sorted_times = sorted(times)
    ranges = []
    current_start = sorted_times[0]

    for i in range(1, len(sorted_times)):
        # Check if this slot is 15 minutes after the previous slot
        if (
            sorted_times[i] - sorted_times[i - 1]
        ).total_seconds() != 900:  # 900 seconds = 15 minutes
            ranges.append(format_time_range(current_start, sorted_times[i - 1]))
            current_start = sorted_times[i]

    # Add the last range
    ranges.append(format_time_range(current_start, sorted_times[-1]))

    return "\n".join(ranges)


def get_all_slugs(api_client: HubSpot) -> list[str]:
    """Get all slugs from the HubSpot API."""
    try:
        response = api_client.api_request(
            {
                "path": "/scheduler/v3/meetings/meeting-links",
                "method": "GET",
                "headers": {"Content-Type": "application/json"},
            }
        )
        response = response.json()
        return [link["slug"] for link in response["results"]]
    except Exception as e:
        logger.error(f"Error getting slugs: {e}")
        return []


def check_slug_availability(
    api_client: HubSpot,
    meeting_slug: str,
    start_time: datetime,
    duration: int,
    timezone: str,
) -> tuple[bool, list[datetime]]:
    alternate_times_on_same_day = []
    month_offset = 0
    has_more = True

    while has_more:
        try:
            res = api_client.api_request(
                {
                    "path": f"/scheduler/v3/meetings/meeting-links/book/availability-page/{meeting_slug}",
                    "method": "GET",
                    "headers": {"Content-Type": "application/json"},
                    "qs": {"timezone": timezone, "monthOffset": month_offset},
                }
            )
            res = res.json()

            if res.get("status") == "error":
                logger.error(f"Error getting availability: {res}")
                return False, []

            availabilities = res["linkAvailability"]["linkAvailabilityByDuration"][
                str(duration)
            ]["availabilities"]
            has_more = res["linkAvailability"].get("hasMore", False)

            for avail_time in availabilities:
                avail_time_utc = datetime.fromtimestamp(
                    avail_time["startMillisUtc"] / 1000, tz=pytz.utc
                )
                avail_time_local = avail_time_utc.astimezone(pytz.timezone(timezone))

                if avail_time_local == start_time:
                    return True, None
                elif avail_time_local.date() == start_time.date():
                    alternate_times_on_same_day.append(avail_time_local)

            month_offset += 1

        except Exception as e:
            logger.error(f"Error getting availability: {e}")
            logger.exception(e)
            return False, []

    return False, alternate_times_on_same_day
