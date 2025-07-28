from datetime import datetime

import pytz
from hubspot import HubSpot

from arklex.env.tools.tools import register_tool
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

description = (
    "Check the availability of the company representative from Husbspot calendar"
)

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
        "description": f"The start time that the meeting will take place. The meeting's start time includes the hour, as the date alone is not sufficient. The format should be 'YYYY-MM-DDTHH:MM:SS'. Today is {datetime.now().isoformat()}.",
    },
]

outputs = []

errors = []


@register_tool(description, slots, outputs, lambda x: x not in errors)
def check_availability(
    timezone: str, duration: int, start_time: str, **kwargs: dict[str, object]
) -> str:
    slug = kwargs.get("slug")
    log_context.info(
        f"Getting availability for {slug} in {timezone} for {duration} meeting at {start_time}"
    )
    meeting_duration_ms = 3600000
    if duration == 15:
        meeting_duration_ms = 15 * 60 * 1000
    elif duration == 30:
        meeting_duration_ms = 30 * 60 * 1000
    elif duration == 60:
        meeting_duration_ms = 60 * 60 * 1000
    else:
        return "error: invalid meeting duration. Please choose 15, 30, or 60 minutes."
    tz = pytz.timezone(timezone)
    start_time_obj = datetime.fromisoformat(start_time)
    start_time_obj = tz.localize(start_time_obj)
    log_context.info(f"start_time_obj: {start_time_obj}")
    try:
        api_client = HubSpot(access_token=kwargs.get("access_token"))
        res = api_client.api_request(
            {
                "path": f"/scheduler/v3/meetings/meeting-links/book/availability-page/{slug}",
                "method": "GET",
                "headers": {"Content-Type": "application/json"},
                "qs": {"timezone": timezone},
            }
        )
        res = res.json()
    except Exception as e:
        log_context.error(f"Error getting availability: {e}")
        return "error: could not get availability"

    log_context.info(f"Availability: {res}")
    if res.get("status") == "error":
        return "error: could not get availability"

    alternate_times_on_same_day = []
    month_offset = 0
    has_more = True

    while has_more:
        try:
            api_client = HubSpot(access_token=kwargs.get("access_token"))
            res = api_client.api_request(
                {
                    "path": f"/scheduler/v3/meetings/meeting-links/book/availability-page/{slug}",
                    "method": "GET",
                    "headers": {"Content-Type": "application/json"},
                    "qs": {"timezone": timezone, "monthOffset": month_offset},
                }
            )
            res = res.json()

            if res.get("status") == "error":
                return "error: could not get availability"

            availabilities = res["linkAvailability"]["linkAvailabilityByDuration"][
                str(meeting_duration_ms)
            ]["availabilities"]
            has_more = res["linkAvailability"].get("hasMore", False)

            for avail_time in availabilities:
                # convert the timestamp to the user's timezone
                avail_time_utc = datetime.fromtimestamp(
                    avail_time["startMillisUtc"] / 1000, tz=pytz.utc
                )
                avail_time_local = avail_time_utc.astimezone(pytz.timezone(timezone))

                if avail_time_local == start_time_obj:
                    return "The representative is available at that time."
                elif avail_time_local.date() == start_time_obj.date():
                    alternate_times_on_same_day.append(avail_time_local)

            month_offset += 1

        except Exception as e:
            log_context.error(f"Error getting availability: {e}", exc_info=True)
            raise e

    if alternate_times_on_same_day:
        return (
            "The representative is not available at that time. Here are some alternate times on the same day:\n"
            + summarize_time_slots(alternate_times_on_same_day)
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
