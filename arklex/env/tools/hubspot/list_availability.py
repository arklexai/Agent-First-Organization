from datetime import datetime

import pytz
from hubspot import HubSpot

from arklex.env.tools.tools import register_tool
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

description = (
    "List the availability of the company representative from Husbspot calendar"
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
        "description": "The duration of the meeting in minutes.",
        "required": True,
    },
]

outputs = []

errors = []


@register_tool(description, slots, outputs, lambda x: x not in errors)
def list_availability(timezone: str, duration: int, **kwargs: dict[str, object]) -> str:
    slug = kwargs.get("slug")
    log_context.info(
        f"Getting availability for {slug} in {timezone} for {duration} meeting"
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
        log_context.info(f"Availability response: {res}")
    except Exception as e:
        log_context.exception(e)
        log_context.error(f"Error getting availability: {e}")
        return "error: could not get availability"

    available_times = ""
    try:
        for avail_time in res["linkAvailability"]["linkAvailabilityByDuration"][
            str(meeting_duration_ms)
        ]["availabilities"]:
            # convert the timestamp to the user's timezone
            avail_time_utc = datetime.fromtimestamp(
                avail_time["startMillisUtc"] / 1000, tz=pytz.utc
            )
            avail_time_user = avail_time_utc.astimezone(pytz.timezone(timezone))
            avail_time_str = avail_time_user.strftime("%Y-%m-%d %H:%M:%S %Z")
            available_times += f"{avail_time_str}\n"
    except Exception as e:
        log_context.exception(e)
        log_context.error(f"Error parsing availability: {e}")
        return "error: could not parse availability"

    return f"Here are the available times for a {duration} minute meetings:\n{available_times}"
