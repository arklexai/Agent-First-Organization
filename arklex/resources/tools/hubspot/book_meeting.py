import logging
from datetime import datetime

import pytz
from hubspot import HubSpot

from arklex.resources.tools.hubspot.base.entities import HubspotAuthTokens
from arklex.resources.tools.tools import register_tool

logger = logging.getLogger(__name__)

description = "Schedule a meeting for the customer with the specific representative."

slots = [
    {
        "name": "first_name",
        "type": "str",
        "required": True,
        "description": "First name of the customer.",
    },
    {
        "name": "last_name",
        "type": "str",
        "required": True,
        "description": "Last name of the customer.",
    },
    {
        "name": "email",
        "type": "str",
        "required": True,
        "description": "Email of the customer.",
    },
    {
        "name": "start_time",
        "type": "str",
        "required": True,
        "description": f"The start time that the meeting will take place. The meeting's start time includes the hour, as the date alone is not sufficient. Datetime in ISO8601 format without timezone like 'YYYY-MM-DDTHH:MM:SS'. Eg. {datetime.now().isoformat()}.",
    },
    {
        "name": "duration",
        "type": "int",
        "enum": [15, 30, 60],
        "description": "The duration of the meeting in minutes.",
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


@register_tool(description, slots)
def book_meeting(
    first_name: str,
    last_name: str,
    email: str,
    start_time: str,
    duration: int,
    timezone: str,
    auth: HubspotAuthTokens,
    **kwargs: dict[str, object],
) -> str:
    slug = kwargs.get("slug")
    api_client = HubSpot(access_token=auth["access_token"])

    logger.info(
        f"Booking a meeting for {first_name} {last_name} with {slug} at {start_time} for {duration} minutes in {timezone} timezone."
    )

    duration_ms = duration * 60 * 1000
    try:
        # Set the timezone for the start time
        tz = pytz.timezone(timezone)
        start_time_obj = datetime.fromisoformat(start_time)
        start_time_obj = tz.localize(start_time_obj)
        start_timestamp_ms = int(start_time_obj.timestamp() * 1000)
        logger.info(
            f"Start time: {start_time_obj}, start timestamp: {start_timestamp_ms}"
        )
    except ValueError:
        return "error: the start time is not in the correct format"

    res = api_client.api_request(
        {
            "path": "/scheduler/v3/meetings/meeting-links/book",
            "method": "POST",
            "body": {
                "slug": slug,
                "duration": duration_ms,
                "startTime": start_timestamp_ms,
                "email": email,
                "firstName": first_name,
                "lastName": last_name,
                "timezone": timezone,
                "locale": "en-us",
            },
        }
    )

    res = res.json()
    logger.info(f"Meeting book response: {res}")
    if res.get("status") == "error":
        return "error: " + res.get("message")
    return "The meeting has been booked successfully"
