from datetime import datetime

import pytz
from google.oauth2 import service_account
from googleapiclient.discovery import build

from arklex.env.tools.google.calendar.utils import AUTH_ERROR
from arklex.env.tools.tools import register_tool
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

# Scopes required for accessing Google Calendar
SCOPES = ["https://www.googleapis.com/auth/calendar"]

description = "Get the busy times of the the company from the Google Calendar"

slots = [
    {
        "name": "time_min",
        "type": "str",
        "description": f"The start of the time range to check for. It includes the hour, as the date alone is not sufficient. The format should be 'YYYY-MM-DDTHH:MM:SS'. Today is {datetime.now().isoformat()}.",
        "prompt": "Please provide the minimum time to query the busy times",
        "required": True,
    },
    {
        "name": "time_max",
        "type": "str",
        "description": f"The end of the time range to check for. It includes the hour, as the date alone is not sufficient. The format should be 'YYYY-MM-DDTHH:MM:SS'. Today is {datetime.now().isoformat()}.",
        "prompt": "Please provide the maximum time to query the busy times",
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

outputs = []

errors = [AUTH_ERROR]


@register_tool(description, slots, outputs, lambda x: x not in errors)
def free_busy(
    time_min: str,
    time_max: str,
    timezone: str,
    **kwargs: str | int | float | bool | None,
) -> str:
    # Authenticate using the service account
    try:
        tz = pytz.timezone(timezone)
        service_account_info = kwargs.get("service_account_info")
        delegated_user = kwargs.get("delegated_user")
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=SCOPES
        ).with_subject(delegated_user)

        # Build the Google Calendar API service
        service = build("calendar", "v3", credentials=credentials)
    except Exception:
        return AUTH_ERROR

    # Build the Google Calendar API service
    service = build("calendar", "v3", credentials=credentials)
    time_min = tz.localize(datetime.fromisoformat(time_min)).isoformat()
    time_max = tz.localize(datetime.fromisoformat(time_max)).isoformat()
    body = {
        "timeMin": time_min,
        "timeMax": time_max,
        "timeZone": timezone,
        "items": [{"id": delegated_user}],
    }
    log_context.info(f"free_busy request: {body}")
    res = service.freebusy().query(body=body).execute()
    log_context.info(f"free_busy response: {res}")

    # res = {'kind': 'calendar#freeBusy', 'timeMin': '2025-02-07T18:00:00.000Z', 'timeMax': '2025-02-07T18:30:00.000Z', 'calendars': {'lucylu@arklex.ai': {'busy': [{'start': '2025-02-07T18:00:00Z', 'end': '2025-02-07T18:30:00Z'}]}}}
    # busy_times = res["calendars"][delegated_user]["busy"]

    return str(res)
