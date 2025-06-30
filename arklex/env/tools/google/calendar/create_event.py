import inspect
import json
from datetime import datetime, timedelta
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build

from arklex.env.tools.google.calendar._exception_prompt import (
    GoogleCalendarExceptionPrompt,
)
from arklex.env.tools.google.calendar.utils import AUTH_ERROR
from arklex.env.tools.tools import register_tool
from arklex.types import StreamType
from arklex.utils.exceptions import AuthenticationError, ToolExecutionError
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)


# Scopes required for accessing Google Calendar
SCOPES: list[str] = ["https://www.googleapis.com/auth/calendar"]

description: str = "Create the event in the Google Calendar."
slots: list[dict[str, Any]] = [
    {
        "name": "email",
        "type": "str",
        "description": "The email of the user, such as 'something@example.com'.",
        "prompt": "In order to proceed, please provide the email for setting up the meeting",
        "required": True,
    },
    {
        "name": "name",
        "type": "str",
        "description": "The name of the customer. Make sure to ask the user for his/her name before proceeding.",
        "prompt": "Please provide your name",
        "required": True,
    },
    {
        "name": "description",
        "type": "str",
        "description": "Detailed description of the event",
        "prompt": "Please provide your name",
        "required": True,
    },
    {
        "name": "event_name",
        "type": "str",
        "description": "The purpose/summary of the meeting.",
        "prompt": "",
        "required": True,
    },
    {
        "name": "start_time",
        "type": "str",
        "description": f"The start time that the meeting will take place. The meeting's start time includes the hour, as the date alone is not sufficient. The format should be 'YYYY-MM-DDTHH:MM:SS'. Today is {datetime.now().isoformat()}.",
        "prompt": "Could you please provide the time when will you be available for the meeting?",
        "required": True,
    },
    {
        "name": "timezone",
        "type": "str",
        "enum": [
            "America/New_York",
            "America/Los_Angeles",
            "Asia/Tokyo",
            "Europe/London",
        ],
        "description": "The timezone of the user. For example, 'America/New_York'.",
        "prompt": "Could you please provide your timezone or where are you now?",
        "required": True,
    },
]
outputs: list[dict[str, Any]] = []


SUCCESS: str = "The event has been created successfully at {start_time}. The meeting invitation has been sent to {email}."


@register_tool(description, slots, outputs)
def create_event(
    email: str,
    name: str,
    description: str,
    event_name: str,
    start_time: str,
    timezone: str,
    duration: int = 30,
    **kwargs: str | int | float | bool | None,
) -> str:
    func_name: str = inspect.currentframe().f_code.co_name
    # Authenticate using the service account
    try:
        service_account_info: dict[str, Any] = kwargs.get("service_account_info", {})
        delegated_user: str = kwargs.get("delegated_user", "")
        credentials: service_account.Credentials = (
            service_account.Credentials.from_service_account_info(
                service_account_info, scopes=SCOPES
            ).with_subject(delegated_user)
        )

        # Build the Google Calendar API service
        service: Any = build("calendar", "v3", credentials=credentials)
    except Exception as e:
        raise AuthenticationError(AUTH_ERROR) from e

    # Specify the calendar ID (use 'primary' or the specific calendar's ID)
    calendar_id: str = "primary"

    try:
        # Parse the start time into a datetime object
        start_time_obj: datetime = datetime.fromisoformat(start_time)

        # Define the duration (30 minutes)
        duration_td: timedelta = timedelta(minutes=duration)

        # Calculate the end time
        end_time_obj: datetime = start_time_obj + duration_td

        # Convert the end time back to ISO 8601 format
        end_time: str = end_time_obj.isoformat()

    except Exception as e:
        raise ToolExecutionError(
            func_name, GoogleCalendarExceptionPrompt.DATETIME_ERROR_PROMPT
        ) from e

    try:
        if kwargs.get("phone_no_to"):
            description += f"\nCustomer Phone number: {kwargs.get('phone_no_to')}"

        final_event: dict[str, Any] = {
            "summary": f"{name}: {event_name}",
            "description": description,
            "start": {
                "dateTime": start_time,
                "timeZone": timezone,
            },
            "end": {
                "dateTime": end_time,
                "timeZone": timezone,
            },
        }
        if (
            email is not None
            and kwargs.get("tool_caller") != StreamType.OPENAI_REALTIME_AUDIO
        ):
            final_event["attendees"] = [{"email": email}]

        # Insert the event
        event: dict[str, Any] = (
            service.events().insert(calendarId=calendar_id, body=final_event).execute()
        )
        log_context.info(f"Event created: {event.get('htmlLink')}")

    except Exception as e:
        raise ToolExecutionError(
            func_name,
            GoogleCalendarExceptionPrompt.EVENT_CREATION_ERROR_PROMPT.format(error=e),
        ) from e

    log_context.info(f"tool_caller: {kwargs.get('tool_caller')}")
    if kwargs.get("tool_caller") == StreamType.OPENAI_REALTIME_AUDIO:
        log_context.info(
            f"checking for twilio client: {kwargs.get('twilio_client')}, phone_no_to: {kwargs.get('phone_no_to')}, phone_no_from: {kwargs.get('phone_no_from')}"
        )
        if (
            kwargs.get("twilio_client")
            and kwargs.get("phone_no_to")
            and kwargs.get("phone_no_from")
        ):
            twilio_client = kwargs.get("twilio_client")
            phone_no_to = kwargs.get("phone_no_to")
            phone_no_from = kwargs.get("phone_no_from")
            message = twilio_client.messages.create(
                body=f"Meeting Created\nTitle: {event_name}\nStart Time: {start_time_obj.strftime('%Y-%m-%d %H:%M')}\nLink: {event.get('htmlLink')}",
                from_=phone_no_from,
                to=phone_no_to,
            )
            log_context.info(f"Message sent: {message.sid}")

    # return SUCCESS.format(start_time=start_time, email=email)
    return json.dumps(event)
