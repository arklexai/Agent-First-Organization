from datetime import datetime, timedelta
import json
from typing import Any
from googleapiclient.discovery import build
from google.oauth2 import service_account

from arklex.env.tools.tools import register_tool
import pytz
from arklex.env.tools.google.calendar.utils import AUTH_ERROR
import logging

logger = logging.getLogger(__name__)

# Scopes required for accessing Google Calendar
SCOPES = ["https://www.googleapis.com/auth/calendar"]

description = "Create the event in the Google Calendar and sens a SMS with the details. Confirm with the user event start time, and timezone before creating the event."
slots = [
    # {
    #     "name": "email",
    #     "type": "string",
    #     "description": "The email of the user, such as 'something@example.com'. Reconfirm the value with the user before proceeding.",
    #     "prompt": "In order to proceed, please provide the email for setting up the meeting",
    #     "required": True,
    # },
    {
        "name": "name",
        "type": "string",
        "description": "The name of the customer. Make sure to ask the user for his/her name before proceeding.",
        "prompt": "Please provide your name",
        "required": True,
    },
    {
        "name": "description",
        "type": "string",
        "description": "Detailed description of the event",
        "prompt": "Please provide your name",
        "required": True,
    },
    {
        "name": "event_name",
        "type": "string",
        "description": "The purpose/summary of the meeting.",
        "prompt": "",
        "required": True,
    },
    {
        "name": "start_time",
        "type": "string",
        "description": "The start time that the meeting will take place. The meeting's start time includes the hour, as the date alone is not sufficient. The format should be 'YYYY-MM-DDTHH:MM:SS'. Today is {today}.".format(
            today=datetime.now().isoformat()
        ),
        "prompt": "Could you please provide the time when will you be available for the meeting?",
        "required": True,
    },
    {
        "name": "timezone",
        "type": "string",
        "enum": pytz.common_timezones,
        "description": "The timezone of the user. For example, 'America/New_York'.",
        "prompt": "Could you please provide your timezone or where are you now?",
        "required": True,
    },
]
outputs = []

DATETIME_ERROR = "error: the start time is not in the correct format"
EVENT_CREATION_ERROR = "error: the event could not be created because {error}"

errors = [AUTH_ERROR, DATETIME_ERROR, EVENT_CREATION_ERROR]

SUCCESS = "The event has been created successfully at {start_time}. The meeting invitation has been sent to {email}."


@register_tool(description, slots, outputs, lambda x: x not in errors)
def create_event(
    name: str,
    description: str,
    event_name: str,
    start_time: str,
    timezone: str,
    duration: int = 30,
    **kwargs: Any,
) -> str:
    # Authenticate using the service account
    try:
        service_account_info = kwargs.get("service_account_info")
        delegated_user = kwargs.get("delegated_user")
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=SCOPES
        ).with_subject(delegated_user)

        # Build the Google Calendar API service
        service = build("calendar", "v3", credentials=credentials)
    except Exception as e:
        return AUTH_ERROR

    # Specify the calendar ID (use 'primary' or the specific calendar's ID)
    calendar_id = "primary"

    try:
        # Parse the start time into a datetime object
        start_time_obj = datetime.fromisoformat(start_time)

        # Define the duration (30 minutes)
        duration = timedelta(minutes=duration)

        # Calculate the end time
        end_time_obj = start_time_obj + duration

        # Convert the end time back to ISO 8601 format
        end_time = end_time_obj.isoformat()

    except Exception as e:
        return DATETIME_ERROR

    try:
        if kwargs.get("phone_no_to"):
            description += f"\nCustomer Phone number: {kwargs.get('phone_no_to')}"

        final_event = {
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
            # 'attendees': [
            #     {'email': email},
            # ]
        }

        # Insert the event
        event = (
            service.events().insert(calendarId=calendar_id, body=final_event).execute()
        )
        logger.info(f"Event created: {event.get('htmlLink')}")

    except Exception as e:
        return EVENT_CREATION_ERROR.format(error=e)

    # return SUCCESS.format(start_time=start_time, email=email)
    logger.info(
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
        logger.info(f"Message sent: {message.sid}")
    return json.dumps(event)
