from datetime import datetime, timedelta
import json

from googleapiclient.discovery import build
from google.oauth2 import service_account

from arklex.env.tools.tools import register_tool
from arklex.env.tools.google.calendar.utils import AUTH_ERROR
from arklex.exceptions import AuthenticationError, ToolExecutionError

# Scopes required for accessing Google Calendar
SCOPES = ['https://www.googleapis.com/auth/calendar']

description = "Create the event in the Google Calendar."
slots = [
    {
        "name": "email",
        "type": "str",
        "description": "The email of the user, such as 'something@example.com'.",
        "prompt": "In order to proceed, please provide the email for setting up the meeting",
        "required": True,
    },
    {
        "name": "event",
        "type": "str",
        "description": "The purpose of the meeting. Or the summary of the conversation",
        "prompt": "",
        "required": True
    },
    {
        "name": "start_time",
        "type": "str",
        "description": "The start time that the meeting will take place. The meeting's start time includes the hour, as the date alone is not sufficient. The format should be 'YYYY-MM-DDTHH:MM:SS'. Today is {today}.".format(today=datetime.now().isoformat()),
        "prompt": "Could you please provide the time when will you be available for the meeting?",
        "required": True
    },
    {
        "name": "timezone",
        "type": "str",
        "enum": ["America/New_York", "America/Los_Angeles", "Asia/Tokyo", "Europe/London"],
        "description": "The timezone of the user. For example, 'America/New_York'.",
        "prompt": "Could you please provide your timezone or where are you now?",
        "required": True
    }
]
outputs = []

DATETIME_ERROR_PROMPT = "Datetime error, please check the start time format."
EVENT_CREATION_ERROR_PROMPT = "Event creation error (the event could not be created because {error}), please try again later."


SUCCESS = "The event has been created successfully at {start_time}. The meeting invitation has been sent to {email}."

@register_tool(description, slots, outputs)
def create_event(email:str, event: str, start_time: str, timezone: str, duration=30, **kwargs) -> str:

    # Authenticate using the service account
    try:
        service_account_info = kwargs.get("service_account_info")
        delegated_user = kwargs.get("delegated_user")
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info, scopes=SCOPES).with_subject(delegated_user)

        # Build the Google Calendar API service
        service = build('calendar', 'v3', credentials=credentials)
    except Exception as e:
        raise AuthenticationError(AUTH_ERROR)

    # Specify the calendar ID (use 'primary' or the specific calendar's ID)
    calendar_id = 'primary'

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
        raise ToolExecutionError("create_event failed", DATETIME_ERROR_PROMPT)
    
    try:

        final_event = {
            'summary': event,
            'description': 'A meeting to discuss project updates.',
            'start': {
                'dateTime': start_time,
                'timeZone': timezone,
            },
            'end': {
                'dateTime': end_time,
                'timeZone': timezone,
            },
            'attendees': [
                {'email': email},
            ]
        }

        # Insert the event
        event = service.events().insert(calendarId=calendar_id, body=final_event).execute()
        print('Event created: %s' % (event.get('htmlLink')))

    except Exception as e:
        raise ToolExecutionError("create_event failed", EVENT_CREATION_ERROR_PROMPT.format(error=e))

    # return SUCCESS.format(start_time=start_time, email=email)
    return json.dumps(event)

