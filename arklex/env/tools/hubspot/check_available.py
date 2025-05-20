from datetime import datetime, timezone
import inspect
import pytz
import calendar
import hubspot
import parsedatetime
from hubspot.crm.objects.meetings import ApiException
from arklex.env.tools.tools import register_tool, logger
from arklex.env.tools.hubspot.utils import authenticate_hubspot
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.hubspot._exception_prompt import HubspotExceptionPrompt

description = "Give the customer that the unavailable time of the specific representative and the representative's related meeting link information."

slots = [
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
        "enum": [
            "America/New_York",
            "America/Los_Angeles",
            "Asia/Tokyo",
            "Europe/London",
        ],
        "description": "The timezone of the user. It allows users to input abbreviation like nyc, NYC. If you are not sure, just ask the user to confirm in response.",
        "prompt": "Could you please provide your timezone or where are you now?",
        "required": True,
    },
    {
        "name": "meeting_date",
        "type": "str",
        "description": "The exact date (only the month and day) the customer want to take meeting with the representative. e.g. today, Next Monday, May 1st. If users confirm the specific date, then accept it.",
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
outputs = [
    {
        "name": "meeting_info",
        "type": "dict",
        "description": "The available time slots of the representative and the corresponding slug. Typically, the format is '{'slug': 'veronica-chen', 'available_time_slots': {'start':  , 'end':  }}'",
    }
]


@register_tool(description, slots, outputs)
def check_available(
    owner_id: int, time_zone: str, meeting_date: str, duration: int, **kwargs
) -> str:
    func_name = inspect.currentframe().f_code.co_name
    access_token = authenticate_hubspot(kwargs)
    api_client = hubspot.Client.create(access_token=access_token)

    try:
        meeting_link_response = api_client.api_request(
            {
                "path": "/scheduler/v3/meetings/meeting-links",
                "method": "GET",
                "headers": {"Content-Type": "application/json"},
                "qs": {"organizerUserId": int(owner_id)},
            }
        )
        meeting_link_response = meeting_link_response.json()
        if meeting_link_response.get("status") == "error":
            logger.error(
                f"The error for retrieving the meeting link happens:{meeting_link_response.get('message', 'Unknown error happens')}"
            )
            raise ToolExecutionError(
                func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
            )
        else:
            meeting_links = meeting_link_response["results"][0]

        meeting_slug = meeting_links["slug"]
        cal = parsedatetime.Calendar()
        time_struct, _ = cal.parse(meeting_date)
        meeting_date = datetime(*time_struct[:3])

        last_day = calendar.monthrange(meeting_date.year, meeting_date.month)[1]
        is_last_day = meeting_date.day == last_day

        month_offset = 1 if is_last_day else 0

        try:
            availability_response = api_client.api_request(
                {
                    "path": f"/scheduler/v3/meetings/meeting-links/book/availability-page/{meeting_slug}",
                    "method": "GET",
                    "headers": {"Content-Type": "application/json"},
                    "qs": {"timezone": time_zone, "monthOffset": month_offset},
                }
            )
            availability_response = availability_response.json()
            duration_ms = str(duration * 60 * 1000)
            availability = (
                availability_response.get("linkAvailability", {})
                .get("linkAvailabilityByDuration", {})
                .get(duration_ms, {})
            )
            slots = availability.get("availabilities", [])
            time_zone = pytz.timezone(time_zone)

            ab_times = []

            for slot in slots:
                start_ts = slot["startMillisUtc"]
                end_ts = slot["endMillisUtc"]

                ab_times.append({"start": start_ts, "end": end_ts})
            same_dt_info = {"available_time_slots": []}

            other_dt_info = {"available_time_slots": []}

            for ab_time in ab_times:
                start_dt = datetime.fromtimestamp(
                    ab_time["start"] / 1000, tz=pytz.utc
                ).astimezone(time_zone)
                if meeting_date.date() == start_dt.date():
                    same_dt_info["available_time_slots"].append(
                        {
                            "start": datetime.fromtimestamp(
                                ab_time["start"] / 1000, tz=timezone.utc
                            )
                            .astimezone(time_zone)
                            .isoformat(),
                            "end": datetime.fromtimestamp(
                                ab_time["end"] / 1000, tz=timezone.utc
                            )
                            .astimezone(time_zone)
                            .isoformat(),
                        }
                    )
                else:
                    other_dt_info["available_time_slots"].append(
                        {
                            "start": datetime.fromtimestamp(
                                ab_time["start"] / 1000, tz=timezone.utc
                            )
                            .astimezone(time_zone)
                            .isoformat(),
                            "end": datetime.fromtimestamp(
                                ab_time["end"] / 1000, tz=timezone.utc
                            )
                            .astimezone(time_zone)
                            .isoformat(),
                        }
                    )

            response = ""
            if not len(same_dt_info["available_time_slots"]) == 0:
                response += f"The slug for your meeting is: {meeting_slug}\n"
                response += f"The alternative time for you on the same date is {same_dt_info['available_time_slots']}\n"
                response += "Feel free to choose from it\n"
                response += "You must give some available time slots for users as the reference to choose.\n"
            else:
                response += f"The slug for your meeting is: {meeting_slug}\n"
                response += (
                    "I am sorry there is no available time slots on the same day.\n"
                )
                response += f"If you want to change the date, available times for other dates are {other_dt_info['available_time_slots']}\n"
                response += "Feel free to choose from the list.\n"
                response += "You must give some available time slots for users as the reference so that they could choose from.\n"
            return response
        except ApiException as e:
            logger.info(
                "Exception when extracting booking information of someone: %s\n" % e
            )
            raise ToolExecutionError(
                func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
            )
    except ApiException as e:
        logger.info("Exception when extracting meeting scheduler links: %s\n" % e)
        raise ToolExecutionError(
            func_name, HubspotExceptionPrompt.MEETING_LINK_UNFOUND_PROMPT
        )


def parse_natural_date(date_str, base_date=None, timezone=None, date_input=False):
    cal = parsedatetime.Calendar()
    time_struct, _ = cal.parse(date_str, base_date)
    if date_input:
        parsed_dt = datetime(*time_struct[:3])
    else:
        parsed_dt = datetime(*time_struct[:6])

    if base_date and (parsed_dt.date() != base_date.date()):
        parsed_dt = datetime.combine(base_date.date(), parsed_dt.time())

    if timezone:
        local_timezone = pytz.timezone(timezone)
        parsed_dt = local_timezone.localize(parsed_dt)
        parsed_dt = parsed_dt.astimezone(pytz.utc)
    return parsed_dt
