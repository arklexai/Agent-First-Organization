import inspect
import json

import requests
from requests.auth import HTTPBasicAuth

from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt
from arklex.env.tools.acuity.utils import authenticate_acuity
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError

description = "Get the available times of the info session based on the specific date"
slots = [
    {
        "name": "date",
        "type": "str",
        "description": "The date of the info session the user wants to attend. It should consist of year, month, day. e.g. 2025-04-12. This must be input from the user.",
        "prompt": "",
        "required": True,
    },
    {
        "name": "apt_tid",
        "type": "str",
        "description": "The appointment type id of the info session and it should be consisted of numbers. e.g. 76474933",
        "prompt": "",
        "required": True,
        "verified": True,
    },
]
outputs = [
    {
        "name": "time_ls",
        "type": "string",
        "description": "The available times of the specific info session",
    }
]
CREDENTIAL_NOT_FOUND = "error: missing credential information"


@register_tool(description, slots, outputs)
def get_available_times(
    date: str, apt_tid: str, **kwargs: str | int | float | bool | None
) -> str:
    func_name = inspect.currentframe().f_code.co_name
    user_id, api_key = authenticate_acuity(kwargs)

    base_url = f"https://acuityscheduling.com/api/v1/availability/times?appointmentTypeID={apt_tid}&date={date}"
    response = requests.get(base_url, auth=HTTPBasicAuth(user_id, api_key))

    if response.status_code == 200:
        data = response.json()
        # Filter out unavailable times and return only available ones
        available_times = [item for item in data if item.get("available", False)]
        return json.dumps(available_times)
    else:
        raise ToolExecutionError(
            func_name, AcuityExceptionPrompt.AVAILABLE_DATES_EXCEPTION_PROMPT
        )
