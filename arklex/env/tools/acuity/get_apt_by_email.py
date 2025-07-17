import inspect
from datetime import datetime
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt
from arklex.env.tools.acuity.utils import authenticate_acuity
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError

# Tool description for retrieving appointments
description: str = "Get the list of all information sessions"

# List of required parameters for the tool
slots: list[dict[str, Any]] = [
    {
        "name": "email",
        "type": "str",
        "description": "The email of the student. e.g. sarah@gmail.com",
        "prompt": "Could you please give me your email to check your appointment?",
        "required": True,
    }
]

# List of output parameters for the tool
outputs: list[dict[str, Any]] = [
    {
        "name": "apt_ls",
        "type": "list[dict]",
        "description": 'All appointment information about the specific user. The format should be"{"apt_id": 1470211171, "date": "May 21, 2025", "time": "1:10pm", "endtime": "2:00pm", "apt_type": "Consultation ", "apt_type_id": null}"',
    }
]


@register_tool(description, slots, outputs)
def get_apt_by_email(email: str, **kwargs: dict[str, Any]) -> str:
    """
    Get all future appointments for a given email address.

    Args:
        email (str): Email address to search appointments for
        **kwargs (Dict[str, Any]): Additional keyword arguments

    Returns:
        str: Formatted string containing all future appointments

    Raises:
        ToolExecutionError: If no appointments are found or API call fails
    """
    func_name: str = inspect.currentframe().f_code.co_name
    user_id: str
    api_key: str
    user_id, api_key = authenticate_acuity(kwargs)

    base_url: str = "https://acuityscheduling.com/api/v1/appointments"

    response: requests.Response = requests.get(
        base_url, auth=HTTPBasicAuth(user_id, api_key)
    )
    response_str: str = "Please include all appointments (At least the types and time info) in the response to users. There might be multiple appointments.\n"
    if response.status_code == 200:
        data: list[dict[str, Any]] = response.json()
        today: datetime.date = datetime.now().date()
        found_appointments = False

        for item in data:
            if item.get("email") == email:
                apt_date: datetime.date = datetime.strptime(
                    item["date"], "%B %d, %Y"
                ).date()
                if apt_date > today:
                    found_appointments = True
                    response_str += (
                        f"The appointment id of this appointment is: {item['id']}\n"
                    )
                    response_str += f"The date of this appointment is: {item['date']}\n"
                    response_str += (
                        f"The start time of this appointment is: {item['time']}\n"
                    )
                    response_str += (
                        f"The end time of this appointment is: {item['endTime']}\n"
                    )
                    response_str += f"The appointment type: {item['type']}\n"
                    # response_str += f"The appointment type_id cannot be accessed for this tool\n"

        if not found_appointments:
            raise ToolExecutionError(
                func_name, AcuityExceptionPrompt.GET_APT_BY_EMAIL_EXCEPTION_PROMPT_1
            )
        else:
            return response_str
    else:
        raise ToolExecutionError(
            func_name, AcuityExceptionPrompt.GET_APT_BY_EMAIL_EXCEPTION_PROMPT_2
        )
