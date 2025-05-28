import json
import requests
import inspect
from typing import Dict, Any, List

from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt
from arklex.exceptions import ToolExecutionError
from requests.auth import HTTPBasicAuth

from arklex.env.tools.tools import register_tool
from arklex.env.tools.acuity.utils import authenticate_acuity

# Tool description for rescheduling appointments
description: str = "Help the user to reschedule the appointment"

# List of required parameters for the tool
slots: List[Dict[str, Any]] = [
    {
        "name": "apt_id",
        "type": "str",
        "description": "The appointment id of the info session and it should be consisted of numbers. e.g. 1470211171. NOTE THAT IT IS NOT TYPE ID.",
        "prompt": "",
        "required": True,
        "verified": True,
    },
    {
        "name": "time",
        "type": "str",
        "description": "The time of the info session the user wants to reschedule. It could be like Apr 19th 13:00. If you are not sure, ask them to confirm. The final format is like: 2025-04-19T13:00:00-0400",
        "prompt": "",
        "required": True,
    },
]

# List of output parameters for the tool
outputs: List[Dict[str, Any]] = [
    {
        "name": "res_apt",
        "type": "list[dict]",
        "description": "The rescheduled appointment information after the user reschedules.",
    }
]


@register_tool(description, slots, outputs)
def reschedule(apt_id: str, time: str, **kwargs: Dict[str, Any]) -> str:
    """
    Reschedule an existing appointment to a new time.

    Args:
        apt_id (str): ID of the appointment to reschedule
        time (str): New time for the appointment in ISO format
        **kwargs (Dict[str, Any]): Additional keyword arguments

    Returns:
        str: JSON string containing the rescheduled appointment information

    Raises:
        ToolExecutionError: If rescheduling fails
    """
    func_name: str = inspect.currentframe().f_code.co_name
    user_id: str
    api_key: str
    user_id, api_key = authenticate_acuity(kwargs)

    base_url: str = (
        "https://acuityscheduling.com/api/v1/appointments/{}/reschedule".format(apt_id)
    )
    body: Dict[str, str] = {
        "datetime": time,
    }

    response: requests.Response = requests.put(
        base_url, json=body, auth=HTTPBasicAuth(user_id, api_key)
    )

    if response.status_code == 200:
        data: Dict[str, Any] = response.json()
        return json.dumps(data)
    else:
        return ToolExecutionError(func_name, AcuityExceptionPrompt.RESCHEDULE_PROMPT)
