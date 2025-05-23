import requests
import inspect

from requests.auth import HTTPBasicAuth
from arklex.env.tools.tools import register_tool
from arklex.env.tools.acuity.utils import authenticate_acuity
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt

description = "Help the user to cancel the appointment"
slots = [
    {
        "name": "apt_name",
        "type": "str",
        "description": "The appointment name of the info session. It allow user to input some parts of the name, but if you are unsure, ask the user to confirm.",
        "prompt": "Which info session would you like to cancel?",
        "required": True,
    },
    {
        "name": "apt_id",
        "type": "str",
        "description": "The appointment id of the info session and it should be consisted of numbers. e.g. 76474933",
        "prompt": "",
        "required": True,
    },
]
outputs = [
    {
        "name": "cal_success",
        "type": "string",
        "description": "Cancelling the appointment successfully.",
    }
]


@register_tool(description, slots, outputs)
def cancel(apt_id, **kwargs):
    func_name = inspect.currentframe().f_code.co_name
    user_id, api_key = authenticate_acuity(kwargs)

    base_url = "https://acuityscheduling.com/api/v1/appointments/{}/cancel".format(
        apt_id
    )
    body = {"cancelNote": "Client requested cancellation"}

    response = requests.put(base_url, json=body, auth=HTTPBasicAuth(user_id, api_key))

    if response.status_code == 200:
        return "The appointment is cancelled successfully."
    else:
        raise ToolExecutionError(func_name, AcuityExceptionPrompt.CANCEL_PROMPT)
