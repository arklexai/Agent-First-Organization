import inspect
import requests
from requests.auth import HTTPBasicAuth

from arklex.env.tools.acuity.utils import authenticate_acuity
from arklex.env.tools.tools import register_tool
from arklex.exceptions import ToolExecutionError
from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt


description = "Get the available dates of the info session based on the specific month"
slots = [
    {
        "name": "year",
        "type": "str",
        "description": "The year when the info session takes place. Assume the current year is 2025 unless the user specifies otherwise. Format is like: 2025, 2026.",
        "prompt": "Could you please tell me what the current year is?",
        "required": True,
    },
    {
        "name": "month",
        "type": "str",
        "description": "The month of the available info session held by the organization. e.g. January, 1, Jan. If you have known the date, transform to 01.",
        "prompt": "Could you please give me the month you want to attend the info session?",
        "required": True,
    },
    {
        "name": "apt_type_id",
        "type": "str",
        "description": "The appointment type id of the info session. Not the id of the appointment. Note this! It should be correspond to the appointment type.",
        "prompt": "",
        "required": True,
        "verified": True,
    },
]
outputs = [
    {
        "name": "date_ls",
        "type": "dict",
        "description": "The available date of the specific info session in this specific month. The format is like \"{'date': 2025-05-13}\"",
    }
]


@register_tool(description, slots, outputs)
def get_available_dates(year, month, apt_type_id, **kwargs):
    func_name = inspect.currentframe().f_code.co_name
    user_id, api_key = authenticate_acuity(kwargs)

    base_url = "https://acuityscheduling.com/api/v1/availability/dates?appointmentTypeID={}&month={}".format(
        apt_type_id, year + "-" + month
    )
    response = requests.get(base_url, auth=HTTPBasicAuth(user_id, api_key))
    if response.status_code == 200:
        data = response.json()
        response_text = "The information about the availability is as follows. Please provide this to users.\n"

        for date in data:
            response_text += f"Available dates are {date.get('date')}\n"
        return response_text

    else:
        raise ToolExecutionError(
            func_name, AcuityExceptionPrompt.AVAILABLE_DATES_EXCEPTION_PROMPT
        )
