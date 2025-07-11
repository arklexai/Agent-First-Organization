import inspect
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt
from arklex.env.tools.acuity.utils import authenticate_acuity
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError

# Tool description for retrieving session type ID
description: str = "Retrieve the list of all info sessions for users."

# List of required parameters for the tool
slots: list[dict[str, Any]] = [
    {
        "name": "apt_name",
        "type": "str",
        "description": "The appointment name of the info session. USER NEEDS TO INPUT IT. It allow user to input some parts of the name, but if you are unsure, ask the user to confirm. It should not contain any date, time related information.",
        "prompt": "Which info session would you like to join?",
        "required": True,
    },
]

# List of output parameters for the tool
outputs: list[dict[str, Any]] = [
    {
        "name": "apt_type_id",
        "type": "str",
        "description": "The appointment type id of the info session.",
    }
]


@register_tool(description, slots, outputs)
def get_type_id_by_apt_name(apt_name: str, **kwargs: dict[str, Any]) -> str:
    """
    Get the appointment type ID for a given session name.

    Args:
        apt_name (str): Name of the appointment/session
        **kwargs (Dict[str, Any]): Additional keyword arguments

    Returns:
        str: String containing the appointment type ID

    Raises:
        ToolExecutionError: If retrieving the type ID fails
    """
    func_name: str = inspect.currentframe().f_code.co_name
    user_id: str
    api_key: str
    user_id, api_key = authenticate_acuity(kwargs)

    base_url: str = "https://acuityscheduling.com/api/v1/appointment-types"

    response: requests.Response = requests.get(
        base_url, auth=HTTPBasicAuth(user_id, api_key)
    )

    if response.status_code == 200:
        data: list[dict[str, Any]] = response.json()
        apt_type_id: int | None = next(
            (item["id"] for item in data if item["name"].strip() == apt_name), None
        )

        if apt_type_id is None:
            raise ToolExecutionError(
                func_name, AcuityExceptionPrompt.GET_TYPE_ID_PROMPT
            )

        response_str: str = f"The appointment type id is {apt_type_id}\n"
        return response_str
    else:
        raise ToolExecutionError(func_name, AcuityExceptionPrompt.GET_TYPE_ID_PROMPT)
