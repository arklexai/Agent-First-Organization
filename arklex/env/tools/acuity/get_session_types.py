import inspect
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt
from arklex.env.tools.acuity.utils import authenticate_acuity
from arklex.env.tools.tools import register_tool
from arklex.utils.exceptions import ToolExecutionError

# Tool description for retrieving session types
description: str = "Retrieve the list of all info sessions for users."

# List of required parameters for the tool
slots: list[dict[str, Any]] = []

# List of output parameters for the tool
outputs: list[dict[str, Any]] = [
    {
        "name": "sessions",
        "type": "list[dict]",
        "description": "All available information sessions. In sessions it should like \"[{'apt_type_id': 76850002, 'name': 'Test N1'}]\".",
    }
]


@register_tool(description, slots, outputs)
def get_session_types(**kwargs: dict[str, Any]) -> str:
    """
    Retrieve all available information session types from Acuity.

    Args:
        **kwargs (Dict[str, Any]): Additional keyword arguments

    Returns:
        str: Formatted string containing all session types and their IDs

    Raises:
        ToolExecutionError: If retrieving session types fails
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
        response_text: str = "The information below is the details of all info session types. You must include the list of all session names in the following response at least.\n"

        # Collect session names for the list
        session_names = []

        for session in data:
            session_id = session.get("id")
            session_name = session.get("name")
            response_text += f"Info session ID (Appointment Type ID):{session_id}\n"
            response_text += f"Info session Name: {session_name}\n"
            if session_name:
                session_names.append(session_name)

        # Add the list of session names
        if session_names:
            response_text += (
                f"\nList of all session names: {', '.join(session_names)}\n"
            )

        return response_text
    else:
        raise ToolExecutionError(
            func_name, AcuityExceptionPrompt.AVAILABLE_TYPES_EXCEPTION_PROMPT
        )
