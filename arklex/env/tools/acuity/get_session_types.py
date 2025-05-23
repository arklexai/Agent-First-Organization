import inspect
import requests
from requests.auth import HTTPBasicAuth
from typing import Dict, Any, List

from arklex.env.tools.acuity._exception_prompt import AcuityExceptionPrompt
from arklex.env.tools.tools import register_tool
from arklex.env.tools.acuity.utils import authenticate_acuity
from arklex.exceptions import ToolExecutionError

# Tool description for retrieving session types
description: str = "Retrieve the list of all info sessions for users."

# List of required parameters for the tool
slots: List[Dict[str, Any]] = []

# List of output parameters for the tool
outputs: List[Dict[str, Any]] = [
    {
        "name": "sessions",
        "type": "list[dict]",
        "description": "All available information sessions. In sessions it should like \"[{'apt_type_id': 76850002, 'name': 'Test N1'}]\".",
    }
]


@register_tool(description, slots, outputs)
def get_session_types(**kwargs: Dict[str, Any]) -> str:
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
        data: List[Dict[str, Any]] = response.json()
        response_text: str = "The information below is the details of all info session types. You must include the list of all session names in the following response at least.\n"

        for session in data:
            response_text += (
                f"Info session ID (Appointment Type ID):{session.get('id')}\n"
            )
            response_text += f"Info session Name: {session.get('name')}\n"
        return response_text
    else:
        raise ToolExecutionError(
            func_name, AcuityExceptionPrompt.AVAILABLE_TYPES_EXCEPTION_PROMPT
        )
