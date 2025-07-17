from typing import Any

from arklex.utils.exceptions import AuthenticationError

# Error message for missing Acuity authentication parameters
ACUITY_AUTH_ERROR: str = "Missing some or all required hubspot authentication parameters: ACUITY_USER_ID, ACUITY_API_KEY. Please set up 'fixed_args' in the config file. For example, {'name': <unique name of the tool>, 'fixed_args': {'ACUITY_USER_ID': <acuity_user_id>, 'ACUITY_API_KEY': <acuity_api_key>}}"


def authenticate_acuity(kwargs: dict[str, Any]) -> tuple[str, str]:
    """
    Authenticate with Acuity using the provided user ID and API key.

    Args:
        kwargs (Dict[str, Any]): Dictionary containing authentication parameters

    Returns:
        Tuple[str, str]: Tuple containing (user_id, api_key)

    Raises:
        AuthenticationError: If user_id or api_key is missing
    """
    user_id: str = kwargs.get("ACUITY_USER_ID")
    api_key: str = kwargs.get("ACUITY_API_KEY")
    if not user_id and not api_key:
        raise AuthenticationError(ACUITY_AUTH_ERROR)

    return user_id, api_key


def get_acuity_client() -> object:
    """
    Get an Acuity client instance.

    Returns:
        Acuity: An Acuity client instance for making API calls.
    """
    try:
        from acuityscheduling import Acuity

        return Acuity()
    except ImportError as err:
        raise ImportError(
            "acuityscheduling package is required. Please install it with: pip install acuityscheduling"
        ) from err
