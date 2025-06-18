from arklex.utils.exceptions import ExceptionPrompt


class HubspotExceptionPrompt(ExceptionPrompt):
    """
    HubSpot-specific exception prompts.

    This class contains all the error messages used in HubSpot-related operations.
    Each constant represents a specific error scenario that can occur during
    HubSpot API interactions.
    """

    # Error message when a meeting link cannot be found for a representative
    MEETING_LINK_UNFOUND_PROMPT: str = (
        "The representative does not have a meeting link."
    )

    # Error message when a representative is not available during the requested time
    MEETING_UNAVAILABLE_PROMPT: str = (
        "The representative is not available during the required period."
    )

    # Error message when ticket creation fails
    TICKET_CREATION_ERROR_PROMPT: str = (
        "Ticket creation failed, please try again later."
    )

    # Error message when a user cannot be found in the system
    USER_NOT_FOUND_PROMPT: str = "User not found (not an existing customer)"

    # Error message when an owner cannot be found for a contact
    OWNER_UNFOUND_PROMPT: str = "Owner not found (not an existing customer)"
