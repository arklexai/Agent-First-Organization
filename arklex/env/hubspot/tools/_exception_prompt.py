from arklex.exceptions import ExceptionPrompt


class HubspotExceptionPrompt(ExceptionPrompt):
    """
    HubSpot-specific exception prompts.
    """
    # check_available exception prompt
    MEETING_LINK_UNFOUND_PROMPT = "The representative does not have a meeting link."

    # create_meeting exception prompt
    MEETING_UNAVAILABLE_PROMPT = "The representative is not available during the required period."

    # create_ticket exception prompt
    TICKET_CREATION_ERROR_PROMPT = "Ticket creation failed, please try again later."

    # find_contact_by_email exception prompt
    USER_NOT_FOUND_PROMPT = "User not found (not an existing customer)"

    # find_owner_id_by_contact_id exception prompt
    OWNER_UNFOUND_PROMPT = 'Owner not found (not an existing customer)'