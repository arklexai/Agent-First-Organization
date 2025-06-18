from arklex.utils.exceptions import ExceptionPrompt


class AcuityExceptionPrompt(ExceptionPrompt):
    """
    Acuity-specific exception prompts.

    This class contains all the error messages used in Acuity-related operations.
    Each constant represents a specific error scenario that can occur during
    Acuity API interactions.
    """

    # Error message when finding available dates fails
    AVAILABLE_DATES_EXCEPTION_PROMPT: str = "Finding the available dates failed"

    # Error message when finding available times fails
    AVAILABLE_TIMES_EXCEPTION_PROMPT: str = "Finding the available times failed"

    # Error message when finding session types fails
    AVAILABLE_TYPES_EXCEPTION_PROMPT: str = "Finding all types of info session failed"

    # Error message when booking a session fails
    BOOK_SESSION_EXCEPTION_PROMPT: str = "Failed to book the info session"

    # Error message when no matching session is found
    BOOK_CHECK_EXCEPTION_PROMPT: str = "No info session in the available info sessions"

    # Error message when user has no appointments
    GET_APT_BY_EMAIL_EXCEPTION_PROMPT_1: str = "User didn't make any appointment"

    # Error message when retrieving appointments fails
    GET_APT_BY_EMAIL_EXCEPTION_PROMPT_2: str = "Retrieving all appointments failed"

    # Error message when canceling an appointment fails
    CANCEL_PROMPT: str = "Cancel the appointment failed"

    # Error message when retrieving type ID fails
    GET_TYPE_ID_PROMPT: str = "Retrieving type id failed"

    # Error message when rescheduling an appointment fails
    RESCHEDULE_PROMPT: str = "Rescheduling the appointment failed"
