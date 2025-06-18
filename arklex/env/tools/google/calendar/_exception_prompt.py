from arklex.utils.exceptions import ExceptionPrompt


class GoogleCalendarExceptionPrompt(ExceptionPrompt):
    """
    Google Calendar-specific exception prompts.
    """

    # create_event exception prompt
    EVENT_CREATION_ERROR_PROMPT: str = "Event creation error (the event could not be created because {error}), please try again later."
    DATETIME_ERROR_PROMPT: str = "Datetime error, please check the start time format."
