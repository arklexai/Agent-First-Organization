import logging
import time
import threading
from typing import Any
from arklex.env.tools.tools import register_tool
from twilio.rest import Client as TwilioClient

logger = logging.getLogger(__name__)

description = "Hangup the call when end of the conversation is detected"

slots = []

outputs = []

errors = []


def _end_call_thread(
    twilio_client: TwilioClient, call_sid: str, response_played_event: threading.Event
) -> None:
    """Helper function to end the call in a separate thread."""
    try:
        logger.info(
            f"Ending call with call_sid: {call_sid}. Sleeping for 3 seconds to allow for final answer"
        )
        time.sleep(3)
        logger.info(
            f"Ending call with call_sid: {call_sid}. Waiting for response to be played"
        )
        response_played_event.wait()
        logger.info(f"Response played. Ending call")
        call = twilio_client.calls(call_sid).update(status="completed")
        logger.info(f"Call end response: {call}")
    except Exception as e:
        logger.error(f"Error ending call: {str(e)}")
        logger.exception(e)


@register_tool(description, slots, outputs, lambda x: x not in errors)
def end_call(**kwargs: Any) -> str:
    twilio_client = TwilioClient(kwargs.get("sid"), kwargs.get("auth_token"))
    call_sid = kwargs.get("call_sid")
    response_played_event = kwargs.get("response_played_event")
    if twilio_client is not None and call_sid is not None:
        # Start a new thread to end the call
        thread = threading.Thread(
            target=_end_call_thread,
            args=(twilio_client, call_sid, response_played_event),
            daemon=True,  # Thread will exit when main program exits
        )
        thread.start()
        logger.info("Started thread to end call")
    return "Call ending success"
