import logging
import threading
import time
from typing import Any
from arklex.env.tools.tools import register_tool
from twilio.rest import Client as TwilioClient

logger = logging.getLogger(__name__)

description = "Call this tool when you detect that you have reached the voicemail of the person you are calling."

slots = []

outputs = []

errors = []


def _end_call_thread(
    twilio_client: TwilioClient, call_sid: str, response_played_event: threading.Event
) -> None:
    """Helper function to end the call in a separate thread."""
    try:
        logger.info(
            f"Ending call with call_sid: {call_sid}. Sleeping for 10 seconds to allow for final answer"
        )
        time.sleep(20)
        logger.info(
            f"Ending call with call_sid: {call_sid}. Waiting for response to be played"
        )
        response_played_event.wait(timeout=100)
        logger.info(f"Response played. Ending call")
        call = twilio_client.calls(call_sid).update(status="completed")
        logger.info(f"Call end response: {call}")
    except Exception as e:
        logger.error(f"Error ending call: {str(e)}")
        logger.exception(e)


@register_tool(description, slots, outputs, lambda x: x not in errors)
def voicemail(**kwargs: Any) -> str:
    twilio_client = TwilioClient(kwargs.get("sid"), kwargs.get("auth_token"))
    call_sid = kwargs.get("call_sid")
    response_played_event = kwargs.get("response_played_event")
    threading.Thread(
        target=_end_call_thread, args=(twilio_client, call_sid, response_played_event)
    ).start()
    return "call end initiated. make sure the voicemail is left"
