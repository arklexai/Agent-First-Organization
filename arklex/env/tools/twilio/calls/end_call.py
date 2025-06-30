"""End call tool for Twilio integration."""

import threading
import time
from typing import TypedDict

from twilio.rest import Client as TwilioClient

from arklex.env.tools.tools import register_tool
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

description = "Hangup the call when end of the conversation is detected"

slots = []

outputs = []

errors = []


class EndCallKwargs(TypedDict, total=False):
    """Type definition for kwargs used in end_call function."""

    sid: str
    auth_token: str
    call_sid: str
    response_played_event: threading.Event


def _end_call_thread(
    twilio_client: TwilioClient, call_sid: str, response_played_event: threading.Event
) -> None:
    """Helper function to end the call in a separate thread."""
    try:
        log_context.info(
            f"Ending call with call_sid: {call_sid}. Sleeping for 10 seconds to allow for final answer"
        )
        time.sleep(10)
        log_context.info(
            f"Ending call with call_sid: {call_sid}. Waiting for response to be played"
        )
        response_played_event.wait(timeout=100)
        log_context.info("Response played. Ending call")
        call = twilio_client.calls(call_sid).update(status="completed")
        log_context.info(f"Call end response: {call}")
    except Exception as e:
        log_context.error(f"Error ending call: {str(e)}")
        log_context.error(f"Exception: {e}")


@register_tool(description, slots, outputs, lambda x: x not in errors)
def end_call(**kwargs: EndCallKwargs) -> str:
    twilio_client = TwilioClient(kwargs.get("sid"), kwargs.get("auth_token"))
    call_sid = kwargs.get("call_sid")
    response_played_event = kwargs.get("response_played_event")
    threading.Thread(
        target=_end_call_thread, args=(twilio_client, call_sid, response_played_event)
    ).start()
    log_context.info("Started thread to end call")
    return "call end initiated"
