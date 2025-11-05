"""Transfer call tool for Twilio integration."""

import threading
import time
from typing import TypedDict

from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Dial, VoiceResponse

from arklex.resources.tools.tools import register_tool
from arklex.resources.tools.twilio.base.entities import TwilioAuth
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)

description = "Transfer the call to a human agent"

slots = []


class TransferCallKwargs(TypedDict, total=False):
    """Type definition for kwargs used in transfer_call function."""

    call_sid: str
    transfer_to: str
    transfer_message: str
    response_played_event: threading.Event


def _transfer_call_thread(
    twilio_client: TwilioClient,
    call_sid: str,
    transfer_to: str,
    transfer_message: str,
    response_played_event: threading.Event,
) -> None:
    """Helper function to transfer the call in a separate thread."""
    try:
        log_context.info(
            f"Transferring call with call_sid: {call_sid} to {transfer_to}. Sleeping for 5 seconds to allow for final answer"
        )
        time.sleep(5)
        log_context.info(
            f"Transferring call with call_sid: {call_sid} to {transfer_to}. Waiting for response to be played"
        )
        response_played_event.wait(timeout=20)
        log_context.info("Response played. Transferring call")

        # Create TwiML for transfer
        response = VoiceResponse()

        if transfer_message and transfer_message != "":
            time.sleep(1)
            response.say(transfer_message, voice="alice")

        # Transfer the call
        dial = Dial()
        dial.number(transfer_to)
        response.append(dial)

        # Update the call with new TwiML
        call = twilio_client.calls(call_sid)
        call.update(twiml=str(response))

        log_context.info(f"Call transfer response: {call}")
    except Exception as e:
        log_context.error(f"Error transferring call: {str(e)}")
        log_context.error(f"Exception: {e}")
        raise e


@register_tool(description, slots)
def transfer(auth: TwilioAuth, **kwargs: TransferCallKwargs) -> str:
    twilio_client = TwilioClient(auth.get("sid"), auth.get("auth_token"))
    call_sid = kwargs.get("call_sid")
    transfer_to = kwargs.get("transfer_to")
    transfer_message = kwargs.get("transfer_message")
    response_played_event = kwargs.get("response_played_event")
    threading.Thread(
        target=_transfer_call_thread,
        args=(
            twilio_client,
            call_sid,
            transfer_to,
            transfer_message,
            response_played_event,
        ),
    ).start()
    log_context.info("Started thread to transfer call")
    if response_played_event:
        response_played_event.clear()
    return "call transfer initiated"
