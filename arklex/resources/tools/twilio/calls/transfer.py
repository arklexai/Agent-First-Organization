"""Transfer call tool for Twilio integration."""

import os
import threading
import time
from typing import TypedDict

from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Dial, Start, VoiceResponse

from arklex.resources.tools.tools import register_tool
from arklex.resources.tools.twilio.base.entities import TwilioAuth
from arklex.utils.logging.logging_utils import LogContext

log_context = LogContext(__name__)

description = "Transfer the call to a human agent"
DOMAIN = os.getenv("DOMAIN")

slots = [
    {
        "name": "summary",
        "type": "str",
        "description": "Provide detailed summary of the complete dialog flow that happened during the call",
        "required": True,
    },
]


class TransferCallKwargs(TypedDict, total=False):
    """Type definition for kwargs used in transfer_call function."""

    call_sid: str
    transfer_to: str
    transfer_message: str
    response_played_event: threading.Event
    summary: str


def _transfer_call_thread(
    twilio_client: TwilioClient,
    call_sid: str,
    transfer_to: str,
    transfer_message: str,
    response_played_event: threading.Event,
    transcripts_status_callback_url: str,
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
        start = Start()
        # Reference: https://www.twilio.com/docs/voice/twiml/transcription
        start.transcription(
            status_callback_url=transcripts_status_callback_url,
            enable_automatic_punctuation=True,
        )

        response.append(start)

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
    summary = kwargs.get("summary")
    transcripts_status_callback_url = (
        f"https://{DOMAIN}/api/v1alpha2/voice-call/transcribe-callback"
    )
    threading.Thread(
        target=_transfer_call_thread,
        args=(
            twilio_client,
            call_sid,
            transfer_to,
            transfer_message,
            response_played_event,
            transcripts_status_callback_url,
        ),
    ).start()
    log_context.info("Started thread to transfer call")
    if response_played_event:
        response_played_event.clear()
    return f"call transfer initiated. Summary: {summary}"
