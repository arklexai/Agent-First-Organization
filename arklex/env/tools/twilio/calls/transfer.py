"""Transfer call tool for Twilio integration."""

from typing import TypedDict

from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Dial, VoiceResponse

from arklex.env.tools.tools import register_tool
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

description = "Transfer the call to a human agent"

slots = []


class TwilioAuth(TypedDict):
    sid: str
    auth_token: str


class TransferCallKwargs(TypedDict, total=False):
    """Type definition for kwargs used in transfer_call function."""

    auth: TwilioAuth
    call_sid: str
    transfer_to_number: str
    transfer_message: str


@register_tool(description, slots)
def transfer(**kwargs: TransferCallKwargs) -> str:
    try:
        call_sid = kwargs.get("call_sid")
        transfer_to_number = kwargs.get("transfer_to")
        transfer_message = kwargs.get("transfer_message")
        log_context.info(
            f"Executing call transfer for call_sid: {call_sid} to {transfer_to_number}"
        )

        twilio_client = TwilioClient(
            kwargs.get("auth").get("sid"), kwargs.get("auth").get("auth_token")
        )

        # Create TwiML for transfer

        response = VoiceResponse()

        if transfer_message:
            response.say(transfer_message, voice="alice")

        # Transfer the call
        dial = Dial()
        dial.number(transfer_to_number)
        response.append(dial)

        # Update the call with new TwiML
        call = twilio_client.calls(call_sid)
        call.update(twiml=str(response))

        log_context.info("Call transfer completed successfully.")

    except Exception as e:
        log_context.error(f"Error executing call transfer: {e}")
        log_context.exception(e)
