"""Send SMS tool for Twilio integration."""

from typing import TypedDict

from twilio.rest import Client as TwilioClient

from arklex.env.tools.tools import register_tool
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

description = "Send an SMS message"

slots = [
    {
        "name": "message",
        "description": "The message to send",
        "required": True,
        "type": "str",
    }
]

outputs = []

errors = []


class SendSmsKwargs(TypedDict, total=False):
    """Type definition for kwargs used in send_sms function."""

    sid: str
    auth_token: str
    phone_no_to: str
    phone_no_from: str
    message: str


@register_tool(description, slots, outputs, lambda x: x not in errors)
def send_sms(**kwargs: SendSmsKwargs) -> str:
    twilio_client = TwilioClient(kwargs.get("sid"), kwargs.get("auth_token"))
    phone_no_to = kwargs.get("phone_no_to")
    phone_no_from = kwargs.get("phone_no_from")
    message_text = kwargs.get("message")
    log_context.info(
        f"Sending SMS to {phone_no_to} from {phone_no_from}: {message_text}"
    )
    try:
        message = twilio_client.messages.create(
            body=message_text, from_=phone_no_from, to=phone_no_to
        )
        log_context.info(f"Message sent: {message.sid}")
        return f"SMS sent successfully. SID: {message.sid}"
    except Exception as e:
        return f"Error sending SMS: {str(e)}"
