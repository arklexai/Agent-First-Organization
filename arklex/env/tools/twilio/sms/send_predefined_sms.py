"""Send predefined SMS tool for Twilio integration."""

from typing import TypedDict

from twilio.rest import Client as TwilioClient

from arklex.env.tools.tools import register_tool
from arklex.utils.logging_utils import LogContext

log_context = LogContext(__name__)

description = "Send a predefined SMS message"

slots = []
outputs = []
errors = []


class SendPredefinedSmsKwargs(TypedDict, total=False):
    """Type definition for kwargs used in send_predefined_sms function."""

    twilio_client: TwilioClient
    sid: str
    auth_token: str
    phone_no_to: str
    phone_no_from: str
    predefined_message: str
    message: str


@register_tool(description, slots, outputs, lambda x: x not in errors)
def send_predefined_sms(**kwargs: SendPredefinedSmsKwargs) -> str:
    """Send a predefined SMS message.

    Args:
        **kwargs: Arguments including message, Twilio credentials and phone numbers
    """
    # Allow reuse of existing TwilioClient instance or create new one
    twilio_client = kwargs.get("twilio_client")
    if twilio_client is None:
        twilio_client = TwilioClient(kwargs.get("sid"), kwargs.get("auth_token"))

    phone_no_to = kwargs.get("phone_no_to")
    phone_no_from = kwargs.get("phone_no_from")

    # Support both 'message' and 'predefined_message' parameter names for backward compatibility
    predefined_message = kwargs.get("predefined_message") or kwargs.get("message")

    if predefined_message is None:
        return "Error sending predefined SMS: No message content provided"

    log_context.info(
        f"Sending predefined SMS to {phone_no_to} from {phone_no_from}: {predefined_message}"
    )
    try:
        message = twilio_client.messages.create(
            body=predefined_message,
            from_=phone_no_from,
            to=phone_no_to,
        )
        log_context.info(f"Predefined message sent: {message.sid}")
        return f"Predefined SMS sent successfully. SID: {message.sid}"
    except Exception as e:
        return f"Error sending predefined SMS: {str(e)}"
