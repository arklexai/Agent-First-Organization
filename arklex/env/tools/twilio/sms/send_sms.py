import logging
from typing import Any
from arklex.env.tools.tools import register_tool
from twilio.rest import Client as TwilioClient

logger = logging.getLogger(__name__)

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


@register_tool(description, slots, outputs, lambda x: x not in errors)
def send_sms(message: str, **kwargs: Any) -> str:
    logger.info(
        f"checking for twilio client: {kwargs.get('twilio_client')}, phone_no_to: {kwargs.get('phone_no_to')}, phone_no_from: {kwargs.get('phone_no_from')}"
    )
    if kwargs.get("twilio_client"):
        twilio_client = kwargs.get("twilio_client")
    else:
        twilio_client = TwilioClient(kwargs.get("sid"), kwargs.get("auth_token"))
    phone_no_to = kwargs.get("phone_no_to")
    phone_no_from = kwargs.get("phone_no_from")
    message = twilio_client.messages.create(
        body=message, from_=phone_no_from, to=phone_no_to
    )
    logger.info(f"Message sent: {message.sid}")
    return f"Message sent"
