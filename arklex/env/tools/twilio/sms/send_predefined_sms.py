import logging
from typing import Any
from arklex.env.tools.tools import register_tool
from twilio.rest import Client as TwilioClient

logger = logging.getLogger(__name__)

description = "Send a predefined SMS message"

slots = []
outputs = []
errors = []


@register_tool(description, slots, outputs, lambda x: x not in errors)
def send_predefined_sms(**kwargs: Any) -> str:
    """Send a predefined SMS message.

    Args:
        **kwargs: Arguments including message, Twilio credentials and phone numbers
    """
    logger.info(
        f"checking for twilio client: {kwargs.get('twilio_client')}, phone_no_to: {kwargs.get('phone_no_to')}, phone_no_from: {kwargs.get('phone_no_from')}"
    )

    if kwargs.get("twilio_client"):
        twilio_client = kwargs.get("twilio_client")
    else:
        twilio_client = TwilioClient(kwargs.get("sid"), kwargs.get("auth_token"))

    phone_no_to = kwargs.get("phone_no_to")
    phone_no_from = kwargs.get("phone_no_from")
    message_text = kwargs.get("message")

    # Send the message
    message = twilio_client.messages.create(
        body=message_text, from_=phone_no_from, to=phone_no_to
    )

    logger.info(f"Predefined message sent: {message.sid}")
    return f"SMS sent"
