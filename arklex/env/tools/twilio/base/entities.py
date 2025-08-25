from typing import TypedDict


class TwilioAuth(TypedDict):
    sid: str
    auth_token: str
