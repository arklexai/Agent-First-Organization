from pydantic import BaseModel


class HubspotAuthTokens(BaseModel):
    access_token: str
    refresh_token: str
    expiry_time_str: str
