from pydantic import BaseModel


class HubspotAuth(BaseModel):
    access_token: str
