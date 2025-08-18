from pydantic import BaseModel


class ShopifyAdminAuth(BaseModel):
    shop_url: str
    api_version: str
    admin_token: str


class ShopifyStorefrontAuth(BaseModel):
    shop_url: str
    api_version: str
    storefront_token: str
