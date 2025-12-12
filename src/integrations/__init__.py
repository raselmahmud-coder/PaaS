"""External integrations for PaaS."""

from src.integrations.shopify import get_shopify_client, ShopifyClient

__all__ = ["get_shopify_client", "ShopifyClient"]

