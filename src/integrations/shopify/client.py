"""Shopify Admin API client for real-world e-commerce validation.

This module provides an async client for interacting with Shopify's Admin API,
enabling real-world validation of the PaaS resilience system against actual
e-commerce operations.

Usage:
    client = get_shopify_client()
    product = await client.create_product({
        "title": "Test Product",
        "body_html": "<p>Description</p>",
        "vendor": "PaaS Test",
        "product_type": "Test",
        "variants": [{"price": "19.99", "sku": "TEST-001"}]
    })
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from src.config import settings
from src.integrations.shopify.utils import RateLimiter

logger = logging.getLogger(__name__)

# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None

# Test product prefix for cleanup
TEST_PRODUCT_PREFIX = "PAAS_TEST_"


def _get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(requests_per_second=2)
    return _rate_limiter


class ShopifyClient:
    """Async Shopify Admin API client.
    
    This client provides methods for CRUD operations on Shopify products,
    with built-in rate limiting to respect API constraints.
    
    Attributes:
        base_url: The base URL for the Shopify Admin API
        headers: HTTP headers including authentication
        rate_limiter: Rate limiter to prevent API throttling
    """
    
    def __init__(
        self,
        store_url: Optional[str] = None,
        access_token: Optional[str] = None,
        api_version: Optional[str] = None,
    ):
        """Initialize the Shopify client.
        
        Args:
            store_url: Shopify store URL (e.g., "mystore.myshopify.com")
            access_token: Shopify Admin API access token
            api_version: API version (default: from settings)
        """
        self.store_url = store_url or settings.shopify_store_url
        self.access_token = access_token or settings.shopify_access_token
        self.api_version = api_version or settings.shopify_api_version
        
        if not self.store_url or not self.access_token:
            logger.warning(
                "Shopify credentials not configured. "
                "Set SHOPIFY_STORE_URL and SHOPIFY_ACCESS_TOKEN in .env"
            )
        
        self.base_url = f"https://{self.store_url}/admin/api/{self.api_version}"
        self.headers = {
            "X-Shopify-Access-Token": self.access_token,
            "Content-Type": "application/json",
        }
        self.rate_limiter = _get_rate_limiter()
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make a rate-limited request to the Shopify API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., "/products.json")
            json: JSON body for POST/PUT requests
            params: Query parameters
            
        Returns:
            Response JSON as dictionary
            
        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        await self.rate_limiter.acquire()
        
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"
        
        logger.debug(f"Shopify API {method} {url}")
        
        response = await client.request(
            method=method,
            url=url,
            json=json,
            params=params,
        )
        response.raise_for_status()
        
        if response.status_code == 204:
            return {}
        
        return response.json()
    
    # =========================================================================
    # Product Operations
    # =========================================================================
    
    async def create_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new product in Shopify.
        
        Args:
            product_data: Product data following Shopify's product schema
                - title: Product title (required)
                - body_html: Product description HTML
                - vendor: Vendor name
                - product_type: Product type/category
                - variants: List of variant objects with price, sku, etc.
                
        Returns:
            Created product data including the assigned ID
        """
        # Ensure test prefix for cleanup identification
        if not product_data.get("title", "").startswith(TEST_PRODUCT_PREFIX):
            product_data["title"] = f"{TEST_PRODUCT_PREFIX}{product_data.get('title', 'Unnamed')}"
        
        # Add test tags for easy identification
        product_data.setdefault("tags", "paas-test,automated")
        
        logger.info(f"Creating Shopify product: {product_data.get('title')}")
        
        response = await self._request(
            "POST",
            "/products.json",
            json={"product": product_data},
        )
        
        product = response.get("product", {})
        logger.info(f"Created product ID: {product.get('id')}")
        
        return product
    
    async def get_product(self, product_id: int) -> Dict[str, Any]:
        """Get a product by ID.
        
        Args:
            product_id: Shopify product ID
            
        Returns:
            Product data
        """
        logger.debug(f"Getting product: {product_id}")
        
        response = await self._request("GET", f"/products/{product_id}.json")
        return response.get("product", {})
    
    async def update_product(
        self,
        product_id: int,
        product_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update a product.
        
        Args:
            product_id: Shopify product ID
            product_data: Updated product fields
            
        Returns:
            Updated product data
        """
        logger.info(f"Updating product: {product_id}")
        
        response = await self._request(
            "PUT",
            f"/products/{product_id}.json",
            json={"product": product_data},
        )
        
        return response.get("product", {})
    
    async def delete_product(self, product_id: int) -> bool:
        """Delete a product.
        
        Args:
            product_id: Shopify product ID
            
        Returns:
            True if deletion was successful
        """
        logger.info(f"Deleting product: {product_id}")
        
        try:
            await self._request("DELETE", f"/products/{product_id}.json")
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Product {product_id} not found (already deleted)")
                return True
            raise
    
    async def list_products(
        self,
        limit: int = 50,
        title_prefix: Optional[str] = None,
        created_at_min: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List products with optional filters.
        
        Args:
            limit: Maximum number of products to return
            title_prefix: Filter by title prefix
            created_at_min: Filter by minimum creation date (ISO format)
            
        Returns:
            List of product objects
        """
        params: Dict[str, Any] = {"limit": limit}
        
        if created_at_min:
            params["created_at_min"] = created_at_min
        
        response = await self._request("GET", "/products.json", params=params)
        products = response.get("products", [])
        
        # Filter by title prefix if specified
        if title_prefix:
            products = [
                p for p in products
                if p.get("title", "").startswith(title_prefix)
            ]
        
        return products
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def health_check(self) -> bool:
        """Check if the Shopify API is accessible.
        
        Returns:
            True if the API is accessible and credentials are valid
        """
        try:
            await self._request("GET", "/shop.json")
            return True
        except Exception as e:
            logger.error(f"Shopify health check failed: {e}")
            return False


# =============================================================================
# Factory Function
# =============================================================================

_client_instance: Optional[ShopifyClient] = None


def get_shopify_client() -> ShopifyClient:
    """Get or create the global Shopify client instance.
    
    Returns:
        ShopifyClient instance configured from settings
    """
    global _client_instance
    
    if _client_instance is None:
        _client_instance = ShopifyClient()
    
    return _client_instance


async def close_shopify_client():
    """Close the global Shopify client."""
    global _client_instance
    
    if _client_instance is not None:
        await _client_instance.close()
        _client_instance = None

