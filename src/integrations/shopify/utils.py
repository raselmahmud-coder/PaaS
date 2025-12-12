"""Utility functions for Shopify integration.

This module provides rate limiting, cleanup utilities, and helper functions
for safe Shopify API interactions during experiments.
"""

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from src.integrations.shopify.client import ShopifyClient

logger = logging.getLogger(__name__)

# Default test product prefix
TEST_PRODUCT_PREFIX = "PAAS_TEST_"


class RateLimiter:
    """Async rate limiter for Shopify API calls.
    
    Shopify's Admin API has a rate limit of 40 requests per second for
    standard plans. This limiter uses a conservative default of 2 req/s
    to avoid any throttling during experiments.
    
    Usage:
        limiter = RateLimiter(requests_per_second=2)
        await limiter.acquire()  # Blocks if rate limit would be exceeded
        # ... make API call ...
    
    Attributes:
        rate: Maximum requests per second
        timestamps: Deque of recent request timestamps
        _lock: Async lock for thread safety
    """
    
    def __init__(self, requests_per_second: int = 2):
        """Initialize the rate limiter.
        
        Args:
            requests_per_second: Maximum requests allowed per second
        """
        self.rate = requests_per_second
        self.timestamps: deque = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request.
        
        Blocks if the rate limit would be exceeded, waiting until
        a slot becomes available.
        """
        async with self._lock:
            now = time.time()
            
            # Remove timestamps older than 1 second
            while self.timestamps and now - self.timestamps[0] > 1.0:
                self.timestamps.popleft()
            
            # If at capacity, wait for the oldest request to expire
            if len(self.timestamps) >= self.rate:
                oldest = self.timestamps[0]
                sleep_time = 1.0 - (now - oldest)
                if sleep_time > 0:
                    logger.debug(f"Rate limiter sleeping for {sleep_time:.3f}s")
                    await asyncio.sleep(sleep_time)
                
                # Refresh timestamp after sleep
                now = time.time()
                while self.timestamps and now - self.timestamps[0] > 1.0:
                    self.timestamps.popleft()
            
            # Record this request
            self.timestamps.append(time.time())
    
    def reset(self) -> None:
        """Reset the rate limiter (clear all timestamps)."""
        self.timestamps.clear()


async def cleanup_test_products(
    client: "ShopifyClient",
    prefix: str = TEST_PRODUCT_PREFIX,
    dry_run: bool = False,
) -> int:
    """Clean up all test products from the Shopify store.
    
    This function identifies and deletes all products with the specified
    prefix in their title. Used after experiments to clean up test data.
    
    Args:
        client: Shopify client instance
        prefix: Title prefix to identify test products
        dry_run: If True, only log what would be deleted without deleting
        
    Returns:
        Number of products deleted (or would be deleted in dry run)
    """
    logger.info(f"Cleaning up test products with prefix: {prefix}")
    
    # Fetch all products with test prefix
    products = await client.list_products(limit=250, title_prefix=prefix)
    
    if not products:
        logger.info("No test products found to clean up")
        return 0
    
    logger.info(f"Found {len(products)} test products to clean up")
    
    deleted_count = 0
    for product in products:
        product_id = product.get("id")
        product_title = product.get("title", "Unknown")
        
        if dry_run:
            logger.info(f"[DRY RUN] Would delete: {product_title} (ID: {product_id})")
        else:
            try:
                await client.delete_product(product_id)
                logger.debug(f"Deleted: {product_title} (ID: {product_id})")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {product_title}: {e}")
    
    if dry_run:
        logger.info(f"[DRY RUN] Would delete {len(products)} products")
        return len(products)
    else:
        logger.info(f"Cleaned up {deleted_count} test products")
        return deleted_count


async def batch_cleanup(
    client: "ShopifyClient",
    product_ids: List[int],
    ignore_errors: bool = True,
) -> int:
    """Delete a batch of products by their IDs.
    
    Args:
        client: Shopify client instance
        product_ids: List of product IDs to delete
        ignore_errors: If True, continue on deletion errors
        
    Returns:
        Number of products successfully deleted
    """
    deleted = 0
    
    for product_id in product_ids:
        try:
            await client.delete_product(product_id)
            deleted += 1
        except Exception as e:
            if ignore_errors:
                logger.warning(f"Failed to delete product {product_id}: {e}")
            else:
                raise
    
    return deleted


def generate_test_product_data(
    name: str,
    price: float,
    category: str = "Test",
    description: Optional[str] = None,
    sku: Optional[str] = None,
) -> dict:
    """Generate test product data for Shopify.
    
    Args:
        name: Product name (will be prefixed with TEST_PRODUCT_PREFIX)
        price: Product price
        category: Product category/type
        description: Product description HTML
        sku: Stock keeping unit
        
    Returns:
        Dictionary ready for Shopify product creation
    """
    import uuid
    
    sku = sku or f"PAAS-{uuid.uuid4().hex[:8].upper()}"
    description = description or f"<p>Test product created by PaaS experiments.</p>"
    
    return {
        "title": f"{TEST_PRODUCT_PREFIX}{name}",
        "body_html": description,
        "vendor": "PaaS Test",
        "product_type": category,
        "tags": "paas-test,automated,experiment",
        "status": "draft",  # Keep as draft to avoid publishing
        "variants": [
            {
                "price": str(price),
                "sku": sku,
                "inventory_management": None,  # Don't track inventory
                "requires_shipping": False,
            }
        ],
    }

