"""Shopify Product Agent for real-world API validation.

This module provides async agent steps that interact with the real Shopify API,
enabling validation of PaaS resilience against actual e-commerce operations.

Each step includes chaos injection decorators to simulate various failure modes
during real API interactions.
"""

import asyncio
import logging
import random
from functools import wraps
from typing import Any, Callable, Dict, Optional

from langchain_core.messages import AIMessage

from src.agents.base import AgentState
from src.chaos.config import get_chaos_config
from src.chaos.exceptions import (
    AgentCrashException,
    AgentTimeoutException,
)
from src.integrations.shopify import get_shopify_client
from src.integrations.shopify.utils import generate_test_product_data
from src.llm import get_llm
from src.persistence.event_store import event_store

logger = logging.getLogger(__name__)


# =============================================================================
# Async Logging Decorator
# =============================================================================

def with_async_logging(node_func: Callable) -> Callable:
    """Decorator to log async agent step execution."""
    @wraps(node_func)
    async def wrapper(state: AgentState, config: Optional[Dict] = None) -> AgentState:
        agent_id = state.get("agent_id", "unknown")
        thread_id = state.get("thread_id", "unknown")
        step_name = node_func.__name__
        
        # Log step start
        event_store.log_event(
            agent_id=agent_id,
            thread_id=thread_id,
            event_type="step_start",
            step_name=step_name,
            input_data=dict(state),
            state_snapshot=dict(state),
        )
        
        try:
            # Execute the async node function
            result = await node_func(state, config)
            
            # Log step complete
            event_store.log_event(
                agent_id=agent_id,
                thread_id=thread_id,
                event_type="step_complete",
                step_name=step_name,
                output_data=dict(result) if isinstance(result, dict) else {"result": str(result)},
                state_snapshot=dict(result) if isinstance(result, dict) else state,
            )
            
            return result
        except Exception as e:
            # Log error
            event_store.log_event(
                agent_id=agent_id,
                thread_id=thread_id,
                event_type="error",
                step_name=step_name,
                input_data=dict(state),
                output_data={"error": str(e)},
                state_snapshot=dict(state),
            )
            raise
    
    return wrapper


# =============================================================================
# Async Chaos Decorators
# =============================================================================

def async_inject_crash(probability: float = 0.1, message: str = "Simulated crash"):
    """Async decorator to randomly inject crashes."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = get_chaos_config()
            if config.enabled and random.random() < probability:
                logger.warning(f"[CHAOS] Injecting crash in {func.__name__}: {message}")
                raise AgentCrashException(message)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def async_inject_timeout(probability: float = 0.1, timeout_seconds: float = 30.0):
    """Async decorator to simulate timeouts."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = get_chaos_config()
            if config.enabled and random.random() < probability:
                logger.warning(f"[CHAOS] Injecting timeout in {func.__name__}: {timeout_seconds}s")
                await asyncio.sleep(timeout_seconds)
                raise AgentTimeoutException(
                    f"Agent timed out after {timeout_seconds} seconds",
                    timeout_seconds=timeout_seconds,
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def async_inject_delay(probability: float = 0.2, delay_ms: int = 2000):
    """Async decorator to inject network-like delays."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            config = get_chaos_config()
            if config.enabled and random.random() < probability:
                delay_seconds = delay_ms / 1000.0
                logger.warning(f"[CHAOS] Injecting {delay_ms}ms delay in {func.__name__}")
                await asyncio.sleep(delay_seconds)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Extended State for Shopify
# =============================================================================

class ShopifyAgentState(AgentState):
    """Extended state for Shopify agent with API-specific fields."""
    
    # Shopify-specific fields
    shopify_product_id: Optional[int]
    shopify_variant_id: Optional[int]
    shopify_response: Optional[Dict[str, Any]]
    api_call_count: int
    api_errors: list


# =============================================================================
# Shopify Agent Steps
# =============================================================================

@with_async_logging
@async_inject_crash(probability=0.05)
@async_inject_delay(probability=0.15, delay_ms=1500)
async def validate_and_create_shopify(
    state: AgentState,
    config: Optional[Dict] = None,
) -> AgentState:
    """Step 1: Validate product data and create in Shopify.
    
    This step:
    1. Validates the incoming product data
    2. Maps it to Shopify's product format
    3. Creates the product via the Shopify Admin API
    
    Args:
        state: Current agent state with product_data
        config: Optional configuration
        
    Returns:
        Updated state with shopify_product_id
    """
    product_data = state.get("product_data", {})
    messages = list(state.get("messages", []))
    
    # Validate required fields
    required_fields = ["name", "price"]
    missing = [f for f in required_fields if f not in product_data]
    
    if missing:
        error_msg = f"Missing required fields: {', '.join(missing)}"
        messages.append(AIMessage(content=f"Validation failed: {error_msg}"))
        return {
            **state,
            "status": "failed",
            "error": error_msg,
            "messages": messages,
        }
    
    # Get Shopify client
    client = get_shopify_client()
    
    # Map to Shopify format
    shopify_product = generate_test_product_data(
        name=product_data["name"],
        price=float(product_data["price"]),
        category=product_data.get("category", "Test"),
        description=product_data.get("description"),
        sku=product_data.get("sku"),
    )
    
    # Create product in Shopify
    try:
        result = await client.create_product(shopify_product)
        
        product_id = result.get("id")
        variant_id = result.get("variants", [{}])[0].get("id")
        
        messages.append(AIMessage(
            content=f"Product created in Shopify: ID={product_id}, Title={result.get('title')}"
        ))
        
        return {
            **state,
            "status": "in_progress",
            "current_step": 1,
            "shopify_product_id": product_id,
            "shopify_variant_id": variant_id,
            "shopify_response": result,
            "messages": messages,
        }
    except Exception as e:
        logger.error(f"Failed to create Shopify product: {e}")
        messages.append(AIMessage(content=f"API Error: {str(e)}"))
        return {
            **state,
            "status": "failed",
            "error": str(e),
            "messages": messages,
        }


@with_async_logging
@async_inject_timeout(probability=0.05, timeout_seconds=60.0)
@async_inject_delay(probability=0.2, delay_ms=2000)
async def generate_listing_for_shopify(
    state: AgentState,
    config: Optional[Dict] = None,
) -> AgentState:
    """Step 2: Generate enhanced product listing using LLM.
    
    This step uses an LLM to generate compelling product copy
    that will be used to update the Shopify product.
    
    Args:
        state: Current agent state with product data
        config: Optional configuration
        
    Returns:
        Updated state with generated_listing
    """
    product_data = state.get("product_data", {})
    messages = list(state.get("messages", []))
    
    # Get LLM
    llm = get_llm(temperature=0.7)
    
    # Generate listing content
    prompt = f"""Create a compelling e-commerce product listing for Shopify.

Product Information:
- Name: {product_data.get('name', 'Unnamed Product')}
- Description: {product_data.get('description', 'No description provided')}
- Price: ${product_data.get('price', 0)}
- Category: {product_data.get('category', 'General')}

Generate:
1. An engaging product title (max 50 characters)
2. HTML-formatted product description (150-200 words)
3. 5 relevant tags (comma-separated)

Format your response as:
TITLE: [title]
DESCRIPTION: [html description]
TAGS: [tag1, tag2, tag3, tag4, tag5]
"""
    
    try:
        # Run LLM in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: llm.invoke(prompt)
        )
        
        listing_content = response.content
        
        messages.append(AIMessage(
            content=f"Generated listing content:\n{listing_content[:200]}..."
        ))
        
        return {
            **state,
            "status": "in_progress",
            "current_step": 2,
            "generated_listing": listing_content,
            "messages": messages,
        }
    except Exception as e:
        logger.error(f"Failed to generate listing: {e}")
        messages.append(AIMessage(content=f"LLM Error: {str(e)}"))
        return {
            **state,
            "status": "failed",
            "error": str(e),
            "messages": messages,
        }


@with_async_logging
@async_inject_crash(probability=0.03)
@async_inject_delay(probability=0.1, delay_ms=1000)
async def update_shopify_product(
    state: AgentState,
    config: Optional[Dict] = None,
) -> AgentState:
    """Step 3: Update Shopify product with generated listing.
    
    This step takes the LLM-generated content and updates
    the Shopify product with the enhanced listing.
    
    Args:
        state: Current agent state with shopify_product_id and generated_listing
        config: Optional configuration
        
    Returns:
        Updated state with updated product info
    """
    product_id = state.get("shopify_product_id")
    generated_listing = state.get("generated_listing", "")
    messages = list(state.get("messages", []))
    
    if not product_id:
        error_msg = "No Shopify product ID found"
        messages.append(AIMessage(content=f"Error: {error_msg}"))
        return {
            **state,
            "status": "failed",
            "error": error_msg,
            "messages": messages,
        }
    
    # Parse generated listing
    update_data = {}
    
    lines = generated_listing.split("\n")
    for line in lines:
        if line.startswith("TITLE:"):
            update_data["title"] = f"PAAS_TEST_{line.replace('TITLE:', '').strip()}"
        elif line.startswith("DESCRIPTION:"):
            update_data["body_html"] = line.replace("DESCRIPTION:", "").strip()
        elif line.startswith("TAGS:"):
            update_data["tags"] = line.replace("TAGS:", "").strip()
    
    # Get client and update
    client = get_shopify_client()
    
    try:
        result = await client.update_product(product_id, update_data)
        
        messages.append(AIMessage(
            content=f"Updated Shopify product {product_id} with new listing"
        ))
        
        return {
            **state,
            "status": "in_progress",
            "current_step": 3,
            "shopify_response": result,
            "messages": messages,
        }
    except Exception as e:
        logger.error(f"Failed to update Shopify product: {e}")
        messages.append(AIMessage(content=f"API Error: {str(e)}"))
        return {
            **state,
            "status": "failed",
            "error": str(e),
            "messages": messages,
        }


@with_async_logging
@async_inject_delay(probability=0.1, delay_ms=500)
async def cleanup_test_product(
    state: AgentState,
    config: Optional[Dict] = None,
) -> AgentState:
    """Step 4: Clean up test product from Shopify.
    
    This step deletes the test product created during the experiment
    to keep the sandbox clean.
    
    Args:
        state: Current agent state with shopify_product_id
        config: Optional configuration
        
    Returns:
        Updated state with completed status
    """
    product_id = state.get("shopify_product_id")
    messages = list(state.get("messages", []))
    
    if not product_id:
        # No product to clean up, consider success
        messages.append(AIMessage(content="No product to clean up"))
        return {
            **state,
            "status": "completed",
            "current_step": 4,
            "messages": messages,
        }
    
    # Get client and delete
    client = get_shopify_client()
    
    try:
        success = await client.delete_product(product_id)
        
        if success:
            messages.append(AIMessage(
                content=f"Cleaned up Shopify product {product_id}"
            ))
        else:
            messages.append(AIMessage(
                content=f"Product {product_id} may already be deleted"
            ))
        
        return {
            **state,
            "status": "completed",
            "current_step": 4,
            "shopify_product_id": None,  # Clear the ID
            "messages": messages,
        }
    except Exception as e:
        logger.warning(f"Cleanup error (non-fatal): {e}")
        messages.append(AIMessage(content=f"Cleanup warning: {str(e)}"))
        
        # Still mark as completed since cleanup failure is non-critical
        return {
            **state,
            "status": "completed",
            "current_step": 4,
            "messages": messages,
        }


# =============================================================================
# Health Check Step
# =============================================================================

async def shopify_health_check(state: AgentState, config: Optional[Dict] = None) -> AgentState:
    """Check Shopify API connectivity before starting workflow.
    
    Args:
        state: Current agent state
        config: Optional configuration
        
    Returns:
        Updated state with API status
    """
    client = get_shopify_client()
    messages = list(state.get("messages", []))
    
    try:
        healthy = await client.health_check()
        
        if healthy:
            messages.append(AIMessage(content="Shopify API connection verified"))
            return {
                **state,
                "status": "in_progress",
                "metadata": {**state.get("metadata", {}), "shopify_healthy": True},
                "messages": messages,
            }
        else:
            messages.append(AIMessage(content="Shopify API health check failed"))
            return {
                **state,
                "status": "failed",
                "error": "Shopify API health check failed",
                "messages": messages,
            }
    except Exception as e:
        logger.error(f"Shopify health check error: {e}")
        return {
            **state,
            "status": "failed",
            "error": f"Shopify connection error: {e}",
            "messages": messages,
        }

