"""Shopify Product Workflow using LangGraph for real API operations.

This workflow orchestrates real Shopify API operations for product management,
enabling validation of PaaS resilience against actual e-commerce systems.

The workflow includes:
1. Health check - Verify Shopify API connectivity
2. Create product - Create a test product in Shopify
3. Generate listing - Use LLM to generate enhanced product content
4. Update product - Update Shopify product with generated content
5. Cleanup - Delete test product after completion
"""

import logging
from typing import Optional

from langgraph.graph import END, StateGraph

from src.agents.base import AgentState
from src.agents.shopify_product import (
    cleanup_test_product,
    generate_listing_for_shopify,
    shopify_health_check,
    update_shopify_product,
    validate_and_create_shopify,
)
from src.persistence.checkpointer import get_checkpointer

logger = logging.getLogger(__name__)


def _should_continue(state: AgentState) -> str:
    """Determine if workflow should continue or stop.

    Args:
        state: Current agent state

    Returns:
        Next node name or END
    """
    status = state.get("status", "pending")

    if status == "failed":
        # On failure, try to cleanup and end
        if state.get("shopify_product_id"):
            return "cleanup"
        return END

    if status == "completed":
        return END

    return "continue"


def _route_after_health(state: AgentState) -> str:
    """Route after health check."""
    if state.get("status") == "failed":
        return END
    return "create"


def _route_after_create(state: AgentState) -> str:
    """Route after product creation."""
    if state.get("status") == "failed":
        return END
    return "generate"


def _route_after_generate(state: AgentState) -> str:
    """Route after listing generation."""
    if state.get("status") == "failed":
        # If we have a product ID, clean it up
        if state.get("shopify_product_id"):
            return "cleanup"
        return END
    return "update"


def _route_after_update(state: AgentState) -> str:
    """Route after product update."""
    # Always try to cleanup
    return "cleanup"


def create_shopify_product_workflow(checkpointer=None) -> StateGraph:
    """Create the Shopify Product workflow graph.

    This workflow performs real Shopify API operations:
    1. Health check - Verify API connectivity
    2. Create product - Create test product in Shopify
    3. Generate listing - Use LLM to create enhanced content
    4. Update product - Apply generated content to Shopify
    5. Cleanup - Delete test product

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled workflow graph
    """
    # Create state graph
    workflow = StateGraph(AgentState)

    # Add nodes - wrap async functions for LangGraph
    workflow.add_node("health_check", _wrap_async(shopify_health_check))
    workflow.add_node("create", _wrap_async(validate_and_create_shopify))
    workflow.add_node("generate", _wrap_async(generate_listing_for_shopify))
    workflow.add_node("update", _wrap_async(update_shopify_product))
    workflow.add_node("cleanup", _wrap_async(cleanup_test_product))

    # Set entry point
    workflow.set_entry_point("health_check")

    # Add conditional edges
    workflow.add_conditional_edges(
        "health_check",
        _route_after_health,
        {
            "create": "create",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "create",
        _route_after_create,
        {
            "generate": "generate",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "generate",
        _route_after_generate,
        {
            "update": "update",
            "cleanup": "cleanup",
            END: END,
        },
    )

    workflow.add_conditional_edges(
        "update",
        _route_after_update,
        {
            "cleanup": "cleanup",
        },
    )

    # Cleanup always ends
    workflow.add_edge("cleanup", END)

    # Compile with checkpointer
    if checkpointer is None:
        checkpointer = get_checkpointer()

    return workflow.compile(checkpointer=checkpointer)


def _wrap_async(async_func):
    """Wrap an async function for use in LangGraph.

    LangGraph supports async nodes, but we need to ensure proper
    event loop handling.

    Args:
        async_func: Async function to wrap

    Returns:
        Wrapped function
    """

    async def wrapper(state: AgentState, config=None) -> AgentState:
        return await async_func(state, config)

    return wrapper


# =============================================================================
# Simplified Workflow (for faster testing)
# =============================================================================


def create_simple_shopify_workflow(checkpointer=None) -> StateGraph:
    """Create a simplified Shopify workflow for quick tests.

    This workflow only includes:
    1. Create product
    2. Cleanup

    Useful for quick validation without LLM calls.

    Args:
        checkpointer: Optional checkpointer

    Returns:
        Compiled workflow graph
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("create", _wrap_async(validate_and_create_shopify))
    workflow.add_node("cleanup", _wrap_async(cleanup_test_product))

    workflow.set_entry_point("create")

    workflow.add_conditional_edges(
        "create",
        lambda s: "cleanup" if s.get("shopify_product_id") else END,
        {
            "cleanup": "cleanup",
            END: END,
        },
    )

    workflow.add_edge("cleanup", END)

    if checkpointer is None:
        checkpointer = get_checkpointer()

    return workflow.compile(checkpointer=checkpointer)


# =============================================================================
# Utility Functions
# =============================================================================


async def run_shopify_workflow(
    product_data: dict,
    task_id: Optional[str] = None,
    agent_id: str = "shopify-agent",
    thread_id: Optional[str] = None,
    use_simple: bool = False,
) -> AgentState:
    """Convenience function to run the Shopify workflow.

    Args:
        product_data: Product data to create (name, price, category, etc.)
        task_id: Optional task ID
        agent_id: Agent identifier
        thread_id: Thread identifier
        use_simple: If True, use simplified workflow (no LLM)

    Returns:
        Final agent state
    """
    import uuid

    from src.persistence.checkpointer import get_async_checkpointer

    task_id = task_id or str(uuid.uuid4())
    thread_id = thread_id or f"shopify-{task_id}"

    initial_state: AgentState = {
        "task_id": task_id,
        "agent_id": agent_id,
        "thread_id": thread_id,
        "current_step": 0,
        "status": "pending",
        "messages": [],
        "product_data": product_data,
        "generated_listing": None,
        "error": None,
        "metadata": {},
    }

    # Get async checkpointer
    checkpointer = await get_async_checkpointer()

    # Create workflow with async checkpointer
    if use_simple:
        workflow = create_simple_shopify_workflow(checkpointer=checkpointer)
    else:
        workflow = create_shopify_product_workflow(checkpointer=checkpointer)

    # Run workflow
    config = {"configurable": {"thread_id": thread_id}}

    final_state = None
    async for event in workflow.astream(initial_state, config):
        for node_name, state in event.items():
            logger.debug(f"Shopify workflow step: {node_name}")
            final_state = state

    return final_state
