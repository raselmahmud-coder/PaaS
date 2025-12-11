"""Vendor workflow combining Product Upload and Marketing agents."""

from langgraph.graph import StateGraph, END
from typing import Dict, Any
from src.agents.base import AgentState
from src.agents.product_upload import (
    validate_product_data,
    generate_listing,
    confirm_upload
)
from src.persistence.checkpointer import get_checkpointer
from src.agents.base import with_logging


@with_logging
def generate_marketing_copy(state: AgentState) -> AgentState:
    """Generate marketing copy from product listing."""
    from langchain_core.messages import HumanMessage, AIMessage
    from src.llm import get_llm
    
    llm = get_llm(temperature=0.7)
    
    product_description = state.get("generated_listing", "")
    product_id = state.get("product_data", {}).get("sku", "unknown")
    messages = state.get("messages", [])
    
    prompt = f"""Create compelling marketing copy for a product launch.

Product Description:
{product_description}

Marketing Channel: email
Product ID: {product_id}

Generate marketing copy that:
1. Captures attention in the first sentence
2. Highlights key benefits and features
3. Creates urgency or desire
4. Includes a clear call-to-action
5. Is appropriate for email channel

Return only the marketing copy text, no additional commentary."""

    response = llm.invoke([HumanMessage(content=prompt)])
    marketing_copy = response.content
    
    messages.append(
        AIMessage(content=f"Generated marketing copy:\n{marketing_copy}")
    )
    
    return {
        **state,
        "status": "in_progress",
        "current_step": 4,
        "metadata": {
            **state.get("metadata", {}),
            "marketing_copy": marketing_copy,
        },
        "messages": messages,
    }


@with_logging
def schedule_campaign(state: AgentState) -> AgentState:
    """Schedule marketing campaign (mock)."""
    from langchain_core.messages import AIMessage
    
    marketing_copy = state.get("metadata", {}).get("marketing_copy", "")
    product_id = state.get("product_data", {}).get("sku", "unknown")
    messages = state.get("messages", [])
    
    schedule_message = (
        f"Marketing campaign scheduled successfully!\n"
        f"Channel: email\n"
        f"Product ID: {product_id}\n"
        f"Campaign will launch in 24 hours.\n"
        f"Copy preview: {marketing_copy[:100]}..."
    )
    
    messages.append(AIMessage(content=schedule_message))
    
    return {
        **state,
        "status": "completed",
        "current_step": 5,
        "metadata": {
            **state.get("metadata", {}),
            "campaign_scheduled": True,
        },
        "messages": messages,
    }


def create_vendor_workflow(checkpointer=None):
    """Create the complete vendor workflow (Product Upload → Marketing)."""
    
    # Create state graph
    workflow = StateGraph(AgentState)
    
    # Add Product Upload nodes
    workflow.add_node("validate_product_data", validate_product_data)
    workflow.add_node("generate_listing", generate_listing)
    workflow.add_node("confirm_upload", confirm_upload)
    
    # Add Marketing nodes
    workflow.add_node("generate_marketing_copy", generate_marketing_copy)
    workflow.add_node("schedule_campaign", schedule_campaign)
    
    # Define edges
    workflow.set_entry_point("validate_product_data")
    workflow.add_edge("validate_product_data", "generate_listing")
    workflow.add_edge("generate_listing", "confirm_upload")
    workflow.add_edge("confirm_upload", "generate_marketing_copy")
    workflow.add_edge("generate_marketing_copy", "schedule_campaign")
    workflow.add_edge("schedule_campaign", END)
    
    # Compile with checkpointer (use LangGraph's built-in)
    if checkpointer is None:
        checkpointer = get_checkpointer()
    
    return workflow.compile(checkpointer=checkpointer)

