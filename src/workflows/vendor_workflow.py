"""Vendor workflow combining Product Upload and Marketing agents."""

from langgraph.graph import StateGraph, END
from typing import Dict, Any
from src.agents.base import AgentState, with_logging
from src.agents.product_upload import (
    validate_product_data,
    generate_listing,
    confirm_upload
)
from src.persistence.checkpointer import get_checkpointer
from src.protocol.handoff import (
    create_task_complete_message,
    create_task_assign_message,
    extract_state_from_message,
    message_to_state_dict,
)
from src.protocol.validator import validate_message_structure


@with_logging
def protocol_handoff_product_to_marketing(state: AgentState, config=None) -> AgentState:
    """
    Protocol-aware handoff node: Product Upload → Marketing.
    Converts Product agent completion to Marketing agent task assignment.
    """
    from src.persistence.event_store import event_store
    
    product_agent_id = state.get("agent_id", "product-agent-1")
    marketing_agent_id = "marketing-agent-1"
    thread_id = state.get("thread_id", "unknown")
    
    # Create TASK_COMPLETE message from Product agent
    task_complete_msg = create_task_complete_message(
        sender=product_agent_id,
        receiver=marketing_agent_id,
        state=state,
        completion_status="Product upload completed, ready for marketing",
    )
    
    # Validate message
    validate_message_structure(task_complete_msg)
    
    # Create TASK_ASSIGN message for Marketing agent
    task_assign_msg = create_task_assign_message(
        sender=product_agent_id,
        receiver=marketing_agent_id,
        state=state,
        task_description="Generate marketing copy for uploaded product",
    )
    
    # Validate message
    validate_message_structure(task_assign_msg)
    
    # Extract state from TASK_ASSIGN message (for Marketing agent)
    marketing_state = message_to_state_dict(task_assign_msg)
    
    # Update agent_id to marketing agent
    marketing_state["agent_id"] = marketing_agent_id
    
    # Store protocol messages in metadata for logging
    marketing_state["metadata"] = {
        **marketing_state.get("metadata", {}),
        "_protocol_messages": {
            "task_complete": task_complete_msg.model_dump(),
            "task_assign": task_assign_msg.model_dump(),
        },
    }
    
    # Log protocol handoff event with protocol messages
    event_store.log_event(
        agent_id=product_agent_id,
        thread_id=thread_id,
        event_type="protocol_handoff",
        step_name="product_to_marketing",
        input_data={"task_complete_message_id": task_complete_msg.message_id},
        output_data={"task_assign_message_id": task_assign_msg.message_id},
        state_snapshot=marketing_state,
        protocol_message=task_complete_msg,  # Log the completion message
    )
    
    # Also log the task assignment from marketing agent perspective
    event_store.log_event(
        agent_id=marketing_agent_id,
        thread_id=thread_id,
        event_type="protocol_receive",
        step_name="product_to_marketing",
        input_data={"task_assign_message_id": task_assign_msg.message_id},
        output_data=None,
        state_snapshot=marketing_state,
        protocol_message=task_assign_msg,  # Log the assignment message
    )
    
    return marketing_state


@with_logging
def generate_marketing_copy(state: AgentState, config=None) -> AgentState:
    """
    Generate marketing copy from product listing.
    Can accept state directly or extract from protocol message.
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from src.llm import get_llm
    
    # Extract state if it came from protocol message
    if "_protocol_message_type" in state:
        # State came from protocol message, use as-is
        pass
    elif "metadata" in state and "_protocol_messages" in state.get("metadata", {}):
        # Protocol messages stored in metadata, state already extracted
        pass
    
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
def schedule_campaign(state: AgentState, config=None) -> AgentState:
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
    
    # Add protocol handoff node
    workflow.add_node("protocol_handoff", protocol_handoff_product_to_marketing)
    
    # Define edges
    workflow.set_entry_point("validate_product_data")
    workflow.add_edge("validate_product_data", "generate_listing")
    workflow.add_edge("generate_listing", "confirm_upload")
    # Use protocol handoff instead of direct edge
    workflow.add_edge("confirm_upload", "protocol_handoff")
    workflow.add_edge("protocol_handoff", "generate_marketing_copy")
    workflow.add_edge("generate_marketing_copy", "schedule_campaign")
    workflow.add_edge("schedule_campaign", END)
    
    # Compile with checkpointer (use LangGraph's built-in)
    if checkpointer is None:
        checkpointer = get_checkpointer()
    
    return workflow.compile(checkpointer=checkpointer)

