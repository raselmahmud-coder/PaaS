"""Product Upload workflow using LangGraph."""

from langgraph.graph import END, StateGraph

from src.agents.base import AgentState
from src.agents.product_upload import (
    confirm_upload,
    generate_listing,
    validate_product_data,
)
from src.persistence.checkpointer import get_checkpointer


def create_product_upload_workflow(checkpointer=None):
    """Create the Product Upload workflow graph."""

    # Create state graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("validate_product_data", validate_product_data)
    workflow.add_node("generate_listing", generate_listing)
    workflow.add_node("confirm_upload", confirm_upload)

    # Define edges
    workflow.set_entry_point("validate_product_data")
    workflow.add_edge("validate_product_data", "generate_listing")
    workflow.add_edge("generate_listing", "confirm_upload")
    workflow.add_edge("confirm_upload", END)

    # Compile with checkpointer (use LangGraph's built-in)
    if checkpointer is None:
        checkpointer = get_checkpointer()

    return workflow.compile(checkpointer=checkpointer)
