"""Product Upload Agent implementation."""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.base import AgentState, with_logging
from src.llm import get_llm


# Initialize LLM using centralized provider
llm = get_llm(temperature=0.7)


def _maybe_fail(config: Any, step: str):
    """Test-only hook to inject failures at a given step."""
    fail_step = None
    try:
        fail_step = (config or {}).get("configurable", {}).get("fail_step")
    except Exception:
        fail_step = None
    if fail_step and str(fail_step) == step:
        raise RuntimeError(f"Injected failure at step: {step}")


@with_logging
def validate_product_data(state: AgentState, config=None) -> AgentState:
    """Step 1: Validate product data completeness."""

    _maybe_fail(config, "validate_product_data")
    
    product_data = state.get("product_data", {})
    
    # Required fields
    required_fields = ["name", "price", "category"]
    missing_fields = [field for field in required_fields if field not in product_data]
    
    if missing_fields:
        error_msg = f"Missing required fields: {', '.join(missing_fields)}"
        return {
            **state,
            "status": "failed",
            "error": error_msg,
            "current_step": 1,
        }
    
    # Add validation message
    messages = state.get("messages", [])
    messages.append(
        AIMessage(content=f"Product data validated successfully. Product: {product_data.get('name')}")
    )
    
    return {
        **state,
        "status": "in_progress",
        "current_step": 1,
        "messages": messages,
    }


@with_logging
def generate_listing(state: AgentState, config=None) -> AgentState:
    """Step 2: Generate product listing using LLM."""

    _maybe_fail(config, "generate_listing")
    
    product_data = state.get("product_data", {})
    messages = state.get("messages", [])
    
    # Build prompt for LLM
    prompt = f"""Create a compelling product listing for an e-commerce marketplace.

Product Information:
- Name: {product_data.get('name')}
- Description: {product_data.get('description', 'No description provided')}
- Price: ${product_data.get('price')}
- Category: {product_data.get('category')}
- SKU: {product_data.get('sku', 'N/A')}

Generate a professional product listing description that:
1. Highlights key features and benefits
2. Uses persuasive language
3. Is suitable for online marketplace
4. Is approximately 150-200 words

Return only the listing text, no additional commentary."""

    # Call LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    listing_text = response.content
    
    messages.append(
        AIMessage(content=f"Generated product listing:\n{listing_text}")
    )
    
    return {
        **state,
        "status": "in_progress",
        "current_step": 2,
        "generated_listing": listing_text,
        "messages": messages,
    }


@with_logging
def confirm_upload(state: AgentState, config=None) -> AgentState:
    """Step 3: Confirm upload and finalize."""

    _maybe_fail(config, "confirm_upload")
    
    product_data = state.get("product_data", {})
    generated_listing = state.get("generated_listing", "")
    messages = state.get("messages", [])
    
    # Simulate upload confirmation
    confirmation_message = (
        f"Product '{product_data.get('name')}' has been successfully uploaded.\n"
        f"Listing generated and ready for publication.\n"
        f"SKU: {product_data.get('sku', 'N/A')}"
    )
    
    messages.append(AIMessage(content=confirmation_message))
    
    return {
        **state,
        "status": "completed",
        "current_step": 3,
        "messages": messages,
    }

