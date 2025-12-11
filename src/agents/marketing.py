"""Marketing Agent implementation."""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from src.agents.base import MarketingState, with_logging
from src.llm import get_llm


# Initialize LLM using centralized provider
llm = get_llm(temperature=0.7)


@with_logging
def generate_marketing_copy(state: MarketingState, config=None) -> MarketingState:
    """Step 1: Generate marketing copy from product description."""
    
    product_description = state.get("product_description", "")
    product_id = state.get("product_id", "unknown")
    marketing_channel = state.get("marketing_channel", "email")
    messages = state.get("messages", [])
    
    # Build prompt for LLM
    prompt = f"""Create compelling marketing copy for a product launch.

Product Description:
{product_description}

Marketing Channel: {marketing_channel}
Product ID: {product_id}

Generate marketing copy that:
1. Captures attention in the first sentence
2. Highlights key benefits and features
3. Creates urgency or desire
4. Includes a clear call-to-action
5. Is appropriate for {marketing_channel} channel

Return only the marketing copy text, no additional commentary."""

    # Call LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    marketing_copy = response.content
    
    messages.append(
        AIMessage(content=f"Generated marketing copy for {marketing_channel}:\n{marketing_copy}")
    )
    
    return {
        **state,
        "status": "in_progress",
        "current_step": 1,
        "marketing_copy": marketing_copy,
        "messages": messages,
    }


@with_logging
def schedule_campaign(state: MarketingState, config=None) -> MarketingState:
    """Step 2: Schedule marketing campaign (mock)."""
    
    marketing_copy = state.get("marketing_copy", "")
    marketing_channel = state.get("marketing_channel", "email")
    product_id = state.get("product_id", "unknown")
    messages = state.get("messages", [])
    
    # Simulate campaign scheduling
    schedule_message = (
        f"Marketing campaign scheduled successfully!\n"
        f"Channel: {marketing_channel}\n"
        f"Product ID: {product_id}\n"
        f"Campaign will launch in 24 hours.\n"
        f"Copy preview: {marketing_copy[:100]}..."
    )
    
    messages.append(AIMessage(content=schedule_message))
    
    return {
        **state,
        "status": "completed",
        "current_step": 2,
        "campaign_scheduled": True,
        "messages": messages,
    }

