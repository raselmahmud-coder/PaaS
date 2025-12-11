"""Base agent classes and state definitions."""

from typing import TypedDict, List, Literal, Optional, Dict, Any, Callable
from functools import wraps
from langchain_core.messages import BaseMessage
from src.persistence.event_store import event_store


class AgentState(TypedDict):
    """Base state structure for all agents."""
    
    # Task identification
    task_id: str
    agent_id: str
    thread_id: str
    
    # Workflow tracking
    current_step: int
    status: Literal["pending", "in_progress", "completed", "failed"]
    
    # Message history
    messages: List[BaseMessage]
    
    # Product data (for Product Upload agent)
    product_data: Optional[Dict[str, Any]]
    
    # Generated content
    generated_listing: Optional[str]
    
    # Error information
    error: Optional[str]
    
    # Metadata
    metadata: Dict[str, Any]


class MarketingState(TypedDict):
    """Extended state for Marketing agent."""
    
    # Inherit from AgentState
    task_id: str
    agent_id: str
    thread_id: str
    current_step: int
    status: Literal["pending", "in_progress", "completed", "failed"]
    messages: List[BaseMessage]
    
    # Marketing-specific fields
    product_id: Optional[str]
    product_description: Optional[str]
    marketing_channel: Optional[str]
    marketing_copy: Optional[str]
    campaign_scheduled: bool
    error: Optional[str]
    metadata: Dict[str, Any]


def with_logging(node_func: Callable) -> Callable:
    """Decorator to log agent step execution."""
    @wraps(node_func)
    def wrapper(state, config=None):
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
            # Execute the node function
            result = node_func(state, config)
            
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
