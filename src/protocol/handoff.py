"""Protocol-aware handoff utilities for inter-agent communication."""

from typing import Dict, Any, Optional
from src.protocol.messages import (
    TaskAssignMessage,
    TaskCompleteMessage,
    AgentMessage,
)
from src.agents.base import AgentState


def create_task_assign_message(
    sender: str,
    receiver: str,
    state: AgentState,
    task_description: Optional[str] = None,
) -> TaskAssignMessage:
    """
    Create a TASK_ASSIGN message from agent state.
    
    Args:
        sender: ID of the agent sending the task
        receiver: ID of the agent receiving the task
        state: Current agent state to include in task payload
        task_description: Optional description of the task
        
    Returns:
        TaskAssignMessage instance
    """
    payload = {
        "state": dict(state),
        "task_description": task_description or f"Task from {sender}",
        "agent_id": state.get("agent_id"),
        "thread_id": state.get("thread_id"),
        "current_step": state.get("current_step", 0),
    }
    
    return TaskAssignMessage(
        sender=sender,
        receiver=receiver,
        payload=payload,
    )


def create_task_complete_message(
    sender: str,
    receiver: str,
    state: AgentState,
    completion_status: Optional[str] = None,
) -> TaskCompleteMessage:
    """
    Create a TASK_COMPLETE message from agent state.
    
    Args:
        sender: ID of the agent completing the task
        receiver: ID of the agent receiving completion notification
        state: Final agent state after task completion
        completion_status: Optional status message
        
    Returns:
        TaskCompleteMessage instance
    """
    payload = {
        "state": dict(state),
        "completion_status": completion_status or state.get("status", "completed"),
        "agent_id": state.get("agent_id"),
        "thread_id": state.get("thread_id"),
        "current_step": state.get("current_step", 0),
        "final_status": state.get("status"),
    }
    
    return TaskCompleteMessage(
        sender=sender,
        receiver=receiver,
        payload=payload,
    )


def extract_state_from_message(message: AgentMessage) -> AgentState:
    """
    Extract agent state from a protocol message.
    
    Args:
        message: Protocol message containing state in payload
        
    Returns:
        AgentState dictionary extracted from message payload
        
    Raises:
        ValueError: If message payload doesn't contain state
    """
    if not message.payload:
        raise ValueError("Message payload is empty")
    
    # Check if payload contains state directly
    if "state" in message.payload:
        state = message.payload["state"]
        if isinstance(state, dict):
            return state
    
    # Fallback: use payload as state if it has required fields
    if all(key in message.payload for key in ["agent_id", "thread_id"]):
        return message.payload
    
    raise ValueError(
        f"Message payload does not contain valid state. "
        f"Expected 'state' key or required fields (agent_id, thread_id)"
    )


def message_to_state_dict(message: AgentMessage) -> Dict[str, Any]:
    """
    Convert a protocol message to a state dictionary for workflow use.
    
    This is a convenience function that extracts state and merges message metadata.
    
    Args:
        message: Protocol message
        
    Returns:
        Dictionary suitable for use as AgentState
    """
    state = extract_state_from_message(message)
    
    # Add protocol message metadata
    state["_protocol_message_id"] = message.message_id
    state["_protocol_message_type"] = message.message_type
    state["_protocol_sender"] = message.sender
    state["_protocol_receiver"] = message.receiver
    state["_protocol_timestamp"] = message.timestamp.isoformat()
    
    return state


def state_dict_to_message_data(
    state: Dict[str, Any],
    message_type: str,
    sender: str,
    receiver: str,
) -> Dict[str, Any]:
    """
    Convert a state dictionary to message data format.
    
    Args:
        state: Agent state dictionary
        message_type: Type of message to create
        sender: Sender agent ID
        receiver: Receiver agent ID
        
    Returns:
        Dictionary ready for message creation
    """
    # Remove protocol metadata if present
    clean_state = {
        k: v
        for k, v in state.items()
        if not k.startswith("_protocol_")
    }
    
    return {
        "message_type": message_type,
        "sender": sender,
        "receiver": receiver,
        "payload": {
            "state": clean_state,
            "agent_id": clean_state.get("agent_id"),
            "thread_id": clean_state.get("thread_id"),
        },
    }

