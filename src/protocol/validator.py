"""Message validation utilities for protocol messages."""

from typing import Dict, Any, Optional
from src.protocol.messages import (
    AgentMessage,
    TaskAssignMessage,
    TaskCompleteMessage,
    RequestContextMessage,
    ProvideContextMessage,
)


class MessageValidationError(Exception):
    """Raised when message validation fails."""
    pass


def validate_message(message_data: Dict[str, Any]) -> AgentMessage:
    """
    Validate and parse a message dictionary into a protocol message object.
    
    Args:
        message_data: Dictionary containing message data
        
    Returns:
        Validated AgentMessage instance (subclass based on message_type)
        
    Raises:
        MessageValidationError: If message is invalid or missing required fields
    """
    if not isinstance(message_data, dict):
        raise MessageValidationError(f"Message must be a dictionary, got {type(message_data)}")
    
    # Check required fields
    required_fields = ["message_type", "sender", "receiver"]
    missing_fields = [field for field in required_fields if field not in message_data]
    if missing_fields:
        raise MessageValidationError(
            f"Missing required fields: {', '.join(missing_fields)}"
        )
    
    message_type = message_data.get("message_type")
    
    # Validate message type and create appropriate message class
    try:
        if message_type == "TASK_ASSIGN":
            return TaskAssignMessage(**message_data)
        elif message_type == "TASK_COMPLETE":
            return TaskCompleteMessage(**message_data)
        elif message_type == "REQUEST_CONTEXT":
            return RequestContextMessage(**message_data)
        elif message_type == "PROVIDE_CONTEXT":
            return ProvideContextMessage(**message_data)
        else:
            raise MessageValidationError(
                f"Unknown message type: {message_type}. "
                f"Expected one of: TASK_ASSIGN, TASK_COMPLETE, REQUEST_CONTEXT, PROVIDE_CONTEXT"
            )
    except Exception as e:
        if isinstance(e, MessageValidationError):
            raise
        raise MessageValidationError(f"Failed to create message: {str(e)}")


def validate_message_structure(message: AgentMessage) -> bool:
    """
    Validate that a message object has correct structure.
    
    Args:
        message: AgentMessage instance to validate
        
    Returns:
        True if valid
        
    Raises:
        MessageValidationError: If message structure is invalid
    """
    if not isinstance(message, AgentMessage):
        raise MessageValidationError(
            f"Message must be an instance of AgentMessage, got {type(message)}"
        )
    
    # Validate required fields are present
    if not message.message_id:
        raise MessageValidationError("message_id is required")
    
    if not message.sender:
        raise MessageValidationError("sender is required")
    
    if not message.receiver:
        raise MessageValidationError("receiver is required")
    
    # Validate message type matches class
    expected_type = None
    if isinstance(message, TaskAssignMessage):
        expected_type = "TASK_ASSIGN"
    elif isinstance(message, TaskCompleteMessage):
        expected_type = "TASK_COMPLETE"
    elif isinstance(message, RequestContextMessage):
        expected_type = "REQUEST_CONTEXT"
    elif isinstance(message, ProvideContextMessage):
        expected_type = "PROVIDE_CONTEXT"
    
    if expected_type and message.message_type != expected_type:
        raise MessageValidationError(
            f"Message type mismatch: expected {expected_type}, got {message.message_type}"
        )
    
    return True


def safe_validate_message(message_data: Dict[str, Any]) -> Optional[AgentMessage]:
    """
    Safely validate a message, returning None on error instead of raising.
    
    Args:
        message_data: Dictionary containing message data
        
    Returns:
        Validated AgentMessage instance or None if validation fails
    """
    try:
        return validate_message(message_data)
    except MessageValidationError:
        return None

