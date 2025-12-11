"""Protocol message definitions for inter-agent communication."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Dict, Any, Optional
import uuid


class AgentMessage(BaseModel):
    """Standard message format for agent-to-agent communication."""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: Literal["TASK_ASSIGN", "TASK_COMPLETE", "REQUEST_CONTEXT", "PROVIDE_CONTEXT"]
    sender: str
    receiver: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class TaskAssignMessage(AgentMessage):
    """Message for assigning a task to an agent."""
    
    message_type: Literal["TASK_ASSIGN"] = "TASK_ASSIGN"
    payload: Dict[str, Any] = Field(
        ...,
        description="Task payload containing task details, parameters, and context"
    )


class TaskCompleteMessage(AgentMessage):
    """Message for reporting task completion."""
    
    message_type: Literal["TASK_COMPLETE"] = "TASK_COMPLETE"
    payload: Dict[str, Any] = Field(
        ...,
        description="Completion payload containing results, status, and output data"
    )


class RequestContextMessage(AgentMessage):
    """Message for requesting context from another agent."""
    
    message_type: Literal["REQUEST_CONTEXT"] = "REQUEST_CONTEXT"
    payload: Dict[str, Any] = Field(
        ...,
        description="Request payload containing target agent ID and time window"
    )


class ProvideContextMessage(AgentMessage):
    """Message for providing context to another agent."""
    
    message_type: Literal["PROVIDE_CONTEXT"] = "PROVIDE_CONTEXT"
    payload: Dict[str, Any] = Field(
        ...,
        description="Context payload containing shared state, interactions, and relevant data"
    )

