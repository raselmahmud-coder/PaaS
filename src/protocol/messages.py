"""Protocol message definitions for inter-agent communication."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Dict, Any, Optional, List
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


# =============================================================================
# Peer Context Retrieval Messages
# =============================================================================


class RequestContextPayload(BaseModel):
    """Payload schema for REQUEST_CONTEXT messages used in peer context retrieval."""
    
    # Required fields
    failed_agent_id: str = Field(
        ...,
        description="ID of the agent that failed and needs context reconstruction"
    )
    thread_id: str = Field(
        ...,
        description="Thread/workflow ID of the failed agent's task"
    )
    requester_id: str = Field(
        ...,
        description="ID of the reconstruction module or agent requesting context"
    )
    
    # Optional fields for filtering
    time_window_seconds: int = Field(
        default=3600,
        description="How far back to look for interactions (default: 1 hour)"
    )
    response_topic: Optional[str] = Field(
        default=None,
        description="Kafka topic where responses should be published"
    )
    
    # Context about the failure
    failure_timestamp: Optional[datetime] = Field(
        default=None,
        description="When the failure was detected"
    )
    last_known_step: Optional[str] = Field(
        default=None,
        description="Last known step/node the failed agent was executing"
    )
    last_known_status: Optional[str] = Field(
        default=None,
        description="Last known status of the failed agent"
    )


class InteractionEvent(BaseModel):
    """Schema for an interaction event between agents."""
    
    event_id: Optional[int] = None
    event_type: str
    step_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    state_snapshot: Optional[Dict[str, Any]] = None


class MemoryState(BaseModel):
    """Schema for agent memory state shared in context response."""
    
    has_interactions: bool = False
    interaction_count: int = 0
    last_interaction_time: Optional[datetime] = None
    last_interaction_type: Optional[str] = None
    last_step_name: Optional[str] = None
    last_known_state: Optional[Dict[str, Any]] = None
    protocol_messages: Optional[List[Dict[str, Any]]] = None


class ProvideContextPayload(BaseModel):
    """Payload schema for PROVIDE_CONTEXT messages used in peer context retrieval."""
    
    # Required fields
    responder_id: str = Field(
        ...,
        description="ID of the agent providing context"
    )
    failed_agent_id: str = Field(
        ...,
        description="ID of the failed agent this context is about"
    )
    thread_id: str = Field(
        ...,
        description="Thread/workflow ID"
    )
    
    # Context data
    interactions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of interaction events with the failed agent"
    )
    memory_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Relevant memory/state from the responder's perspective"
    )
    
    # Response metadata
    response_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this context was collected"
    )
    query_window_seconds: Optional[int] = Field(
        default=None,
        description="Time window that was actually queried"
    )


class RequestContextMessage(AgentMessage):
    """Message for requesting context from peer agents about a failed agent.
    
    Used during state reconstruction to gather distributed context from agents
    that may have interacted with the failed agent.
    """
    
    message_type: Literal["REQUEST_CONTEXT"] = "REQUEST_CONTEXT"
    payload: Dict[str, Any] = Field(
        ...,
        description="Request payload containing target agent ID and time window"
    )
    
    @classmethod
    def create(
        cls,
        requester_id: str,
        failed_agent_id: str,
        thread_id: str,
        time_window_seconds: int = 3600,
        response_topic: Optional[str] = None,
        last_known_step: Optional[str] = None,
        last_known_status: Optional[str] = None,
    ) -> "RequestContextMessage":
        """Create a REQUEST_CONTEXT message with proper payload.
        
        Args:
            requester_id: ID of the reconstruction module requesting context.
            failed_agent_id: ID of the agent that failed.
            thread_id: Thread ID of the failed workflow.
            time_window_seconds: How far back to look for interactions.
            response_topic: Kafka topic for responses.
            last_known_step: Last known step of the failed agent.
            last_known_status: Last known status of the failed agent.
            
        Returns:
            A properly formatted RequestContextMessage.
        """
        payload = RequestContextPayload(
            failed_agent_id=failed_agent_id,
            thread_id=thread_id,
            requester_id=requester_id,
            time_window_seconds=time_window_seconds,
            response_topic=response_topic,
            failure_timestamp=datetime.utcnow(),
            last_known_step=last_known_step,
            last_known_status=last_known_status,
        )
        
        return cls(
            sender=requester_id,
            receiver="broadcast",  # Broadcast to all agents
            payload=payload.model_dump(),
        )


class ProvideContextMessage(AgentMessage):
    """Message for providing context to a reconstruction module.
    
    Sent by peer agents in response to REQUEST_CONTEXT messages.
    Contains interaction history and memory state relevant to the failed agent.
    """
    
    message_type: Literal["PROVIDE_CONTEXT"] = "PROVIDE_CONTEXT"
    payload: Dict[str, Any] = Field(
        ...,
        description="Context payload containing shared state, interactions, and relevant data"
    )
    
    @classmethod
    def create(
        cls,
        responder_id: str,
        requester_id: str,
        failed_agent_id: str,
        thread_id: str,
        interactions: List[Dict[str, Any]],
        memory_state: Optional[Dict[str, Any]] = None,
        query_window_seconds: Optional[int] = None,
    ) -> "ProvideContextMessage":
        """Create a PROVIDE_CONTEXT message with proper payload.
        
        Args:
            responder_id: ID of the agent providing context.
            requester_id: ID of the reconstruction module to respond to.
            failed_agent_id: ID of the failed agent.
            thread_id: Thread ID of the workflow.
            interactions: List of interaction events.
            memory_state: Memory/state data relevant to the failed agent.
            query_window_seconds: Time window that was queried.
            
        Returns:
            A properly formatted ProvideContextMessage.
        """
        payload = ProvideContextPayload(
            responder_id=responder_id,
            failed_agent_id=failed_agent_id,
            thread_id=thread_id,
            interactions=interactions,
            memory_state=memory_state or {},
            query_window_seconds=query_window_seconds,
        )
        
        return cls(
            sender=responder_id,
            receiver=requester_id,
            payload=payload.model_dump(),
        )
    
    def get_interactions(self) -> List[Dict[str, Any]]:
        """Get the interactions from the payload."""
        return self.payload.get("interactions", [])
    
    def get_memory_state(self) -> Dict[str, Any]:
        """Get the memory state from the payload."""
        return self.payload.get("memory_state", {})

