"""Event store for logging agent actions."""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from src.persistence.models import AgentEvent, get_session, init_db
from src.protocol.messages import AgentMessage


def _serialize_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, BaseMessage):
        # Convert LangChain messages to dict
        return {
            "type": obj.__class__.__name__,
            "content": obj.content,
            "additional_kwargs": getattr(obj, "additional_kwargs", {}),
        }
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # Try to serialize custom objects
        try:
            return str(obj)
        except:
            return {"_type": obj.__class__.__name__, "_repr": repr(obj)}
    else:
        # Check if it's already JSON serializable
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


class EventStore:
    """Event store for logging and querying agent events."""
    
    def __init__(self):
        """Initialize the event store."""
        init_db()
    
    def log_event(
        self,
        agent_id: str,
        thread_id: str,
        event_type: str,
        step_name: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        state_snapshot: Optional[Dict[str, Any]] = None,
        protocol_message: Optional[AgentMessage] = None,
    ) -> AgentEvent:
        """
        Log an agent event.
        
        Args:
            agent_id: ID of the agent
            thread_id: Thread/workflow ID
            event_type: Type of event (step_start, step_complete, error, protocol_handoff, etc.)
            step_name: Name of the step (optional)
            input_data: Input data for the event (optional)
            output_data: Output data for the event (optional)
            state_snapshot: State snapshot (optional)
            protocol_message: Protocol message associated with this event (optional)
        """
        session = get_session()
        try:
            # Serialize data to ensure JSON compatibility
            serialized_input = _serialize_for_json(input_data) if input_data else None
            serialized_output = _serialize_for_json(output_data) if output_data else None
            serialized_snapshot = _serialize_for_json(state_snapshot) if state_snapshot else None
            
            # Add protocol message metadata if provided
            if protocol_message:
                if serialized_input is None:
                    serialized_input = {}
                if not isinstance(serialized_input, dict):
                    serialized_input = {"data": serialized_input}
                serialized_input["_protocol_message"] = {
                    "message_id": protocol_message.message_id,
                    "message_type": protocol_message.message_type,
                    "sender": protocol_message.sender,
                    "receiver": protocol_message.receiver,
                    "timestamp": protocol_message.timestamp.isoformat(),
                }
            
            event = AgentEvent(
                agent_id=agent_id,
                thread_id=thread_id,
                event_type=event_type,
                step_name=step_name,
                input_data=serialized_input,
                output_data=serialized_output,
                state_snapshot=serialized_snapshot,
                timestamp=datetime.utcnow(),
            )
            
            session.add(event)
            session.commit()
            return event
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_events(
        self,
        agent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        since: Optional[datetime] = None,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AgentEvent]:
        """Query events with filters."""
        session = get_session()
        try:
            query = session.query(AgentEvent)
            
            if agent_id:
                query = query.filter(AgentEvent.agent_id == agent_id)
            
            if thread_id:
                query = query.filter(AgentEvent.thread_id == thread_id)
            
            if since:
                query = query.filter(AgentEvent.timestamp >= since)
            
            if event_type:
                query = query.filter(AgentEvent.event_type == event_type)
            
            query = query.order_by(AgentEvent.timestamp.asc())
            
            if limit:
                query = query.limit(limit)
            
            return query.all()
        finally:
            session.close()
    
    def get_events_by_thread(self, thread_id: str) -> List[AgentEvent]:
        """Get all events for a specific thread."""
        return self.get_events(thread_id=thread_id)
    
    def get_latest_event(
        self,
        agent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Optional[AgentEvent]:
        """Get the most recent event."""
        session = get_session()
        try:
            query = session.query(AgentEvent)
            
            if agent_id:
                query = query.filter(AgentEvent.agent_id == agent_id)
            
            if thread_id:
                query = query.filter(AgentEvent.thread_id == thread_id)
            
            return query.order_by(AgentEvent.timestamp.desc()).first()
        finally:
            session.close()
    
    def replay_events(self, thread_id: str) -> List[AgentEvent]:
        """Get ordered events for replay."""
        return self.get_events(thread_id=thread_id)


# Global event store instance
event_store = EventStore()

