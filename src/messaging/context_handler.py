"""Handler for context request messages - queries database for agent interactions."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.persistence.event_store import event_store
from src.persistence.models import AgentEvent

logger = logging.getLogger(__name__)


class ContextRequestHandler:
    """Handles REQUEST_CONTEXT messages by querying local event history."""
    
    def __init__(self, agent_id: str):
        """Initialize the context request handler.
        
        Args:
            agent_id: The ID of the agent this handler belongs to.
        """
        self.agent_id = agent_id
    
    def query_interactions_with_agent(
        self,
        target_agent_id: str,
        thread_id: Optional[str] = None,
        time_window_seconds: int = 3600,
    ) -> List[Dict[str, Any]]:
        """Query event store for interactions with a specific agent.
        
        Args:
            target_agent_id: The agent ID to find interactions with.
            thread_id: Optional thread ID to filter by.
            time_window_seconds: How far back to look (default 1 hour).
            
        Returns:
            List of interaction events as dictionaries.
        """
        since = datetime.utcnow() - timedelta(seconds=time_window_seconds)
        
        # Get all events for this agent within the time window
        events = event_store.get_events(
            agent_id=self.agent_id,
            thread_id=thread_id,
            since=since,
        )
        
        interactions = []
        for event in events:
            # Check if this event involves the target agent
            if self._event_involves_agent(event, target_agent_id):
                interactions.append(event.to_dict())
        
        logger.info(
            f"Found {len(interactions)} interactions between {self.agent_id} "
            f"and {target_agent_id}"
        )
        return interactions
    
    def _event_involves_agent(self, event: AgentEvent, target_agent_id: str) -> bool:
        """Check if an event involves a specific target agent.
        
        Args:
            event: The event to check.
            target_agent_id: The agent ID to look for.
            
        Returns:
            True if the event involves the target agent.
        """
        # Check protocol messages in input/output data
        if event.input_data:
            if self._data_contains_agent_ref(event.input_data, target_agent_id):
                return True
        
        if event.output_data:
            if self._data_contains_agent_ref(event.output_data, target_agent_id):
                return True
        
        # Check state snapshot for agent references
        if event.state_snapshot:
            if self._data_contains_agent_ref(event.state_snapshot, target_agent_id):
                return True
        
        # Check event type for handoff events
        if event.event_type in ("protocol_handoff", "protocol_receive"):
            return True
        
        return False
    
    def _data_contains_agent_ref(self, data: Any, target_agent_id: str) -> bool:
        """Recursively check if data contains a reference to the target agent.
        
        Args:
            data: The data to search.
            target_agent_id: The agent ID to look for.
            
        Returns:
            True if the target agent is referenced.
        """
        if isinstance(data, str):
            return target_agent_id in data
        
        if isinstance(data, dict):
            for key, value in data.items():
                # Check protocol message fields
                if key in ("sender", "receiver", "agent_id", "target_agent"):
                    if value == target_agent_id:
                        return True
                
                # Recursive check
                if self._data_contains_agent_ref(value, target_agent_id):
                    return True
        
        if isinstance(data, list):
            for item in data:
                if self._data_contains_agent_ref(item, target_agent_id):
                    return True
        
        return False
    
    def get_memory_state_for_agent(
        self,
        target_agent_id: str,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get relevant memory/state data for interactions with an agent.
        
        This extracts useful context from the agent's perspective about
        its interactions with the target agent.
        
        Args:
            target_agent_id: The agent to get context for.
            thread_id: Optional thread ID to filter by.
            
        Returns:
            Dictionary containing relevant memory state.
        """
        # Get the most recent interaction events
        interactions = self.query_interactions_with_agent(
            target_agent_id=target_agent_id,
            thread_id=thread_id,
            time_window_seconds=3600,
        )
        
        if not interactions:
            return {
                "has_interactions": False,
                "last_interaction": None,
                "interaction_count": 0,
            }
        
        # Extract relevant state from interactions
        last_interaction = interactions[-1]
        
        memory_state = {
            "has_interactions": True,
            "interaction_count": len(interactions),
            "last_interaction_time": last_interaction.get("timestamp"),
            "last_interaction_type": last_interaction.get("event_type"),
            "last_step_name": last_interaction.get("step_name"),
        }
        
        # Extract last known state from snapshot
        last_snapshot = last_interaction.get("state_snapshot")
        if last_snapshot:
            memory_state["last_known_state"] = {
                "status": last_snapshot.get("status"),
                "current_step": last_snapshot.get("current_step"),
                "task_id": last_snapshot.get("task_id"),
            }
        
        # Check for protocol messages
        protocol_info = self._extract_protocol_info(interactions)
        if protocol_info:
            memory_state["protocol_messages"] = protocol_info
        
        return memory_state
    
    def _extract_protocol_info(
        self,
        interactions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract protocol message information from interactions.
        
        Args:
            interactions: List of interaction events.
            
        Returns:
            List of protocol message summaries.
        """
        protocol_messages = []
        
        for event in interactions:
            input_data = event.get("input_data") or {}
            
            # Check for protocol message metadata
            if "_protocol_message" in input_data:
                msg_info = input_data["_protocol_message"]
                protocol_messages.append({
                    "message_id": msg_info.get("message_id"),
                    "message_type": msg_info.get("message_type"),
                    "sender": msg_info.get("sender"),
                    "receiver": msg_info.get("receiver"),
                    "timestamp": msg_info.get("timestamp"),
                })
        
        return protocol_messages
    
    async def handle_context_request(
        self,
        message: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle a REQUEST_CONTEXT message and prepare response data.
        
        Args:
            message: The REQUEST_CONTEXT message.
            
        Returns:
            Dictionary containing the context response data.
        """
        failed_agent_id = message.get("failed_agent_id")
        thread_id = message.get("thread_id")
        time_window = message.get("time_window_seconds", 3600)
        
        if not failed_agent_id:
            logger.warning("REQUEST_CONTEXT message missing failed_agent_id")
            return {
                "error": "Missing failed_agent_id",
                "interactions": [],
                "memory_state": {},
            }
        
        # Query for interactions
        interactions = self.query_interactions_with_agent(
            target_agent_id=failed_agent_id,
            thread_id=thread_id,
            time_window_seconds=time_window,
        )
        
        # Get memory state
        memory_state = self.get_memory_state_for_agent(
            target_agent_id=failed_agent_id,
            thread_id=thread_id,
        )
        
        return {
            "interactions": interactions,
            "memory_state": memory_state,
            "responder_id": self.agent_id,
            "failed_agent_id": failed_agent_id,
            "thread_id": thread_id,
        }

