"""Failure detection for agents."""

from datetime import datetime, timedelta
from typing import List, Optional
from src.persistence.event_store import event_store
from src.config import settings


class FailureDetector:
    """Detects agent failures based on timeout and inactivity."""
    
    def __init__(self, timeout_seconds: Optional[int] = None):
        """Initialize failure detector."""
        self.timeout_seconds = timeout_seconds or settings.agent_timeout_seconds
    
    def check_agent_health(self, agent_id: str) -> bool:
        """Check if agent is still active."""
        last_event = event_store.get_latest_event(agent_id=agent_id)
        
        if last_event is None:
            # No events means agent never started or completely failed
            return False
        
        # Check if last event is within timeout window
        age = datetime.utcnow() - last_event.timestamp
        return age.total_seconds() < self.timeout_seconds
    
    def check_thread_health(self, thread_id: str) -> bool:
        """Check if thread/workflow is still active."""
        last_event = event_store.get_latest_event(thread_id=thread_id)
        
        if last_event is None:
            return False
        
        age = datetime.utcnow() - last_event.timestamp
        return age.total_seconds() < self.timeout_seconds
    
    def get_failed_agents(self) -> List[str]:
        """Return list of agent IDs that appear to have failed."""
        failed_agents = []
        
        # Get all unique agent IDs from events
        from src.persistence.models import get_session, AgentEvent
        session = get_session()
        try:
            agent_ids = session.query(AgentEvent.agent_id).distinct().all()
            
            for (agent_id,) in agent_ids:
                if not self.check_agent_health(agent_id):
                    # Check if agent has incomplete workflow
                    last_event = event_store.get_latest_event(agent_id=agent_id)
                    if last_event and last_event.event_type != "step_complete":
                        failed_agents.append(agent_id)
        finally:
            session.close()
        
        return failed_agents
    
    def get_failed_threads(self) -> List[str]:
        """Return list of thread IDs that appear to have failed."""
        failed_threads = []
        
        from src.persistence.models import get_session, AgentEvent
        session = get_session()
        try:
            thread_ids = session.query(AgentEvent.thread_id).distinct().all()
            
            for (thread_id,) in thread_ids:
                if not self.check_thread_health(thread_id):
                    # Check if thread has incomplete workflow
                    last_event = event_store.get_latest_event(thread_id=thread_id)
                    if last_event and last_event.event_type != "step_complete":
                        failed_threads.append(thread_id)
        finally:
            session.close()
        
        return failed_threads

