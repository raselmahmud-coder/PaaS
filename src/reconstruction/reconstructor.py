"""Agent state reconstruction module."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage

from src.llm import get_llm
from src.persistence.checkpointer import get_checkpointer
from src.persistence.event_store import event_store
from src.reconstruction.detector import FailureDetector

logger = logging.getLogger(__name__)


class AgentReconstructor:
    """Reconstructs agent state after failure.
    
    Supports both synchronous and asynchronous reconstruction with optional
    peer context retrieval via Kafka for enhanced state recovery.
    """

    def __init__(
        self,
        checkpointer=None,
        llm_model: Optional[str] = None,
        enable_peer_context: bool = True,
        peer_context_timeout: float = 5.0,
    ):
        """Initialize reconstructor.
        
        Args:
            checkpointer: LangGraph checkpointer instance.
            llm_model: LLM model to use for reconstruction inference.
            enable_peer_context: Whether to query peer agents for context.
            peer_context_timeout: Timeout for peer context collection (seconds).
        """
        # Use LangGraph's built-in checkpointer
        self.checkpointer = checkpointer or get_checkpointer()
        # Use lower temperature for more deterministic reconstruction
        self.llm = get_llm(model=llm_model, temperature=0.3)
        
        # Peer context settings
        self.enable_peer_context = enable_peer_context
        self.peer_context_timeout = peer_context_timeout
        self._reconstructor_id = f"reconstructor-{id(self)}"

    def reconstruct(self, agent_id: str, thread_id: str) -> Dict[str, Any]:
        """Reconstruct agent state from checkpoint and events (synchronous).
        
        For peer context retrieval, use `reconstruct_async()` instead.
        """

        # Step 1: Load last checkpoint
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "default"}}
        # LangGraph's checkpointer uses get_tuple method
        checkpoint_tuple = self.checkpointer.get_tuple(config)

        if checkpoint_tuple is None:
            # Fallback: try without checkpoint_ns
            alt_config = {"configurable": {"thread_id": thread_id}}
            checkpoint_tuple = self.checkpointer.get_tuple(alt_config)

        if checkpoint_tuple is None and hasattr(self.checkpointer, "list"):
            try:
                saved = list(self.checkpointer.list(config))
                if not saved:
                    saved = list(
                        self.checkpointer.list(
                            {"configurable": {"thread_id": thread_id}}
                        )
                    )
                if saved:
                    checkpoint_tuple = saved[-1]
            except Exception:
                checkpoint_tuple = None

        if checkpoint_tuple is None:
            raise ValueError(f"No checkpoint found for thread {thread_id}")

        checkpoint_data = checkpoint_tuple.checkpoint
        if isinstance(checkpoint_data, dict):
            ts_value = checkpoint_data.get("ts")
            channel_values = checkpoint_data.get("channel_values", {})
        else:
            ts_value = checkpoint_data.ts
            channel_values = checkpoint_data.channel_values

        checkpoint_timestamp = (
            datetime.fromisoformat(ts_value) if ts_value else datetime.utcnow()
        )

        # Step 2: Get events since checkpoint
        events = event_store.get_events(
            agent_id=agent_id, thread_id=thread_id, since=checkpoint_timestamp
        )

        checkpoint_struct = {"ts": ts_value, "channel_values": channel_values}

        # Step 3: Build reconstruction prompt
        prompt = self._build_reconstruction_prompt(checkpoint_struct, events)

        # Step 4: LLM inference for next action
        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            inferred_data = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            inferred_data = {
                "current_step": channel_values.get("current_step", 0),
                "next_action": "continue",
                "missing_data": [],
                "status": channel_values.get("status", "in_progress"),
            }

        # Step 5: Merge state
        reconstructed_state = self._merge_state(
            checkpoint_struct, events, inferred_data
        )

        return {
            "checkpoint": checkpoint_struct,
            "events_since": [e.to_dict() for e in events],
            "inferred_next_action": inferred_data,
            "reconstructed_state": reconstructed_state,
            "reconstruction_timestamp": datetime.utcnow().isoformat(),
        }

    async def reconstruct_async(
        self,
        agent_id: str,
        thread_id: str,
        use_peer_context: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Reconstruct agent state with optional peer context retrieval (async).
        
        This method supports distributed peer context retrieval via Kafka to gather
        additional context from agents that interacted with the failed agent.
        
        Args:
            agent_id: ID of the failed agent.
            thread_id: Thread ID of the failed workflow.
            use_peer_context: Whether to query peers. Uses instance default if None.
            
        Returns:
            Reconstruction result including peer context if available.
        """
        use_peers = use_peer_context if use_peer_context is not None else self.enable_peer_context
        
        # Step 1: Load checkpoint (same as sync version)
        checkpoint_struct, channel_values, checkpoint_timestamp = self._load_checkpoint(thread_id)
        
        # Step 2: Get events since checkpoint
        events = event_store.get_events(
            agent_id=agent_id, thread_id=thread_id, since=checkpoint_timestamp
        )
        
        # Step 3: Query peer agents for context (if enabled)
        peer_context = []
        if use_peers:
            try:
                peer_context = await self.query_peer_agents(
                    failed_agent_id=agent_id,
                    thread_id=thread_id,
                    last_known_step=channel_values.get("current_step"),
                    last_known_status=channel_values.get("status"),
                )
                logger.info(f"Collected context from {len(peer_context)} peer agents")
            except Exception as e:
                logger.warning(f"Failed to query peer agents: {e}")
                peer_context = []
        
        # Step 4: Build reconstruction prompt with peer context
        prompt = self._build_reconstruction_prompt_with_peer_context(
            checkpoint_struct, events, peer_context
        )
        
        # Step 5: LLM inference for next action
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            inferred_data = json.loads(response.content)
        except json.JSONDecodeError:
            inferred_data = {
                "current_step": channel_values.get("current_step", 0),
                "next_action": "continue",
                "missing_data": [],
                "status": channel_values.get("status", "in_progress"),
            }
        
        # Step 6: Merge state with peer context consideration
        reconstructed_state = self._merge_state_with_peer_context(
            checkpoint_struct, events, inferred_data, peer_context
        )
        
        return {
            "checkpoint": checkpoint_struct,
            "events_since": [e.to_dict() for e in events],
            "peer_context": peer_context,
            "peer_agents_queried": len(peer_context),
            "inferred_next_action": inferred_data,
            "reconstructed_state": reconstructed_state,
            "reconstruction_timestamp": datetime.utcnow().isoformat(),
        }
    
    def _load_checkpoint(self, thread_id: str) -> tuple:
        """Load checkpoint data for a thread.
        
        Returns:
            Tuple of (checkpoint_struct, channel_values, checkpoint_timestamp)
        """
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "default"}}
        checkpoint_tuple = self.checkpointer.get_tuple(config)
        
        if checkpoint_tuple is None:
            alt_config = {"configurable": {"thread_id": thread_id}}
            checkpoint_tuple = self.checkpointer.get_tuple(alt_config)
        
        if checkpoint_tuple is None and hasattr(self.checkpointer, "list"):
            try:
                saved = list(self.checkpointer.list(config))
                if not saved:
                    saved = list(self.checkpointer.list({"configurable": {"thread_id": thread_id}}))
                if saved:
                    checkpoint_tuple = saved[-1]
            except Exception:
                checkpoint_tuple = None
        
        if checkpoint_tuple is None:
            raise ValueError(f"No checkpoint found for thread {thread_id}")
        
        checkpoint_data = checkpoint_tuple.checkpoint
        if isinstance(checkpoint_data, dict):
            ts_value = checkpoint_data.get("ts")
            channel_values = checkpoint_data.get("channel_values", {})
        else:
            ts_value = checkpoint_data.ts
            channel_values = checkpoint_data.channel_values
        
        checkpoint_timestamp = (
            datetime.fromisoformat(ts_value) if ts_value else datetime.utcnow()
        )
        
        checkpoint_struct = {"ts": ts_value, "channel_values": channel_values}
        
        return checkpoint_struct, channel_values, checkpoint_timestamp
    
    async def query_peer_agents(
        self,
        failed_agent_id: str,
        thread_id: str,
        last_known_step: Optional[int] = None,
        last_known_status: Optional[str] = None,
        time_window_seconds: int = 3600,
    ) -> List[Dict[str, Any]]:
        """Query peer agents for context about a failed agent via Kafka.
        
        Broadcasts a REQUEST_CONTEXT message and collects PROVIDE_CONTEXT responses
        from peer agents within the timeout window.
        
        Args:
            failed_agent_id: ID of the failed agent.
            thread_id: Thread ID of the failed workflow.
            last_known_step: Last known step the agent was on.
            last_known_status: Last known status of the agent.
            time_window_seconds: How far back to look for interactions.
            
        Returns:
            List of peer context responses.
        """
        try:
            from src.messaging.kafka_config import get_kafka_config
            from src.messaging.producer import KafkaMessageProducer
            from src.messaging.consumer import KafkaMessageConsumer
        except ImportError:
            logger.warning("Kafka messaging module not available")
            return []
        
        config = get_kafka_config()
        response_topic = config.get_response_topic(self._reconstructor_id)
        
        peer_responses: List[Dict[str, Any]] = []
        
        try:
            async with KafkaMessageProducer(config) as producer:
                # Publish REQUEST_CONTEXT to broadcast topic
                success = await producer.publish_context_request(
                    requester_id=self._reconstructor_id,
                    failed_agent_id=failed_agent_id,
                    thread_id=thread_id,
                    time_window_seconds=time_window_seconds,
                )
                
                if not success:
                    logger.warning("Failed to publish context request")
                    return []
                
                logger.info(
                    f"Published context request for failed agent {failed_agent_id}, "
                    f"waiting for responses on {response_topic}"
                )
                
                # Create consumer for response topic
                async with KafkaMessageConsumer(
                    topics=[response_topic],
                    group_id=f"{config.consumer_group_prefix}-reconstructor-{self._reconstructor_id}",
                    config=config,
                ) as consumer:
                    # Collect responses within timeout
                    peer_responses = await consumer.collect_messages(
                        timeout=self.peer_context_timeout,
                        message_type="PROVIDE_CONTEXT",
                    )
        
        except Exception as e:
            logger.error(f"Error querying peer agents: {e}")
            return []
        
        logger.info(f"Collected {len(peer_responses)} peer context responses")
        return peer_responses
    
    def _build_reconstruction_prompt_with_peer_context(
        self,
        checkpoint: Dict[str, Any],
        events: list,
        peer_context: List[Dict[str, Any]],
    ) -> str:
        """Build reconstruction prompt including peer context.
        
        Args:
            checkpoint: Checkpoint data structure.
            events: List of events since checkpoint.
            peer_context: List of peer context responses.
            
        Returns:
            Prompt string for LLM reconstruction.
        """
        channel_values = checkpoint.get("channel_values", {})
        ts_value = checkpoint.get("ts")
        
        checkpoint_data = {
            "timestamp": ts_value,
            "channel_values": self._to_jsonable(channel_values),
            "current_step": channel_values.get("current_step", 0),
            "status": channel_values.get("status", "unknown"),
        }
        
        events_data = [e.to_dict() for e in events]
        
        # Build peer context section
        peer_context_section = ""
        if peer_context:
            peer_summaries = []
            for ctx in peer_context:
                responder = ctx.get("responder_id", "unknown")
                interactions = ctx.get("interactions", [])
                memory = ctx.get("memory_state", {})
                
                summary = {
                    "peer_agent": responder,
                    "interaction_count": len(interactions),
                    "last_interaction_type": memory.get("last_interaction_type"),
                    "last_step_name": memory.get("last_step_name"),
                    "has_protocol_messages": bool(memory.get("protocol_messages")),
                }
                peer_summaries.append(summary)
            
            # Build detailed peer interactions (limit to 5 most relevant per peer)
            detailed_interactions = [{
                "peer": ctx.get("responder_id"),
                "interactions": ctx.get("interactions", [])[:5],
                "memory_state": ctx.get("memory_state", {})
            } for ctx in peer_context]
            
            peer_context_section = f"""
        
        Peer Agent Context ({len(peer_context)} agents responded):
        {json.dumps(peer_summaries, indent=2)}
        
        Detailed peer interactions:
        {json.dumps(detailed_interactions, indent=2, default=str)}"""
        
        prompt = f"""An agent failed and needs to be reconstructed.

        Last checkpoint state:
        {json.dumps(checkpoint_data, indent=2)}

        Events since checkpoint ({len(events)} events):
        {json.dumps(events_data, indent=2, default=str)}{peer_context_section}

        Based on this information (including peer agent context if available), analyze:
        1. What step was the agent on when it failed?
        2. What should be the next action?
        3. Is any data missing that needs to be re-computed?
        4. What is the current status of the workflow?
        5. What insights do peer agents provide about the failed agent's state?

        Respond with valid JSON only:
        {{
            "current_step": <integer>,
            "next_action": "<action_description>",
            "missing_data": ["<list of missing data fields>"],
            "status": "<pending|in_progress|completed|failed>",
            "recommendation": "<what should happen next>",
            "peer_insights": "<key insights from peer agents if available>"
        }}"""
        
        return prompt
    
    def _merge_state_with_peer_context(
        self,
        checkpoint: Dict[str, Any],
        events: list,
        inferred_data: Dict[str, Any],
        peer_context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge checkpoint, events, inferred data, and peer context.
        
        Args:
            checkpoint: Checkpoint data.
            events: Events since checkpoint.
            inferred_data: LLM-inferred reconstruction data.
            peer_context: Peer context responses.
            
        Returns:
            Reconstructed state dictionary.
        """
        # Start with basic merge
        reconstructed = self._merge_state(checkpoint, events, inferred_data)
        
        # Add peer context metadata
        reconstructed["peer_context_used"] = len(peer_context) > 0
        reconstructed["peer_agents_count"] = len(peer_context)
        
        if peer_context:
            # Extract useful data from peer context
            peer_insights = []
            for ctx in peer_context:
                memory_state = ctx.get("memory_state", {})
                if memory_state.get("has_interactions"):
                    peer_insights.append({
                        "agent": ctx.get("responder_id"),
                        "last_interaction": memory_state.get("last_interaction_type"),
                        "interaction_count": memory_state.get("interaction_count", 0),
                    })
            
            reconstructed["peer_insights"] = peer_insights
            
            # If peer context suggests a different status, note the conflict
            for ctx in peer_context:
                last_known = ctx.get("memory_state", {}).get("last_known_state", {})
                if last_known.get("status") and last_known["status"] != reconstructed.get("status"):
                    reconstructed["status_conflict_detected"] = True
                    reconstructed["peer_suggested_status"] = last_known["status"]
                    break
        
        return reconstructed

    def _to_jsonable(self, obj):
        """Make objects JSON-serializable for prompt construction."""
        if isinstance(obj, BaseMessage):
            return {
                "type": obj.__class__.__name__,
                "content": obj.content,
                "additional_kwargs": getattr(obj, "additional_kwargs", {}),
            }
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    def _build_reconstruction_prompt(self, checkpoint, events) -> str:
        """Build prompt for LLM reconstruction."""

        channel_values = checkpoint.get("channel_values", {})
        ts_value = checkpoint.get("ts")

        checkpoint_data = {
            "timestamp": ts_value,
            "channel_values": self._to_jsonable(channel_values),
            "current_step": channel_values.get("current_step", 0),
            "status": channel_values.get("status", "unknown"),
        }

        events_data = [e.to_dict() for e in events]

        prompt = f"""An agent failed and needs to be reconstructed.

        Last checkpoint state:
        {json.dumps(checkpoint_data, indent=2)}

        Events since checkpoint ({len(events)} events):
        {json.dumps(events_data, indent=2, default=str)}

        Based on this information, analyze:
        1. What step was the agent on when it failed?
        2. What should be the next action?
        3. Is any data missing that needs to be re-computed?
        4. What is the current status of the workflow?

        Respond with valid JSON only:
        {{
            "current_step": <integer>,
            "next_action": "<action_description>",
            "missing_data": ["<list of missing data fields>"],
            "status": "<pending|in_progress|completed|failed>",
            "recommendation": "<what should happen next>"
        }}"""

        return prompt

    def _merge_state(
        self, checkpoint, events, inferred_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge checkpoint, events, and inferred data into reconstructed state."""

        # Start with checkpoint state
        channel_values = checkpoint.get("channel_values", {})
        reconstructed = dict(channel_values)

        # Update with inferred data
        if "current_step" in inferred_data:
            reconstructed["current_step"] = inferred_data["current_step"]

        if "status" in inferred_data:
            reconstructed["status"] = inferred_data["status"]

        # Add reconstruction metadata
        reconstructed["reconstructed"] = True
        reconstructed["reconstruction_timestamp"] = datetime.utcnow().isoformat()
        reconstructed["events_since_checkpoint"] = len(events)

        return reconstructed


def recover_and_resume_workflow(
    workflow,
    agent_id: str,
    thread_id: str,
    initial_state: Dict[str, Any],
    timeout_seconds: int = 30,
    checkpointer=None,
) -> Dict[str, Any]:
    """
    Helper function to detect failure, reconstruct state, and resume workflow.
    
    This function provides a complete recovery flow:
    1. Detects if the agent/thread has failed (timeout-based)
    2. Reconstructs the agent state from checkpoint and events
    3. Resumes the workflow from the reconstructed state
    
    For peer context retrieval, use `recover_and_resume_workflow_async()` instead.
    
    Args:
        workflow: The LangGraph workflow to resume
        agent_id: ID of the agent that failed
        thread_id: Thread ID of the workflow
        initial_state: Initial state dict (used as base for resume)
        timeout_seconds: Timeout for failure detection (default: 30)
        checkpointer: Optional checkpointer instance (uses default if None)
    
    Returns:
        Dict containing:
            - "recovered": bool indicating if recovery was needed
            - "reconstruction_result": reconstruction data (if recovered)
            - "final_result": final workflow result
            - "events": list of all events for the thread
    
    Example:
        >>> from src.workflows.product_workflow import create_product_upload_workflow
        >>> workflow = create_product_upload_workflow()
        >>> result = recover_and_resume_workflow(
        ...     workflow=workflow,
        ...     agent_id="product-agent-1",
        ...     thread_id="thread-123",
        ...     initial_state={...}
        ... )
        >>> if result["recovered"]:
        ...     print("Workflow was recovered and completed!")
    """
    from src.reconstruction.detector import FailureDetector
    
    detector = FailureDetector(timeout_seconds=timeout_seconds)
    reconstructor = AgentReconstructor(checkpointer=checkpointer, enable_peer_context=False)
    
    # Check if thread has failed
    has_failed = not detector.check_thread_health(thread_id)
    
    if not has_failed:
        # No failure detected, workflow may still be running or completed
        # Try to get current state from checkpoint
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "default"}}
        try:
            checkpoint_tuple = reconstructor.checkpointer.get_tuple(config)
            if checkpoint_tuple:
                # Workflow has checkpoint, try to resume from it
                checkpoint_state = checkpoint_tuple.checkpoint
                if isinstance(checkpoint_state, dict):
                    channel_values = checkpoint_state.get("channel_values", {})
                else:
                    channel_values = checkpoint_state.channel_values
                
                # If workflow is already completed, return current state
                if channel_values.get("status") == "completed":
                    return {
                        "recovered": False,
                        "reconstruction_result": None,
                        "final_result": channel_values,
                        "events": event_store.get_events(thread_id=thread_id),
                    }
        except Exception:
            pass
        
        # No checkpoint or workflow not completed, run normally
        config = {"configurable": {"thread_id": thread_id}}
        final_result = workflow.invoke(initial_state, config)
        return {
            "recovered": False,
            "reconstruction_result": None,
            "final_result": final_result,
            "events": event_store.get_events(thread_id=thread_id),
        }
    
    # Failure detected - reconstruct and resume
    try:
        reconstruction_result = reconstructor.reconstruct(agent_id=agent_id, thread_id=thread_id)
        
        # Merge reconstructed state with initial state
        reconstructed_state = reconstruction_result["reconstructed_state"]
        resume_state = {
            **initial_state,
            **reconstructed_state,
            "status": "in_progress",  # Reset to in_progress for resume
        }
        
        # Resume workflow
        config = {"configurable": {"thread_id": thread_id}}
        final_result = workflow.invoke(resume_state, config)
        
        return {
            "recovered": True,
            "reconstruction_result": reconstruction_result,
            "final_result": final_result,
            "events": event_store.get_events(thread_id=thread_id),
        }
    except Exception as e:
        # Reconstruction or resume failed
        return {
            "recovered": False,
            "reconstruction_result": None,
            "final_result": None,
            "error": str(e),
            "events": event_store.get_events(thread_id=thread_id),
        }


async def recover_and_resume_workflow_async(
    workflow,
    agent_id: str,
    thread_id: str,
    initial_state: Dict[str, Any],
    timeout_seconds: int = 30,
    checkpointer=None,
    use_peer_context: bool = True,
    peer_context_timeout: float = 5.0,
) -> Dict[str, Any]:
    """
    Async helper function to detect failure, reconstruct state with peer context, and resume.
    
    This function provides a complete recovery flow with distributed peer context retrieval:
    1. Detects if the agent/thread has failed (timeout-based)
    2. Queries peer agents via Kafka for context about failed agent
    3. Reconstructs the agent state using checkpoint, events, and peer context
    4. Resumes the workflow from the reconstructed state
    
    Args:
        workflow: The LangGraph workflow to resume
        agent_id: ID of the agent that failed
        thread_id: Thread ID of the workflow
        initial_state: Initial state dict (used as base for resume)
        timeout_seconds: Timeout for failure detection (default: 30)
        checkpointer: Optional checkpointer instance (uses default if None)
        use_peer_context: Whether to query peer agents for context (default: True)
        peer_context_timeout: Timeout for peer context collection (default: 5.0)
    
    Returns:
        Dict containing:
            - "recovered": bool indicating if recovery was needed
            - "reconstruction_result": reconstruction data (if recovered)
            - "peer_context_used": bool indicating if peer context was used
            - "final_result": final workflow result
            - "events": list of all events for the thread
    
    Example:
        >>> import asyncio
        >>> from src.workflows.product_workflow import create_product_upload_workflow
        >>> workflow = create_product_upload_workflow()
        >>> result = await recover_and_resume_workflow_async(
        ...     workflow=workflow,
        ...     agent_id="product-agent-1",
        ...     thread_id="thread-123",
        ...     initial_state={...},
        ...     use_peer_context=True,
        ... )
        >>> if result["recovered"]:
        ...     print(f"Workflow recovered with {result['peer_context_used']} peer context!")
    """
    from src.reconstruction.detector import FailureDetector
    
    detector = FailureDetector(timeout_seconds=timeout_seconds)
    reconstructor = AgentReconstructor(
        checkpointer=checkpointer,
        enable_peer_context=use_peer_context,
        peer_context_timeout=peer_context_timeout,
    )
    
    # Check if thread has failed
    has_failed = not detector.check_thread_health(thread_id)
    
    if not has_failed:
        # No failure detected, workflow may still be running or completed
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "default"}}
        try:
            checkpoint_tuple = reconstructor.checkpointer.get_tuple(config)
            if checkpoint_tuple:
                checkpoint_state = checkpoint_tuple.checkpoint
                if isinstance(checkpoint_state, dict):
                    channel_values = checkpoint_state.get("channel_values", {})
                else:
                    channel_values = checkpoint_state.channel_values
                
                if channel_values.get("status") == "completed":
                    return {
                        "recovered": False,
                        "reconstruction_result": None,
                        "peer_context_used": False,
                        "final_result": channel_values,
                        "events": event_store.get_events(thread_id=thread_id),
                    }
        except Exception:
            pass
        
        # Run workflow normally
        config = {"configurable": {"thread_id": thread_id}}
        final_result = workflow.invoke(initial_state, config)
        return {
            "recovered": False,
            "reconstruction_result": None,
            "peer_context_used": False,
            "final_result": final_result,
            "events": event_store.get_events(thread_id=thread_id),
        }
    
    # Failure detected - reconstruct with peer context and resume
    try:
        reconstruction_result = await reconstructor.reconstruct_async(
            agent_id=agent_id,
            thread_id=thread_id,
            use_peer_context=use_peer_context,
        )
        
        # Merge reconstructed state with initial state
        reconstructed_state = reconstruction_result["reconstructed_state"]
        resume_state = {
            **initial_state,
            **reconstructed_state,
            "status": "in_progress",
        }
        
        # Resume workflow
        config = {"configurable": {"thread_id": thread_id}}
        final_result = workflow.invoke(resume_state, config)
        
        return {
            "recovered": True,
            "reconstruction_result": reconstruction_result,
            "peer_context_used": reconstruction_result.get("peer_agents_queried", 0) > 0,
            "final_result": final_result,
            "events": event_store.get_events(thread_id=thread_id),
        }
    except Exception as e:
        logger.error(f"Failed to recover workflow: {e}")
        return {
            "recovered": False,
            "reconstruction_result": None,
            "peer_context_used": False,
            "final_result": None,
            "error": str(e),
            "events": event_store.get_events(thread_id=thread_id),
        }
