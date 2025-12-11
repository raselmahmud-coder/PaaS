"""Agent state reconstruction module."""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from langchain_core.messages import BaseMessage, HumanMessage

from src.llm import get_llm
from src.persistence.checkpointer import get_checkpointer
from src.persistence.event_store import event_store
from src.reconstruction.detector import FailureDetector


class AgentReconstructor:
    """Reconstructs agent state after failure."""

    def __init__(
        self,
        checkpointer=None,
        llm_model: Optional[str] = None,
    ):
        """Initialize reconstructor."""
        # Use LangGraph's built-in checkpointer
        self.checkpointer = checkpointer or get_checkpointer()
        # Use lower temperature for more deterministic reconstruction
        self.llm = get_llm(model=llm_model, temperature=0.3)

    def reconstruct(self, agent_id: str, thread_id: str) -> Dict[str, Any]:
        """Reconstruct agent state from checkpoint and events."""

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
    reconstructor = AgentReconstructor(checkpointer=checkpointer)
    
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
