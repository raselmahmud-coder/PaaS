"""Tests for persistence layer."""

import pytest
from datetime import datetime
from src.persistence.models import Checkpoint, AgentEvent, init_db, get_session
from src.persistence.checkpointer import get_checkpointer
from src.agents.base import AgentState


def test_checkpoint_save_load(temp_databases):
    """Test saving and loading checkpoints."""
    init_db()
    checkpointer = get_checkpointer()
    
    # Create test state
    thread_id = "test-thread-1"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Create a mock checkpoint
    from langgraph.checkpoint.base import Checkpoint
    checkpoint = Checkpoint(
        v=1,
        id="test-checkpoint-1",
        ts=datetime.utcnow().isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
    )
    
    metadata = {}
    new_versions = {}
    
    # Save checkpoint
    checkpointer.put(config, checkpoint, metadata, new_versions)
    
    # Load checkpoint
    loaded = checkpointer.get_tuple(config)
    
    assert loaded is not None
    assert loaded.checkpoint.id == checkpoint.id


def test_checkpoint_list(temp_databases):
    """Test listing checkpoints."""
    init_db()
    checkpointer = get_checkpointer()
    
    thread_id = "test-thread-2"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Create multiple checkpoints
    from langgraph.checkpoint.base import Checkpoint
    for i in range(3):
        checkpoint = Checkpoint(
            v=1,
            id=f"test-checkpoint-{i}",
            ts=datetime.utcnow().isoformat(),
            channel_values={},
            channel_versions={},
            versions_seen={},
        )
        checkpointer.put(config, checkpoint, {}, {})
    
    # List checkpoints
    checkpoints = list(checkpointer.list(config, limit=5))
    
    assert len(checkpoints) == 3


def test_resume_from_checkpoint(temp_databases):
    """Test resuming workflow from checkpoint."""
    init_db()
    checkpointer = get_checkpointer()
    
    thread_id = "test-thread-resume"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Create initial checkpoint
    from langgraph.checkpoint.base import Checkpoint
    checkpoint = Checkpoint(
        v=1,
        id="resume-checkpoint",
        ts=datetime.utcnow().isoformat(),
        channel_values={
            "current_step": 2,
            "status": "in_progress",
        },
        channel_versions={},
        versions_seen={},
    )
    
    checkpointer.put(config, checkpoint, {}, {})
    
    # Simulate restart - load checkpoint
    loaded = checkpointer.get_tuple(config)
    
    assert loaded is not None
    assert loaded.checkpoint.channel_values["current_step"] == 2
    assert loaded.checkpoint.channel_values["status"] == "in_progress"

