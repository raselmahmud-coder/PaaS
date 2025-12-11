"""Tests for reconstruction module."""

import pytest
from datetime import datetime, timedelta

from src.reconstruction.detector import FailureDetector
from src.reconstruction.reconstructor import AgentReconstructor
from src.persistence.event_store import event_store
from src.persistence.models import init_db
from src.persistence.checkpointer import get_checkpointer
from src.workflows.product_workflow import create_product_upload_workflow
from src.config import settings


def test_failure_detector_healthy_agent(temp_databases):
    """Test failure detector with healthy agent."""
    init_db()
    detector = FailureDetector(timeout_seconds=5)

    # Create recent event
    event_store.log_event(
        agent_id="test-agent-1",
        thread_id="test-thread-1",
        event_type="step_complete",
        step_name="test_step",
    )

    # Agent should be healthy
    assert detector.check_agent_health("test-agent-1") is True


def test_failure_detector_timeout(temp_databases):
    """Test failure detector with timed-out agent."""
    init_db()
    detector = FailureDetector(timeout_seconds=1)

    # Create old event
    from src.persistence.models import AgentEvent, get_session

    session = get_session()
    try:
        old_event = AgentEvent(
            agent_id="test-agent-2",
            thread_id="test-thread-2",
            event_type="step_start",
            timestamp=datetime.utcnow() - timedelta(seconds=10),
        )
        session.add(old_event)
        session.commit()
    finally:
        session.close()

    # Agent should be failed (timeout)
    assert detector.check_agent_health("test-agent-2") is False


def test_reconstruction_basic(temp_databases, llm_stub):
    """Test basic reconstruction functionality using in-memory checkpointer."""
    init_db()

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.base import Checkpoint

    checkpointer = MemorySaver()
    thread_id = "test-thread-recon"
    agent_id = "test-agent-recon"
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "default"}}

    # Create checkpoint
    checkpoint = Checkpoint(
        v=1,
        id="test-checkpoint",
        ts=datetime.utcnow().isoformat(),
        channel_values={
            "current_step": 2,
            "status": "in_progress",
            "product_data": {"name": "Test Product"},
        },
        channel_versions={},
        versions_seen={},
    )

    checkpointer.put(config, checkpoint, {}, {})

    # Create some events
    event_store.log_event(
        agent_id=agent_id,
        thread_id=thread_id,
        event_type="step_start",
        step_name="generate_listing",
    )

    # Test reconstruction
    reconstructor = AgentReconstructor(checkpointer=checkpointer)

    result = reconstructor.reconstruct(agent_id=agent_id, thread_id=thread_id)

    assert result is not None
    assert "checkpoint" in result
    assert "reconstructed_state" in result
    assert "inferred_next_action" in result
    assert "current_step" in result["reconstructed_state"]


def test_reconstruction_integration_failure_and_recovery(temp_databases, llm_stub):
    """Simulate failure, reconstruct, then resume workflow to completion."""
    init_db()

    # Use real sqlite checkpointer (temp db via fixture)
    checkpointer = get_checkpointer()
    workflow = create_product_upload_workflow(checkpointer=checkpointer)

    thread_id = "test-thread-fail"
    agent_id = "product-agent-1"
    initial_state = {
        "task_id": "task-test",
        "agent_id": agent_id,
        "thread_id": thread_id,
        "current_step": 0,
        "status": "pending",
        "messages": [],
        "product_data": {
            "name": "Wireless Bluetooth Headphones",
            "description": "High-quality wireless headphones with noise cancellation",
            "price": 79.99,
            "category": "Electronics",
            "sku": "WBH-001",
        },
        "generated_listing": None,
        "error": None,
        "metadata": {},
    }

    # Step 1: run with injected failure at generate_listing
    config_fail = {"configurable": {"thread_id": thread_id, "fail_step": "generate_listing"}}
    with pytest.raises(RuntimeError):
        workflow.invoke(initial_state, config_fail)

    # Step 2: reconstruct state
    reconstructor = AgentReconstructor(checkpointer=checkpointer)
    recon_result = reconstructor.reconstruct(agent_id=agent_id, thread_id=thread_id)

    assert recon_result["checkpoint"] is not None
    assert recon_result["reconstructed_state"]["status"] in {"in_progress", "failed", "pending"}

    # Step 3: resume without failure flag
    config_resume = {"configurable": {"thread_id": thread_id}}
    result = workflow.invoke(initial_state, config_resume)

    assert result["status"] == "completed"
    assert result["current_step"] == 3
    assert "generated_listing" in result

    # Validate event log captured error and recovery path
    events = event_store.get_events(thread_id=thread_id)
    event_types = [evt.event_type for evt in events]
    assert "error" in event_types  # failure was logged
    assert any(evt.event_type == "step_complete" and evt.step_name == "confirm_upload" for evt in events)


def test_reconstruction_late_step_failure(temp_databases, llm_stub):
    """Test reconstruction when failure occurs at a later step (confirm_upload)."""
    init_db()

    # Use real sqlite checkpointer (temp db via fixture)
    checkpointer = get_checkpointer()
    workflow = create_product_upload_workflow(checkpointer=checkpointer)

    thread_id = "test-thread-late-fail"
    agent_id = "product-agent-1"
    initial_state = {
        "task_id": "task-test-late",
        "agent_id": agent_id,
        "thread_id": thread_id,
        "current_step": 0,
        "status": "pending",
        "messages": [],
        "product_data": {
            "name": "Test Product Late",
            "description": "Product for late failure test",
            "price": 49.99,
            "category": "Electronics",
            "sku": "TEST-LATE-001",
        },
        "generated_listing": None,
        "error": None,
        "metadata": {},
    }

    # Step 1: run with injected failure at confirm_upload (step 3)
    config_fail = {"configurable": {"thread_id": thread_id, "fail_step": "confirm_upload"}}
    with pytest.raises(RuntimeError):
        workflow.invoke(initial_state, config_fail)

    # Step 2: Verify checkpoint exists from earlier steps
    reconstructor = AgentReconstructor(checkpointer=checkpointer)
    recon_result = reconstructor.reconstruct(agent_id=agent_id, thread_id=thread_id)

    assert recon_result["checkpoint"] is not None
    # Should have checkpointed at step 2 (generate_listing completed)
    checkpoint_state = recon_result["reconstructed_state"]
    assert checkpoint_state.get("current_step", 0) >= 1  # At least step 1 completed

    # Step 3: Verify event log contains error and earlier step completions
    events = event_store.get_events(thread_id=thread_id)
    event_types = [evt.event_type for evt in events]
    step_names = [evt.step_name for evt in events if evt.step_name]
    
    # Should have error event
    assert "error" in event_types
    # Should have completed validate_product_data and generate_listing
    assert "validate_product_data" in step_names
    assert "generate_listing" in step_names
    # Should have error for confirm_upload
    error_events = [evt for evt in events if evt.event_type == "error"]
    assert any(evt.step_name == "confirm_upload" for evt in error_events)

    # Step 4: Resume workflow without failure flag
    config_resume = {"configurable": {"thread_id": thread_id}}
    result = workflow.invoke(initial_state, config_resume)

    # Should complete successfully
    assert result["status"] == "completed"
    assert result["current_step"] == 3
    assert "generated_listing" in result

    # Step 5: Verify final event log shows recovery completion
    final_events = event_store.get_events(thread_id=thread_id)
    final_step_completes = [evt for evt in final_events if evt.event_type == "step_complete" and evt.step_name == "confirm_upload"]
    assert len(final_step_completes) > 0  # Should have completed confirm_upload after recovery

