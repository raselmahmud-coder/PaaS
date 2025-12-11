"""Tests for protocol message integration."""

import pytest
from datetime import datetime
from src.protocol.messages import (
    TaskAssignMessage,
    TaskCompleteMessage,
    RequestContextMessage,
    ProvideContextMessage,
)
from src.protocol.validator import (
    validate_message,
    validate_message_structure,
    MessageValidationError,
    safe_validate_message,
)
from src.protocol.handoff import (
    create_task_assign_message,
    create_task_complete_message,
    extract_state_from_message,
    message_to_state_dict,
    state_dict_to_message_data,
)
from src.agents.base import AgentState


def test_message_validation_valid():
    """Test validation of valid messages."""
    message_data = {
        "message_type": "TASK_ASSIGN",
        "sender": "agent-1",
        "receiver": "agent-2",
        "payload": {"task": "test"},
    }
    
    message = validate_message(message_data)
    assert isinstance(message, TaskAssignMessage)
    assert message.sender == "agent-1"
    assert message.receiver == "agent-2"


def test_message_validation_invalid_type():
    """Test validation rejects invalid message types."""
    message_data = {
        "message_type": "INVALID_TYPE",
        "sender": "agent-1",
        "receiver": "agent-2",
        "payload": {},
    }
    
    with pytest.raises(MessageValidationError):
        validate_message(message_data)


def test_message_validation_missing_fields():
    """Test validation rejects messages with missing required fields."""
    message_data = {
        "message_type": "TASK_ASSIGN",
        # Missing sender and receiver
    }
    
    with pytest.raises(MessageValidationError):
        validate_message(message_data)


def test_message_structure_validation():
    """Test structure validation of message objects."""
    message = TaskAssignMessage(
        sender="agent-1",
        receiver="agent-2",
        payload={"task": "test"},
    )
    
    assert validate_message_structure(message) is True


def test_safe_validate_message():
    """Test safe validation returns None on error."""
    invalid_data = {"invalid": "data"}
    
    result = safe_validate_message(invalid_data)
    assert result is None


def test_create_task_assign_message(temp_databases):
    """Test creating TASK_ASSIGN message from state."""
    from src.persistence.models import init_db
    
    init_db()
    
    state: AgentState = {
        "task_id": "task-123",
        "agent_id": "product-agent-1",
        "thread_id": "thread-123",
        "current_step": 3,
        "status": "completed",
        "messages": [],
        "product_data": {"name": "Test Product"},
        "generated_listing": "Test listing",
        "error": None,
        "metadata": {},
    }
    
    message = create_task_assign_message(
        sender="product-agent-1",
        receiver="marketing-agent-1",
        state=state,
        task_description="Generate marketing copy",
    )
    
    assert isinstance(message, TaskAssignMessage)
    assert message.sender == "product-agent-1"
    assert message.receiver == "marketing-agent-1"
    assert "state" in message.payload
    assert message.payload["state"]["agent_id"] == "product-agent-1"


def test_create_task_complete_message(temp_databases):
    """Test creating TASK_COMPLETE message from state."""
    from src.persistence.models import init_db
    
    init_db()
    
    state: AgentState = {
        "task_id": "task-123",
        "agent_id": "product-agent-1",
        "thread_id": "thread-123",
        "current_step": 3,
        "status": "completed",
        "messages": [],
        "product_data": {"name": "Test Product"},
        "generated_listing": "Test listing",
        "error": None,
        "metadata": {},
    }
    
    message = create_task_complete_message(
        sender="product-agent-1",
        receiver="marketing-agent-1",
        state=state,
        completion_status="Product uploaded successfully",
    )
    
    assert isinstance(message, TaskCompleteMessage)
    assert message.sender == "product-agent-1"
    assert message.receiver == "marketing-agent-1"
    assert message.payload["completion_status"] == "Product uploaded successfully"


def test_extract_state_from_message():
    """Test extracting state from protocol message."""
    state: AgentState = {
        "task_id": "task-123",
        "agent_id": "product-agent-1",
        "thread_id": "thread-123",
        "current_step": 2,
        "status": "in_progress",
        "messages": [],
        "product_data": {"name": "Test"},
        "generated_listing": None,
        "error": None,
        "metadata": {},
    }
    
    message = create_task_assign_message(
        sender="agent-1",
        receiver="agent-2",
        state=state,
    )
    
    extracted_state = extract_state_from_message(message)
    assert extracted_state["agent_id"] == "product-agent-1"
    assert extracted_state["thread_id"] == "thread-123"
    assert extracted_state["current_step"] == 2


def test_message_to_state_dict():
    """Test converting message to state dictionary with metadata."""
    state: AgentState = {
        "task_id": "task-123",
        "agent_id": "product-agent-1",
        "thread_id": "thread-123",
        "current_step": 2,
        "status": "in_progress",
        "messages": [],
        "product_data": {"name": "Test"},
        "generated_listing": None,
        "error": None,
        "metadata": {},
    }
    
    message = create_task_assign_message(
        sender="agent-1",
        receiver="agent-2",
        state=state,
    )
    
    state_dict = message_to_state_dict(message)
    assert state_dict["agent_id"] == "product-agent-1"
    assert state_dict["_protocol_message_id"] == message.message_id
    assert state_dict["_protocol_message_type"] == "TASK_ASSIGN"
    assert state_dict["_protocol_sender"] == "agent-1"


def test_protocol_handoff_in_workflow(temp_databases, llm_stub):
    """Test protocol handoff in vendor workflow end-to-end."""
    from src.persistence.models import init_db
    from src.workflows.vendor_workflow import create_vendor_workflow
    from src.persistence.checkpointer import get_checkpointer
    from src.persistence.event_store import event_store
    
    init_db()
    
    checkpointer = get_checkpointer()
    workflow = create_vendor_workflow(checkpointer=checkpointer)
    
    thread_id = "test-thread-protocol"
    initial_state: AgentState = {
        "task_id": "task-protocol-test",
        "agent_id": "product-agent-1",
        "thread_id": thread_id,
        "current_step": 0,
        "status": "pending",
        "messages": [],
        "product_data": {
            "name": "Protocol Test Product",
            "description": "Testing protocol integration",
            "price": 99.99,
            "category": "Electronics",
            "sku": "PROTO-001",
        },
        "generated_listing": None,
        "error": None,
        "metadata": {},
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run workflow
    result = workflow.invoke(initial_state, config)
    
    # Verify workflow completed
    assert result["status"] == "completed"
    assert result["current_step"] == 5
    
    # Verify protocol handoff occurred
    events = event_store.get_events(thread_id=thread_id)
    protocol_handoff_events = [
        evt for evt in events if evt.event_type == "protocol_handoff"
    ]
    assert len(protocol_handoff_events) > 0
    
    # Verify protocol message metadata in events
    protocol_receive_events = [
        evt for evt in events if evt.event_type == "protocol_receive"
    ]
    assert len(protocol_receive_events) > 0
    
    # Check that protocol message info is in event data
    handoff_event = protocol_handoff_events[0]
    if handoff_event.input_data:
        assert "_protocol_message" in handoff_event.input_data or "task_complete_message_id" in handoff_event.input_data


def test_backward_compatibility_direct_state(temp_databases, llm_stub):
    """Test that workflows still work with direct state (backward compatibility)."""
    from src.persistence.models import init_db
    from src.workflows.product_workflow import create_product_upload_workflow
    from src.persistence.checkpointer import get_checkpointer
    
    init_db()
    
    checkpointer = get_checkpointer()
    workflow = create_product_upload_workflow(checkpointer=checkpointer)
    
    thread_id = "test-thread-direct"
    initial_state: AgentState = {
        "task_id": "task-direct",
        "agent_id": "product-agent-1",
        "thread_id": thread_id,
        "current_step": 0,
        "status": "pending",
        "messages": [],
        "product_data": {
            "name": "Direct State Test",
            "description": "Testing backward compatibility",
            "price": 49.99,
            "category": "Electronics",
            "sku": "DIRECT-001",
        },
        "generated_listing": None,
        "error": None,
        "metadata": {},
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Product workflow should still work without protocol messages
    result = workflow.invoke(initial_state, config)
    
    assert result["status"] == "completed"
    assert result["current_step"] == 3

