"""Tests for agent implementations."""

import pytest
from src.agents.base import AgentState
from src.agents.product_upload import (
    validate_product_data,
    generate_listing,
    confirm_upload
)


def test_validate_product_data_success():
    """Test successful product data validation."""
    state: AgentState = {
        "task_id": "test-1",
        "agent_id": "product-agent-1",
        "thread_id": "thread-1",
        "current_step": 0,
        "status": "pending",
        "messages": [],
        "product_data": {
            "name": "Test Product",
            "price": 29.99,
            "category": "Electronics",
            "description": "A test product",
            "sku": "TEST-001"
        },
        "generated_listing": None,
        "error": None,
        "metadata": {},
    }
    
    result = validate_product_data(state)
    
    assert result["status"] == "in_progress"
    assert result["current_step"] == 1
    assert result["error"] is None
    assert len(result["messages"]) > 0


def test_validate_product_data_failure():
    """Test product data validation with missing fields."""
    state: AgentState = {
        "task_id": "test-2",
        "agent_id": "product-agent-2",
        "thread_id": "thread-2",
        "current_step": 0,
        "status": "pending",
        "messages": [],
        "product_data": {
            "name": "Test Product",
            # Missing price and category
        },
        "generated_listing": None,
        "error": None,
        "metadata": {},
    }
    
    result = validate_product_data(state)
    
    assert result["status"] == "failed"
    assert result["error"] is not None
    assert "Missing required fields" in result["error"]


def test_confirm_upload():
    """Test upload confirmation step."""
    state: AgentState = {
        "task_id": "test-3",
        "agent_id": "product-agent-3",
        "thread_id": "thread-3",
        "current_step": 2,
        "status": "in_progress",
        "messages": [],
        "product_data": {
            "name": "Test Product",
            "sku": "TEST-001"
        },
        "generated_listing": "A great product listing...",
        "error": None,
        "metadata": {},
    }
    
    result = confirm_upload(state)
    
    assert result["status"] == "completed"
    assert result["current_step"] == 3
    assert len(result["messages"]) > 0

