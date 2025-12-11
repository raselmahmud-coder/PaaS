# Agents Module

## Overview

This module contains the core agent implementations for the PaaS system. Agents are autonomous entities that perform specific tasks within workflows, such as product validation, listing generation, and marketing campaign creation.

## What's in This Directory

- **`base.py`** - Base agent state definitions and logging decorator
- **`product_upload.py`** - Product Upload agent with validation, listing generation, and upload confirmation
- **`marketing.py`** - Marketing agent for generating marketing copy and scheduling campaigns

## Why This Code Exists

Agents form the building blocks of our multi-agent system. They encapsulate domain-specific logic and provide:

- **Separation of Concerns**: Each agent handles a specific domain (product management, marketing)
- **Reusability**: Agents can be composed into different workflows
- **Observability**: Built-in logging for all agent actions
- **Testability**: Agents can be tested independently with failure injection

## When It's Used

Agents are invoked by workflows (see [`src/workflows/`](../workflows/README.md)) during workflow execution:

- **Product Upload Agent**: Used in product upload workflows to validate, generate listings, and confirm uploads
- **Marketing Agent**: Used in vendor workflows to create marketing content and schedule campaigns
- **Base Classes**: Used by all agents to maintain consistent state structure and logging

## Key Components

### AgentState TypedDict

The base state structure that all agents use:

```python
class AgentState(TypedDict):
    # Task identification
    task_id: str
    agent_id: str
    thread_id: str
    
    # Workflow tracking
    current_step: int
    status: Literal["pending", "in_progress", "completed", "failed"]
    
    # Message history
    messages: List[BaseMessage]
    
    # Product data (for Product Upload agent)
    product_data: Optional[Dict[str, Any]]
    
    # Generated content
    generated_listing: Optional[str]
    
    # Error information
    error: Optional[str]
    
    # Metadata
    metadata: Dict[str, Any]
```

### Logging Decorator (`@with_logging`)

Automatically logs agent step execution to the event store:

```python
@with_logging
def validate_product_data(state: AgentState, config=None) -> AgentState:
    # Agent logic here
    # Automatically logs: step_start, step_complete, errors
    return updated_state
```

**Features:**
- Logs step start and completion events
- Captures input/output data and state snapshots
- Handles errors and logs them appropriately
- Integrates with [`src/persistence/event_store.py`](../persistence/event_store.py)

### Agent Function Pattern

All agent functions follow this pattern:

```python
@with_logging
def agent_step(state: AgentState, config=None) -> AgentState:
    """
    Agent step function.
    
    Args:
        state: Current agent state
        config: Optional LangGraph configuration (may contain fail_step for testing)
    
    Returns:
        Updated agent state
    """
    # 1. Extract data from state
    # 2. Perform agent logic (e.g., call LLM, validate data)
    # 3. Update state with results
    # 4. Return updated state
    return {
        **state,
        "status": "in_progress",
        "current_step": state["current_step"] + 1,
        # ... other updates
    }
```

### Failure Injection (Testing)

For testing reconstruction scenarios, agents support controlled failure injection:

```python
# In workflow invocation
config = {
    "configurable": {
        "thread_id": "test-thread",
        "fail_step": "generate_listing"  # Inject failure at this step
    }
}

# Agent function checks for failure injection
def _maybe_fail(config: Any, step: str):
    fail_step = (config or {}).get("configurable", {}).get("fail_step")
    if fail_step and str(fail_step) == step:
        raise RuntimeError(f"Injected failure at step: {step}")
```

## Usage Examples

### Product Upload Agent

```python
from src.agents.product_upload import validate_product_data, generate_listing, confirm_upload
from src.agents.base import AgentState

state: AgentState = {
    "task_id": "task-123",
    "agent_id": "product-agent-1",
    "thread_id": "thread-123",
    "current_step": 0,
    "status": "pending",
    "messages": [],
    "product_data": {
        "name": "Wireless Headphones",
        "price": 79.99,
        "category": "Electronics"
    },
    "generated_listing": None,
    "error": None,
    "metadata": {},
}

# Validate product data
state = validate_product_data(state)

# Generate listing
state = generate_listing(state)

# Confirm upload
state = confirm_upload(state)
```

### Marketing Agent

```python
from src.agents.marketing import generate_marketing_copy, schedule_campaign

# Generate marketing copy from product listing
state = generate_marketing_copy(state)

# Schedule campaign
state = schedule_campaign(state)
```

## Architecture/Design Decisions

### TypedDict for Type Safety

Using `TypedDict` instead of regular dictionaries provides:
- Type checking support in IDEs
- Clear documentation of state structure
- Runtime type validation

### Decorator Pattern for Logging

The `@with_logging` decorator:
- Reduces boilerplate code
- Ensures consistent logging across all agents
- Separates logging concerns from business logic

### Config Parameter Pattern

All agent functions accept an optional `config` parameter:
- Required by LangGraph for checkpointing
- Enables failure injection for testing
- Allows future configuration extensions

### State Immutability

Agents return new state dictionaries rather than mutating input:
- Enables state history tracking
- Supports checkpointing and recovery
- Makes testing easier

## Related Modules

- **[`src/workflows/`](../workflows/README.md)** - Workflows orchestrate agent execution
- **[`src/persistence/event_store.py`](../persistence/event_store.py)** - Event logging system
- **[`src/llm/`](../llm/README.md)** - LLM provider used by agents for AI tasks
- **[`src/reconstruction/`](../reconstruction/README.md)** - Uses agent state for failure recovery

