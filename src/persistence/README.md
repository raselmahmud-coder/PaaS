# Persistence Module

## Overview

This module provides state persistence, checkpointing, and event logging for the PaaS system. It enables workflow recovery, audit trails, and state reconstruction after failures.

## What's in This Directory

- **`models.py`** - SQLAlchemy database models (Checkpoint, AgentEvent) and database initialization
- **`checkpointer.py`** - LangGraph checkpointer factory for workflow state persistence
- **`event_store.py`** - Event logging system for agent actions and protocol messages

## Why This Code Exists

The persistence layer solves several critical problems:

- **State Recovery**: Enables workflows to resume after failures or restarts
- **Audit Trail**: Provides complete history of agent actions for debugging and compliance
- **Reconstruction**: Supplies data needed by the reconstruction module to rebuild agent state
- **Checkpointing**: Allows workflows to save progress at each step for recovery

## When It's Used

### Database Models (`models.py`)
- **Initialization**: Called at system startup to create database tables
- **Event Logging**: Used by `EventStore` to persist agent events
- **Custom Checkpoints**: Legacy checkpoint model (now primarily using LangGraph's checkpointer)

### Checkpointer (`checkpointer.py`)
- **Workflow Creation**: Used when creating workflows to enable state persistence
- **State Recovery**: Used by reconstruction module to load previous workflow state
- **Testing**: Uses `MemorySaver` for in-memory testing without persistence

### Event Store (`event_store.py`)
- **Agent Logging**: Automatically called by `@with_logging` decorator on every agent step
- **Protocol Messages**: Logs protocol handoff events with message metadata
- **Reconstruction**: Queried by reconstruction module to build state recovery prompts
- **Debugging**: Used to trace workflow execution and diagnose issues

## Key Components

### Database Models

#### Checkpoint Model

Legacy checkpoint model (now primarily using LangGraph's built-in checkpointer):

```python
class Checkpoint(Base):
    __tablename__ = "checkpoints"
    
    id = Column(Integer, primary_key=True)
    thread_id = Column(String(100), nullable=False, index=True)
    checkpoint_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

#### AgentEvent Model

Event log for all agent actions:

```python
class AgentEvent(Base):
    __tablename__ = "agent_events"
    
    event_id = Column(Integer, primary_key=True)
    agent_id = Column(String(100), nullable=False, index=True)
    thread_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    event_type = Column(String(50), nullable=False)  # "step_start", "step_complete", "error", etc.
    step_name = Column(String(100), nullable=True)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    state_snapshot = Column(JSON, nullable=True)
```

**Event Types:**
- `step_start` - Agent step execution started
- `step_complete` - Agent step completed successfully
- `error` - Agent step failed with error
- `protocol_handoff` - Protocol message handoff between agents
- `protocol_receive` - Protocol message received by agent

### Checkpointer Usage

The checkpointer factory provides LangGraph-compatible checkpointers:

```python
from src.persistence.checkpointer import get_checkpointer

# Get persistent SQLite checkpointer (default)
checkpointer = get_checkpointer()

# Get in-memory checkpointer (for testing)
checkpointer = get_checkpointer(use_memory=True)
```

**Database Separation:**

The system uses **two separate database files**:

1. **`agent_system.db`** - Custom SQLAlchemy models (AgentEvent, legacy Checkpoint)
2. **`agent_system_checkpoint.db`** - LangGraph's SqliteSaver schema

This separation prevents schema conflicts between custom models and LangGraph's internal checkpoint schema.

### Event Store API

#### Logging Events

```python
from src.persistence.event_store import event_store

event_store.log_event(
    agent_id="product-agent-1",
    thread_id="thread-123",
    event_type="step_complete",
    step_name="validate_product_data",
    input_data={"product_data": {...}},
    output_data={"status": "validated"},
    state_snapshot={"current_step": 1, ...},
    protocol_message=task_assign_message,  # Optional: for protocol events
)
```

#### Querying Events

```python
# Get all events for a thread
events = event_store.get_events(thread_id="thread-123")

# Get events for a specific agent
events = event_store.get_events(agent_id="product-agent-1")

# Get events since a timestamp
from datetime import datetime, timedelta
since = datetime.utcnow() - timedelta(hours=1)
events = event_store.get_events(since=since)

# Get events of a specific type
events = event_store.get_events(event_type="error")

# Get latest event
latest = event_store.get_latest_event(agent_id="product-agent-1")
```

#### Event Serialization

The event store automatically serializes complex objects (LangChain messages, protocol messages) to JSON-compatible formats:

- **LangChain Messages**: Converted to dictionaries with `type`, `content`, `additional_kwargs`
- **Protocol Messages**: Metadata extracted and stored in `input_data._protocol_message`
- **Nested Objects**: Recursively serialized

## Usage Examples

### Database Initialization

```python
from src.persistence.models import init_db

# Initialize database tables
init_db()
```

### Workflow with Checkpointing

```python
from src.persistence.checkpointer import get_checkpointer
from src.workflows.product_workflow import create_product_upload_workflow

# Create checkpointer
checkpointer = get_checkpointer()

# Create workflow with checkpointing enabled
workflow = create_product_upload_workflow(checkpointer=checkpointer)

# Run workflow (state automatically checkpointed at each step)
config = {"configurable": {"thread_id": "thread-123", "checkpoint_ns": ""}}
result = workflow.invoke(initial_state, config)
```

### Event Logging in Agents

```python
from src.persistence.event_store import event_store

# Events are automatically logged by @with_logging decorator
# Manual logging example:
event_store.log_event(
    agent_id="custom-agent",
    thread_id="thread-456",
    event_type="custom_event",
    step_name="custom_step",
    input_data={"custom": "data"},
)
```

### Querying Events for Reconstruction

```python
from src.persistence.event_store import event_store

# Get all events for a failed workflow
events = event_store.get_events(thread_id="failed-thread-123")

# Filter for specific event types
error_events = [e for e in events if e.event_type == "error"]
step_events = [e for e in events if e.event_type == "step_complete"]

# Access event data
for event in events:
    print(f"Event: {event.event_type} at {event.timestamp}")
    print(f"Input: {event.input_data}")
    print(f"Output: {event.output_data}")
    print(f"State: {event.state_snapshot}")
```

## Architecture/Design Decisions

### Database Separation

**Why two databases?**

1. **Schema Compatibility**: LangGraph's `SqliteSaver` has its own schema requirements that differ from our custom models
2. **Framework Independence**: Keeps custom persistence logic separate from LangGraph internals
3. **Migration Path**: Allows gradual migration from custom checkpoints to LangGraph checkpoints

### Event Store Design

**Why JSON columns?**

- **Flexibility**: Can store varying event data structures without schema changes
- **Serialization**: Handles complex objects (messages, state snapshots) automatically
- **Querying**: Still allows filtering by indexed fields (agent_id, thread_id, event_type)

### Checkpointer Factory Pattern

**Why a factory function?**

- **Testing**: Easy to switch to `MemorySaver` for tests
- **Configuration**: Centralized checkpointer creation logic
- **Future-proof**: Easy to add new checkpointer types (PostgreSQL, Redis, etc.)

### Session Management

**Why explicit session handling?**

- **Thread Safety**: Each operation gets its own session
- **Error Handling**: Proper rollback on errors
- **Resource Management**: Sessions are properly closed

## Database Schema

### agent_system.db (Custom Models)

```sql
-- Agent events table
CREATE TABLE agent_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id VARCHAR(100) NOT NULL,
    thread_id VARCHAR(100) NOT NULL,
    timestamp DATETIME NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    step_name VARCHAR(100),
    input_data JSON,
    output_data JSON,
    state_snapshot JSON
);

CREATE INDEX idx_agent_timestamp ON agent_events(agent_id, timestamp);
CREATE INDEX idx_thread_timestamp ON agent_events(thread_id, timestamp);
```

### agent_system_checkpoint.db (LangGraph Schema)

Managed by LangGraph's `SqliteSaver.setup()`. Contains:
- `checkpoints` table - Workflow checkpoint data
- `writes` table - Checkpoint write history

## Related Modules

- **[`src/workflows/`](../workflows/README.md)** - Workflows use checkpointer for state persistence
- **[`src/agents/`](../agents/README.md)** - Agents use event store via `@with_logging` decorator
- **[`src/reconstruction/`](../reconstruction/README.md)** - Uses checkpointer and event store for state recovery
- **[`src/protocol/`](../protocol/README.md)** - Protocol messages logged via event store

