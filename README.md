# Protocol-Aware Agentic Swarm (PaaS) for E-Commerce Vendor Management

A resilient multi-agent system for autonomous e-commerce workflows with state persistence, event logging, and failure recovery capabilities.

## Phase 1: Minimal Viable Agent System

This phase implements:
- Basic agent framework with LangGraph
- SQLite state persistence and checkpointing
- Event logging for all agent actions
- Multi-agent workflow (Product Upload → Marketing)
- Basic state reconstruction after failures

## Setup

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd PaaS
```

2. Install dependencies:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

4. Activate the virtual environment:
```bash
poetry shell
```

5. Initialize the database:
```bash
python -m src.persistence.models init_db
```

## Project Structure

```
PaaS/
├── src/
│   ├── agents/          # Agent implementations
│   ├── workflows/       # LangGraph workflow definitions
│   ├── persistence/     # State persistence and event logging
│   ├── reconstruction/  # Failure detection and state reconstruction
│   └── protocol/       # Message schemas
├── tests/              # Test suite
├── data/               # SQLite database files
└── notebooks/          # Jupyter notebooks for exploration
```

## Usage

### Run a simple agent workflow:
```bash
python -m src.main
```

### Run tests:
```bash
pytest
```

### Run recovery tests:
```bash
pytest -k reconstruction
```

This will run all reconstruction and recovery tests, including:
- Failure detection tests
- Basic reconstruction functionality
- Integration tests for failure recovery at different steps
- Event log validation after recovery

### Using the Recovery Helper

The `recover_and_resume_workflow()` helper function provides a complete recovery flow:

```python
from src.workflows.product_workflow import create_product_upload_workflow
from src.reconstruction.reconstructor import recover_and_resume_workflow

workflow = create_product_upload_workflow()
initial_state = {
    "task_id": "task-123",
    "agent_id": "product-agent-1",
    "thread_id": "thread-123",
    # ... other state fields
}

# Automatically detects failure, reconstructs, and resumes
result = recover_and_resume_workflow(
    workflow=workflow,
    agent_id="product-agent-1",
    thread_id="thread-123",
    initial_state=initial_state
)

if result["recovered"]:
    print("Workflow was recovered and completed!")
    print(f"Final status: {result['final_result']['status']}")
```

### Failure Injection for Testing

To test failure scenarios, you can inject failures at specific steps using the `fail_step` config:

```python
config = {"configurable": {"thread_id": "thread-123", "fail_step": "generate_listing"}}
# This will raise RuntimeError at the generate_listing step
workflow.invoke(initial_state, config)
```

### Run recovery tests (failure + reconstruction):
```bash
pytest -k reconstruction
```

This will run all reconstruction and recovery tests, including:
- Failure detection tests
- Basic reconstruction functionality
- Integration tests for failure recovery at different steps
- Event log validation after recovery

### Using the Recovery Helper

The `recover_and_resume_workflow()` helper function provides a complete recovery flow that automatically detects failures, reconstructs state, and resumes workflows:

```python
from src.workflows.product_workflow import create_product_upload_workflow
from src.reconstruction.reconstructor import recover_and_resume_workflow

workflow = create_product_upload_workflow()
initial_state = {
    "task_id": "task-123",
    "agent_id": "product-agent-1",
    "thread_id": "thread-123",
    "current_step": 0,
    "status": "pending",
    "messages": [],
    "product_data": {...},
    # ... other required state fields
}

# Automatically detects failure, reconstructs, and resumes
result = recover_and_resume_workflow(
    workflow=workflow,
    agent_id="product-agent-1",
    thread_id="thread-123",
    initial_state=initial_state
)

if result["recovered"]:
    print("Workflow was recovered and completed!")
    print(f"Final status: {result['final_result']['status']}")
```

### Failure Injection for Testing

To test failure scenarios, you can inject failures at specific steps using the `fail_step` config:

```python
config = {"configurable": {"thread_id": "thread-123", "fail_step": "generate_listing"}}
# This will raise RuntimeError at the generate_listing step
workflow.invoke(initial_state, config)
```

Available failure injection points:
- `"validate_product_data"` - Fails at step 1
- `"generate_listing"` - Fails at step 2
- `"confirm_upload"` - Fails at step 3

### Protocol Message Integration

The system now supports structured protocol messages for inter-agent communication. The vendor workflow uses protocol messages for handoff between Product Upload and Marketing agents.

**Protocol Message Types:**
- `TASK_ASSIGN` - Assign a task to an agent
- `TASK_COMPLETE` - Report task completion
- `REQUEST_CONTEXT` - Request context from another agent
- `PROVIDE_CONTEXT` - Provide context to another agent

**Using Protocol Messages:**

```python
from src.protocol.handoff import create_task_assign_message, extract_state_from_message
from src.protocol.validator import validate_message

# Create a protocol message
message = create_task_assign_message(
    sender="product-agent-1",
    receiver="marketing-agent-1",
    state=agent_state,
    task_description="Generate marketing copy"
)

# Validate message
validate_message(message.model_dump())

# Extract state from message
state = extract_state_from_message(message)
```

**Protocol Handoff in Workflows:**

The vendor workflow automatically uses protocol messages for handoff:
- Product agent completes → creates `TASK_COMPLETE` message
- Marketing agent receives → creates `TASK_ASSIGN` message
- State extracted from message for processing
- Protocol messages logged in event store

**Run Protocol Tests:**
```bash
pytest tests/test_protocol.py -v
```

## Development Status

- [x] Phase 0: Environment Setup
- [x] Week 1: Foundation Infrastructure
- [x] Week 2: Event Logging
- [x] Week 3: Multi-Agent Workflow
- [x] Week 4: Basic State Reconstruction
- [x] Phase 2 Week 5: Protocol Message Integration

## License

MIT

