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

## Development Status

- [x] Phase 0: Environment Setup
- [ ] Week 1: Foundation Infrastructure
- [ ] Week 2: Event Logging
- [ ] Week 3: Multi-Agent Workflow
- [ ] Week 4: Basic State Reconstruction

## License

MIT

