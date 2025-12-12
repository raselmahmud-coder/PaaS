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
│   │   ├── hybrid.py           # Hybrid reconstruction (automata + LLM)
│   │   └── automata_reconstructor.py  # Automata-based recovery
│   ├── protocol/        # Message schemas and validation
│   ├── messaging/       # Kafka messaging for peer context retrieval
│   ├── chaos/           # Chaos engineering framework (Phase 3)
│   │   ├── decorators.py       # Fault injection decorators
│   │   ├── scenarios.py        # Chaos scenarios
│   │   ├── runner.py           # Scenario runner
│   │   ├── metrics.py          # MTTR-A metrics collection
│   │   └── export.py           # JSON/CSV export
│   ├── semantic/        # Semantic protocol layer (Phase 4)
│   │   ├── embedder.py         # Sentence-Transformers wrapper
│   │   ├── similarity.py       # Term alignment checking
│   │   ├── handshake.py        # 5-step handshake protocol
│   │   └── negotiator.py       # LLM-based term negotiation
│   └── automata/        # L* automata learning (Phase 4)
│       ├── learner.py          # L* algorithm wrapper
│       ├── predictor.py        # Behavior prediction
│       └── event_generator.py  # Synthetic event generation
├── tests/              # Test suite (210+ tests)
├── data/               # SQLite database files
│   └── experiments/    # Experiment results (900+ runs)
├── notebooks/          # Jupyter notebooks for analysis
│   └── thesis_evaluation.ipynb  # Thesis figures & statistics
└── scenarios/          # YAML scenario templates (4 e-commerce scenarios)
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

### Async Recovery with Peer Context

For enhanced reconstruction using peer agent context via Kafka:

```python
import asyncio
from src.reconstruction.reconstructor import recover_and_resume_workflow_async

result = await recover_and_resume_workflow_async(
    workflow=workflow,
    agent_id="product-agent-1",
    thread_id="thread-123",
    initial_state=initial_state,
    use_peer_context=True,  # Query peer agents via Kafka
    peer_context_timeout=5.0,  # Wait up to 5 seconds for responses
)

if result["recovered"]:
    print(f"Workflow recovered with peer context: {result['peer_context_used']}")
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

## Experiment Framework (Phase 5)

Run thesis experiments with 3 conditions:
- **Baseline**: No resilience (agent failure = workflow failure)
- **Reconstruction**: LLM-based reconstruction only
- **Full System**: Semantic + Automata + LLM hybrid

### Run Experiments
```bash
# Run all experiments (900+ runs)
python -m src.experiments.runner --runs 300 --all-conditions --seed 42 --output data/experiments

# Run single condition
python -m src.experiments.runner --runs 100 --condition full_system
```

### Experiment Results (Key Findings)
| Condition | Success Rate | Recovery Rate | MTTR Mean |
|-----------|--------------|---------------|-----------|
| Baseline | 35.7% | N/A | N/A |
| Reconstruction | 83.7% | 74.7% | 0.102s |
| Full System | 94.7% | 91.6% | 0.139s |

- **+59 percentage points** improvement (Baseline → Full System)
- **+16.9 percentage points** recovery rate (Reconstruction → Full System)
- All comparisons statistically significant (p < 0.001)

### Generate Thesis Figures
```bash
jupyter notebook notebooks/thesis_evaluation.ipynb
```

## Development Status

### Phase 1: Minimal Viable Agent System (Complete)
- [x] Week 1: Foundation Infrastructure
- [x] Week 2: Event Logging
- [x] Week 3: Multi-Agent Workflow
- [x] Week 4: Basic State Reconstruction

### Phase 2: Protocol Layer & Enhanced Reconstruction (Complete)
- [x] Week 5: Protocol Message Integration
- [x] Week 6: Message Broker Integration (Kafka)
- [x] Week 7: Peer Context Retrieval
- [x] Week 8: Checkpoint Optimization (skipped - not needed)

### Phase 3: Failure Injection & Chaos Engineering (Complete)
- [x] Week 9: Fault Injection Decorators (`src/chaos/decorators.py`)
- [x] Week 10: Chaos Scenarios (`src/chaos/scenarios.py`, `runner.py`)
- [x] Week 11: Kubernetes + LitmusChaos (skipped - Python-only approach)
- [x] Week 12: MTTR-A Metrics Collection (`src/chaos/metrics.py`, `export.py`)

### Phase 4: Semantic Protocol & Advanced Reconstruction (Complete)
- [x] Week 13: Semantic Embeddings (`src/semantic/embedder.py`, `similarity.py`)
- [x] Week 14: Semantic Handshake Protocol (`src/semantic/handshake.py`, `negotiator.py`)
- [x] Week 15: L* Automata Learning (`src/automata/learner.py`, `predictor.py`)
- [x] Week 16: Hybrid Reconstruction (`src/reconstruction/hybrid.py`)

### Phase 5: Evaluation & Benchmarking (Complete)
- [x] Week 17: Synthetic Scenarios (`scenarios/*.yaml`)
- [x] Week 18: Experiment Runner (`src/experiments/`)
- [x] Week 19: 900+ Experiments (300 per condition)
- [x] Week 20: Analysis Notebook (`notebooks/thesis_evaluation.ipynb`)

### Phase 6: Thesis Writing & Finalization (Pending)
- [ ] Week 21: System Documentation
- [ ] Week 22: Thesis Chapters (Draft)
- [ ] Week 23: Results Chapter & Discussion
- [ ] Week 24: Revision & Defense Prep

## License

MIT

