# Phase 1 MVP - Quick Start Guide

## Prerequisites

1. Python 3.11+ installed
2. Poetry installed (`pip install poetry`)
3. LLM API key (OpenAI or Moonshot/Kimi)

## Setup

1. **Install dependencies:**
```bash
poetry install
```

2. **Set up environment:**
```bash
# Create .env file with LLM provider configuration:

# Option 1: Use Moonshot/Kimi API (default)
LLM_PROVIDER=moonshot
LLM_API_KEY=your_moonshot_api_key_here
# Or use: KIMI_MOONSHOT_API_KEY=your_moonshot_api_key_here
LLM_BASE_URL=https://api.moonshot.ai/v1
LLM_MODEL=kimi-k2-turbo-preview
LLM_TEMPERATURE=0.7

# Option 2: Use OpenAI API
LLM_PROVIDER=openai
LLM_API_KEY=your_openai_api_key_here
# Or use: OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7
```

3. **Initialize database:**
```bash
poetry shell

& C:/Users/rasel/AppData/Local/pypoetry/Cache/virtualenvs/paas-htZyYLQx-py3.11/Scripts/Activate.ps1

python -m src.persistence.models
```

## Running the System

### Basic Product Upload Workflow

```bash
python -m src.main
```

This will:
- Initialize the database
- Create a Product Upload workflow
- Execute 3 steps (validate → generate listing → confirm)
- Save checkpoints after each step
- Log all events

### Full Vendor Workflow (Product + Marketing)

```python
from src.workflows.vendor_workflow import create_vendor_workflow
from src.agents.base import AgentState
import uuid

workflow = create_vendor_workflow()
thread_id = str(uuid.uuid4())

initial_state: AgentState = {
    "task_id": f"task-{uuid.uuid4()}",
    "agent_id": "product-agent-1",
    "thread_id": thread_id,
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

config = {"configurable": {"thread_id": thread_id}}
result = workflow.invoke(initial_state, config)
print(f"Status: {result['status']}")
```

### Reconstruction Demo

```bash
python -m src.reconstruction.demo
```

This demonstrates:
- Workflow execution
- Failure detection
- State reconstruction using LLM
- Workflow resumption

## Testing

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_agents.py
pytest tests/test_persistence.py
pytest tests/test_reconstruction.py
```

## Project Structure

```
PaaS/
├── src/
│   ├── agents/          # Agent implementations
│   │   ├── base.py      # Base state definitions and logging decorator
│   │   ├── product_upload.py  # Product Upload agent (3 steps)
│   │   └── marketing.py      # Marketing agent (2 steps)
│   ├── workflows/       # LangGraph workflow definitions
│   │   ├── product_workflow.py  # Single agent workflow
│   │   └── vendor_workflow.py   # Multi-agent workflow
│   ├── persistence/     # State persistence
│   │   ├── models.py    # SQLAlchemy models
│   │   ├── checkpointer.py  # LangGraph checkpoint saver
│   │   └── event_store.py   # Event logging
│   ├── reconstruction/  # Failure recovery
│   │   ├── detector.py  # Failure detection
│   │   ├── reconstructor.py  # State reconstruction
│   │   └── demo.py      # Demo script
│   └── protocol/        # Message schemas
│       └── messages.py  # Pydantic message models
├── tests/               # Test suite
└── data/               # SQLite database (created automatically)
```

## Key Features Implemented

### Week 1: Foundation
- ✅ Product Upload agent with 3 steps
- ✅ SQLite checkpointing
- ✅ State persistence and resume

### Week 2: Event Logging
- ✅ Event store with SQLAlchemy
- ✅ Automatic logging decorator
- ✅ Event query interface

### Week 3: Multi-Agent Workflow
- ✅ Marketing agent
- ✅ Vendor workflow (Product → Marketing)
- ✅ Simple JSON message protocol

### Week 4: Reconstruction
- ✅ Failure detection (timeout-based)
- ✅ LLM-based state reconstruction
- ✅ End-to-end demo

## Next Steps (Phase 2+)

- Migrate to PostgreSQL
- Add Kafka message broker
- Implement semantic protocol
- Add L* automata learning
- Integrate chaos engineering

## Troubleshooting

**Database errors:**
- Ensure `data/` directory exists
- Run `python -m src.persistence.models` to initialize

**OpenAI API errors:**
- Check `.env` file has valid `OPENAI_API_KEY`
- Verify API key has credits

**Import errors:**
- Ensure you're in the virtual environment: `poetry shell`
- Reinstall dependencies: `poetry install`

