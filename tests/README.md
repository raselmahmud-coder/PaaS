# Tests Module

## Overview

This module contains the test suite for the PaaS system. Tests ensure code quality, validate expected behavior, and enable regression testing.

## What's in This Directory

- **`conftest.py`** - Pytest configuration and shared fixtures
- **`test_agents.py`** - Tests for agent implementations
- **`test_persistence.py`** - Tests for database models, checkpointer, and event store
- **`test_protocol.py`** - Tests for protocol message integration and validation
- **`test_reconstruction.py`** - Tests for failure detection and state reconstruction
- **`kimi_llm_test.py`** - Manual test for Moonshot/Kimi LLM integration

## Why This Code Exists

The test suite provides:

- **Quality Assurance**: Ensures code works as expected
- **Regression Prevention**: Catches bugs introduced by changes
- **Documentation**: Tests serve as usage examples
- **Confidence**: Enables safe refactoring and feature additions
- **CI/CD Support**: Automated testing for continuous integration

## When It's Used

### During Development

- **Before Commits**: Run tests to ensure changes don't break existing functionality
- **Feature Development**: Write tests alongside new features (TDD approach)
- **Bug Fixes**: Write tests to reproduce bugs, then fix them

### In CI/CD Pipeline

- **Automated Testing**: Tests run automatically on every commit/PR
- **Quality Gates**: Prevents merging code that fails tests
- **Coverage Reports**: Tracks test coverage metrics

### Manual Testing

- **LLM Integration**: `kimi_llm_test.py` for manual LLM provider testing
- **Integration Testing**: End-to-end workflow testing

## Key Components

### Test Fixtures

#### temp_databases

Creates isolated temporary databases for each test:

```python
@pytest.fixture
def temp_databases(tmp_path):
    """Create isolated temp databases for state and checkpoint."""
    state_db = tmp_path / "agent_system.db"
    checkpoint_db = tmp_path / "agent_system_checkpoint.db"
    # ... setup and teardown
    yield {"state_db": state_db, "checkpoint_db": checkpoint_db}
```

**Usage:**

```python
def test_something(temp_databases):
    # Test runs with isolated databases
    # Databases are cleaned up automatically after test
    pass
```

**Benefits:**

- **Isolation**: Each test gets fresh databases
- **No Conflicts**: Tests don't interfere with each other
- **Cleanup**: Automatic cleanup after test completion

#### llm_stub

Provides deterministic LLM stub to avoid network calls:

```python
@pytest.fixture
def llm_stub(monkeypatch):
    """Provide a deterministic LLM stub."""
    class DummyLLM:
        def invoke(self, messages):
            return type("LLMResult", (), {"content": "stubbed response"})
    
    dummy = DummyLLM("stubbed response")
    monkeypatch.setattr("src.agents.product_upload.llm", dummy)
    yield dummy
```

**Usage:**

```python
def test_agent_function(llm_stub):
    # LLM calls return stubbed response
    # No actual API calls made
    result = agent_function(state)
    assert result["status"] == "completed"
```

**Benefits:**

- **Speed**: Tests run faster without network calls
- **Determinism**: Consistent test results
- **Cost**: No API costs during testing
- **Reliability**: Tests don't depend on external services

#### sample_product_data

Provides sample product data for testing:

```python
@pytest.fixture
def sample_product_data():
    return {
        "name": "Test Product",
        "description": "A test product description",
        "price": 29.99,
        "category": "Electronics",
        "sku": "TEST-001"
    }
```

### Test Files

#### test_agents.py

Tests for agent implementations:

- **Agent Function Tests**: Validates agent step functions
- **State Validation**: Tests state structure and updates
- **Error Handling**: Tests error scenarios

**Example:**

```python
def test_validate_product_data_success(temp_databases, llm_stub):
    """Test successful product data validation."""
    state = {
        "product_data": sample_product_data(),
        # ... other fields
    }
    result = validate_product_data(state)
    assert result["status"] == "in_progress"
```

#### test_persistence.py

Tests for persistence layer:

- **Checkpoint Tests**: Tests checkpoint save/load
- **Event Store Tests**: Tests event logging and querying
- **Database Tests**: Tests database initialization

**Example:**

```python
def test_checkpoint_save_load(temp_databases):
    """Test saving and loading checkpoints."""
    checkpointer = get_checkpointer()
    # ... test checkpoint operations
```

#### test_protocol.py

Tests for protocol message integration:

- **Message Validation**: Tests message validation logic
- **Handoff Tests**: Tests protocol handoff utilities
- **Workflow Integration**: Tests protocol messages in workflows

**Example:**

```python
def test_protocol_handoff_in_workflow(temp_databases, llm_stub):
    """Test protocol handoff in vendor workflow."""
    workflow = create_vendor_workflow(checkpointer=checkpointer)
    result = workflow.invoke(initial_state, config)
    # Verify protocol handoff occurred
```

#### test_reconstruction.py

Tests for failure detection and reconstruction:

- **Failure Detection**: Tests failure detector logic
- **State Reconstruction**: Tests state reconstruction process
- **Recovery Integration**: Tests end-to-end recovery workflows

**Example:**

```python
def test_reconstruction_integration_failure_and_recovery(temp_databases, llm_stub):
    """Test failure, reconstruction, and recovery."""
    # Inject failure
    config = {"configurable": {"fail_step": "generate_listing"}}
    with pytest.raises(RuntimeError):
        workflow.invoke(initial_state, config)
    
    # Reconstruct and resume
    reconstructor = AgentReconstructor(checkpointer=checkpointer)
    recon_result = reconstructor.reconstruct(agent_id, thread_id)
    # ... verify reconstruction
```

## Usage Examples

### Running All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src
```

### Running Specific Test Files

```bash
# Run agent tests only
pytest tests/test_agents.py

# Run protocol tests only
pytest tests/test_protocol.py

# Run reconstruction tests only
pytest tests/test_reconstruction.py
```

### Running Specific Tests

```bash
# Run specific test function
pytest tests/test_agents.py::test_validate_product_data_success

# Run tests matching pattern
pytest -k "reconstruction"

# Run tests matching pattern (verbose)
pytest -k "reconstruction" -v
```

### Running Tests with Options

```bash
# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Run in parallel (requires pytest-xdist)
pytest -n auto

# Show local variables on failure
pytest -l
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# View coverage report
# Opens htmlcov/index.html in browser

# Generate terminal coverage report
pytest --cov=src --cov-report=term
```

## Test Organization

### Test Structure

```
tests/
├── conftest.py          # Shared fixtures and configuration
├── test_agents.py       # Agent function tests
├── test_persistence.py  # Database and persistence tests
├── test_protocol.py    # Protocol message tests
├── test_reconstruction.py  # Reconstruction and recovery tests
└── kimi_llm_test.py     # Manual LLM integration test
```

### Test Naming Convention

- **Test files**: `test_*.py`
- **Test functions**: `test_*`
- **Test classes**: `Test*`

### Test Categories

1. **Unit Tests**: Test individual functions/classes in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows

## Writing New Tests

### Basic Test Structure

```python
import pytest
from src.agents.product_upload import validate_product_data

def test_my_feature(temp_databases, llm_stub):
    """Test description."""
    # Arrange: Set up test data
    state = {
        "product_data": {...},
        # ... other fields
    }
    
    # Act: Execute function
    result = validate_product_data(state)
    
    # Assert: Verify results
    assert result["status"] == "in_progress"
    assert result["current_step"] == 1
```

### Using Fixtures

```python
def test_with_fixtures(temp_databases, llm_stub, sample_product_data):
    """Test using multiple fixtures."""
    # temp_databases: Isolated databases
    # llm_stub: Stubbed LLM
    # sample_product_data: Sample data
    pass
```

### Testing Exceptions

```python
def test_error_handling(temp_databases):
    """Test error scenarios."""
    with pytest.raises(ValueError):
        function_that_raises_error()
```

### Testing Async Code

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_function()
    assert result is not None
```

## Architecture/Design Decisions

### Fixture-Based Isolation

**Why fixtures for database isolation?**

- **Clean State**: Each test starts with fresh databases
- **No Side Effects**: Tests don't affect each other
- **Automatic Cleanup**: Fixtures handle cleanup automatically

### LLM Stubbing

**Why stub LLM instead of mocking?**

- **Simplicity**: Stub is easier to understand and maintain
- **Determinism**: Consistent test results
- **Speed**: Faster test execution
- **Cost**: No API costs

### Test File Organization

**Why organize by module?**

- **Clarity**: Easy to find tests for specific modules
- **Maintainability**: Tests close to code they test
- **Parallel Execution**: Can run test files in parallel

## Related Modules

- **[`src/agents/`](../src/agents/README.md)** - Agent tests validate agent implementations
- **[`src/persistence/`](../src/persistence/README.md)** - Persistence tests validate database operations
- **[`src/protocol/`](../src/protocol/README.md)** - Protocol tests validate message handling
- **[`src/reconstruction/`](../src/reconstruction/README.md)** - Reconstruction tests validate recovery mechanisms
- **[`src/workflows/`](../src/workflows/README.md)** - Workflow tests validate orchestration

