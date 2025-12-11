"""Pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path

import pytest

from src.config import settings
import src.persistence.models as models
from src.persistence.event_store import event_store


@pytest.fixture
def temp_databases(tmp_path):
    """
    Create isolated temp databases for state and checkpoint.
    Resets SQLAlchemy engine/session caches and cleans up files.
    """
    state_db = tmp_path / "agent_system.db"
    checkpoint_db = tmp_path / "agent_system_checkpoint.db"

    # Override database URL for tests
    original_url = settings.database_url
    settings.database_url = f"sqlite:///{state_db}"

    # Reset cached engine/session so new URL is used
    models._engine = None
    models._SessionLocal = None

    try:
        yield {
            "state_db": state_db,
            "checkpoint_db": checkpoint_db,
        }
    finally:
        # Cleanup files
        for path in [state_db, checkpoint_db]:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
        settings.database_url = original_url
        models._engine = None
        models._SessionLocal = None


@pytest.fixture
def sample_product_data():
    """Sample product data for testing."""
    return {
        "name": "Test Product",
        "description": "A test product description",
        "price": 29.99,
        "category": "Electronics",
        "sku": "TEST-001"
    }


@pytest.fixture
def llm_stub(monkeypatch):
    """Provide a deterministic LLM stub to avoid network calls."""

    class DummyLLM:
        def __init__(self, content: str = "stubbed response"):
            self.content = content

        def invoke(self, messages):
            return type("LLMResult", (), {"content": self.content})

    dummy = DummyLLM("stubbed response")

    # Patch LLM instances used in agents
    monkeypatch.setattr("src.agents.product_upload.llm", dummy)
    monkeypatch.setattr("src.agents.marketing.llm", dummy, raising=False)

    yield dummy

