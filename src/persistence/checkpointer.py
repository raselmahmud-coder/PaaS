"""Checkpointer factory using LangGraph's built-in checkpointers."""

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from src.config import settings, get_database_path


def get_checkpointer(use_memory: bool = False):
    """
    Get a LangGraph checkpointer instance.
    
    Args:
        use_memory: If True, use MemorySaver (in-memory, no persistence).
                   If False, use SqliteSaver (persistent SQLite storage).
    
    Returns:
        MemorySaver or SqliteSaver instance
    """
    if use_memory:
        return MemorySaver()
    
    # Get database path and ensure directory exists
    # Use a separate database file for LangGraph checkpointer to avoid conflicts
    # with custom models (Checkpoint, AgentEvent) in src.persistence.models
    db_path = get_database_path()
    checkpoint_db_path = db_path.parent / f"{db_path.stem}_checkpoint{db_path.suffix}"
    checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create sqlite3 connection and pass to SqliteSaver
    # SqliteSaver requires a connection object, not a connection string
    # check_same_thread=False allows the connection to be used across threads
    conn = sqlite3.connect(str(checkpoint_db_path), check_same_thread=False)
    
    checkpointer = SqliteSaver(conn)
    
    # Initialize database schema (creates required tables)
    # This will create the checkpoints and writes tables with LangGraph's schema
    checkpointer.setup()
    
    return checkpointer


# For backward compatibility, create a factory function
def create_checkpointer():
    """Create checkpointer based on settings."""
    # Check if we should use memory (for testing)
    use_memory = settings.database_url.startswith("memory://")
    return get_checkpointer(use_memory=use_memory)
