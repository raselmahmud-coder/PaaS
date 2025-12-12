"""Checkpointer factory using LangGraph's built-in checkpointers."""

import sqlite3

import aiosqlite
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.config import get_database_path, settings


def get_checkpointer(use_memory: bool = False):
    """
    Get a LangGraph checkpointer instance (synchronous).

    Args:
        use_memory: If True, use MemorySaver (in-memory, no persistence).
                   If False, use SqliteSaver (persistent SQLite storage).

    Returns:
        MemorySaver or SqliteSaver instance
    """
    if use_memory:
        return MemorySaver()

    # Get database path and ensure directory exists
    db_path = get_database_path()
    checkpoint_db_path = db_path.parent / f"{db_path.stem}_checkpoint{db_path.suffix}"
    checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(checkpoint_db_path), check_same_thread=False)

    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    return checkpointer


async def get_async_checkpointer(use_memory: bool = False):
    """
    Get a LangGraph async checkpointer instance for async workflows.

    Args:
        use_memory: If True, use MemorySaver (in-memory, no persistence).
                   If False, use AsyncSqliteSaver (persistent SQLite storage).

    Returns:
        MemorySaver or AsyncSqliteSaver instance
    """
    if use_memory:
        return MemorySaver()

    # Get database path and ensure directory exists
    db_path = get_database_path()
    checkpoint_db_path = db_path.parent / f"{db_path.stem}_checkpoint{db_path.suffix}"
    checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create async connection
    conn = await aiosqlite.connect(str(checkpoint_db_path))

    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()

    return checkpointer


# For backward compatibility
def create_checkpointer():
    """Create checkpointer based on settings."""
    use_memory = settings.database_url.startswith("memory://")
    return get_checkpointer(use_memory=use_memory)
