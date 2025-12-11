"""SQLAlchemy models for persistence."""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Index, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import get_database_path, settings

Base = declarative_base()


class Checkpoint(Base):
    """Checkpoint model for agent state persistence."""

    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(100), nullable=False, index=True)
    checkpoint_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Index for efficient queries
    __table_args__ = (Index("idx_thread_created", "thread_id", "created_at"),)

    def __repr__(self):
        return f"<Checkpoint(thread_id={self.thread_id}, created_at={self.created_at})>"


class AgentEvent(Base):
    """Event log model for agent actions."""

    __tablename__ = "agent_events"

    event_id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(100), nullable=False, index=True)
    thread_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    event_type = Column(
        String(50), nullable=False
    )  # "step_start", "step_complete", "error"
    step_name = Column(String(100), nullable=True)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    state_snapshot = Column(JSON, nullable=True)

    # Indexes for efficient queries
    __table_args__ = (
        Index("idx_agent_timestamp", "agent_id", "timestamp"),
        Index("idx_thread_timestamp", "thread_id", "timestamp"),
    )

    def __repr__(self):
        return f"<AgentEvent(agent_id={self.agent_id}, event_type={self.event_type}, timestamp={self.timestamp})>"

    def to_dict(self):
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "event_type": self.event_type,
            "step_name": self.step_name,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "state_snapshot": self.state_snapshot,
        }


# Database engine and session
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        # Ensure database directory exists
        db_path = get_database_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        _engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False}
            if "sqlite" in settings.database_url
            else {},
        )
    return _engine


def get_session():
    """Get database session."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def init_db():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"Database initialized at: {get_database_path()}")


if __name__ == "__main__":
    # Allow running this module to initialize the database
    init_db()
