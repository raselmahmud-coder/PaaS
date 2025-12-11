"""Configuration management for the PaaS system."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Allow extra fields in .env file (ignore them)
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields instead of raising errors
    )

    # LLM Provider Configuration
    llm_provider: str = os.getenv("LLM_PROVIDER", "moonshot")  # "openai" or "moonshot"
    llm_api_key: str = os.getenv(
        "LLM_API_KEY",
        os.getenv("MOONSHOT_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    )
    # Base URL: empty for OpenAI, set for Moonshot (defaults to Moonshot if provider is moonshot)
    llm_base_url: str = os.getenv(
        "LLM_BASE_URL",
        "https://api.moonshot.cn/v1"
        if os.getenv("LLM_PROVIDER", "moonshot") == "moonshot"
        else "",
    )
    # Model: defaults based on provider
    llm_model: str = os.getenv(
        "LLM_MODEL",
        "kimi-k2-turbo-preview"
        if os.getenv("LLM_PROVIDER", "moonshot") == "moonshot"
        else "gpt-4o-mini",
    )
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # Legacy OpenAI Configuration (for backward compatibility)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Database Configuration
    database_url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{Path(__file__).parent.parent / 'data' / 'agent_system.db'}",
    )

    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Checkpoint Configuration
    checkpoint_interval_seconds: int = int(
        os.getenv("CHECKPOINT_INTERVAL_SECONDS", "30")
    )

    # Agent Configuration
    agent_timeout_seconds: int = int(os.getenv("AGENT_TIMEOUT_SECONDS", "30"))

    # ==========================================================================
    # Kafka Configuration (for peer context retrieval)
    # ==========================================================================
    
    # Kafka broker connection
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    
    # Kafka topic names
    kafka_context_request_topic: str = os.getenv(
        "KAFKA_CONTEXT_REQUEST_TOPIC", "agent.context.request"
    )
    kafka_context_response_topic_prefix: str = os.getenv(
        "KAFKA_CONTEXT_RESPONSE_TOPIC_PREFIX", "agent.context.response"
    )
    
    # Kafka consumer settings
    kafka_consumer_group_prefix: str = os.getenv(
        "KAFKA_CONSUMER_GROUP_PREFIX", "paas-agent"
    )
    
    # Peer context retrieval settings
    peer_context_enabled: bool = os.getenv("PEER_CONTEXT_ENABLED", "true").lower() == "true"
    peer_context_timeout_seconds: float = float(
        os.getenv("PEER_CONTEXT_TIMEOUT_SECONDS", "5.0")
    )
    peer_context_time_window_seconds: int = int(
        os.getenv("PEER_CONTEXT_TIME_WINDOW_SECONDS", "3600")
    )


# Global settings instance
settings = Settings()


def get_database_path() -> Path:
    """Get the database file path from the database URL."""
    if settings.database_url.startswith("sqlite:///"):
        db_path = Path(settings.database_url.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path
    raise ValueError(f"Unsupported database URL: {settings.database_url}")
