"""Kafka configuration and topic definitions."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KafkaConfig:
    """Configuration for Kafka connection and topics."""
    
    # Connection settings
    bootstrap_servers: str = field(default="localhost:9092")
    
    # Topic definitions
    context_request_topic: str = "agent.context.request"
    context_response_topic_prefix: str = "agent.context.response"
    
    # Consumer group settings
    consumer_group_prefix: str = "paas-agent"
    
    # Timeout settings (seconds)
    context_collection_timeout: float = 5.0
    producer_timeout: float = 10.0
    consumer_poll_timeout: float = 1.0
    
    # Retry settings
    max_retries: int = 3
    retry_backoff_ms: int = 100
    
    # Message settings
    max_message_size: int = 1048576  # 1MB
    
    def get_response_topic(self, requester_id: str) -> str:
        """Get the dedicated response topic for a specific requester."""
        return f"{self.context_response_topic_prefix}.{requester_id}"
    
    def get_consumer_group(self, agent_id: str) -> str:
        """Get the consumer group ID for an agent."""
        return f"{self.consumer_group_prefix}-{agent_id}"


# Global config instance
_kafka_config: Optional[KafkaConfig] = None


def get_kafka_config() -> KafkaConfig:
    """Get or create the global Kafka configuration from app settings."""
    global _kafka_config
    if _kafka_config is None:
        try:
            from src.config import settings
            _kafka_config = KafkaConfig(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                context_request_topic=settings.kafka_context_request_topic,
                context_response_topic_prefix=settings.kafka_context_response_topic_prefix,
                consumer_group_prefix=settings.kafka_consumer_group_prefix,
                context_collection_timeout=settings.peer_context_timeout_seconds,
            )
        except ImportError:
            # Fallback to defaults if settings module not available
            _kafka_config = KafkaConfig()
    return _kafka_config


def set_kafka_config(config: KafkaConfig) -> None:
    """Set custom Kafka configuration (for testing)."""
    global _kafka_config
    _kafka_config = config


def reset_kafka_config() -> None:
    """Reset Kafka configuration to force reload from settings."""
    global _kafka_config
    _kafka_config = None

