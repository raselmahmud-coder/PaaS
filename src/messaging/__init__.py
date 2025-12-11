"""Messaging module for Kafka-based inter-agent communication."""

from src.messaging.kafka_config import KafkaConfig, get_kafka_config
from src.messaging.producer import KafkaMessageProducer
from src.messaging.consumer import KafkaMessageConsumer
from src.messaging.context_handler import ContextRequestHandler
from src.messaging.agent_context_service import AgentContextService

__all__ = [
    "KafkaConfig",
    "get_kafka_config",
    "KafkaMessageProducer",
    "KafkaMessageConsumer",
    "ContextRequestHandler",
    "AgentContextService",
]

