"""Async Kafka producer for publishing messages."""

import asyncio
import json
import logging
from typing import Any, Dict, Optional
from datetime import datetime

from src.messaging.kafka_config import get_kafka_config, KafkaConfig

logger = logging.getLogger(__name__)


class KafkaMessageProducer:
    """Async Kafka producer for publishing protocol messages."""
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        """Initialize the Kafka producer.
        
        Args:
            config: Optional Kafka configuration. Uses global config if not provided.
        """
        self.config = config or get_kafka_config()
        self._producer = None
        self._started = False
    
    async def start(self) -> None:
        """Start the Kafka producer."""
        if self._started:
            return
        
        try:
            from aiokafka import AIOKafkaProducer
            
            # Use longer timeout for bootstrap (Docker Desktop on Windows needs ~40s)
            # request_timeout_ms affects bootstrap, so use at least 60s
            bootstrap_timeout_ms = max(60000, int(self.config.producer_timeout * 1000))
            
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                max_request_size=self.config.max_message_size,
                request_timeout_ms=bootstrap_timeout_ms,
                retry_backoff_ms=self.config.retry_backoff_ms,
            )
            
            await self._producer.start()
            self._started = True
            logger.info(f"Kafka producer started, connected to {self.config.bootstrap_servers}")
        except ImportError:
            logger.warning("aiokafka not installed. Kafka producer will operate in mock mode.")
            self._producer = None
            self._started = True
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if self._producer and self._started:
            try:
                await self._producer.stop()
            except Exception as e:
                logger.error(f"Error stopping Kafka producer: {e}")
            finally:
                self._started = False
                self._producer = None
    
    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
    ) -> bool:
        """Publish a message to a Kafka topic.
        
        Args:
            topic: The Kafka topic to publish to.
            message: The message payload as a dictionary.
            key: Optional message key for partitioning.
            
        Returns:
            True if message was published successfully, False otherwise.
        """
        if not self._started:
            await self.start()
        
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
        
        if self._producer is None:
            # Mock mode - log the message
            logger.info(f"[MOCK] Publishing to {topic}: {message}")
            return True
        
        try:
            await self._producer.send_and_wait(topic, value=message, key=key)
            logger.debug(f"Published message to {topic}: {message.get('message_type', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish message to {topic}: {e}")
            return False
    
    async def publish_context_request(
        self,
        requester_id: str,
        failed_agent_id: str,
        thread_id: str,
        time_window_seconds: Optional[int] = None,
    ) -> bool:
        """Publish a REQUEST_CONTEXT message to query peer agents.
        
        Args:
            requester_id: ID of the agent/module requesting context.
            failed_agent_id: ID of the failed agent to get context about.
            thread_id: Thread ID of the failed workflow.
            time_window_seconds: How far back to look for interactions.
            
        Returns:
            True if message was published successfully.
        """
        message = {
            "message_type": "REQUEST_CONTEXT",
            "requester_id": requester_id,
            "failed_agent_id": failed_agent_id,
            "thread_id": thread_id,
            "time_window_seconds": time_window_seconds or 3600,  # Default 1 hour
            "response_topic": self.config.get_response_topic(requester_id),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        return await self.publish(
            topic=self.config.context_request_topic,
            message=message,
            key=failed_agent_id,
        )
    
    async def publish_context_response(
        self,
        response_topic: str,
        responder_id: str,
        failed_agent_id: str,
        thread_id: str,
        interactions: list,
        memory_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publish a PROVIDE_CONTEXT response message.
        
        Args:
            response_topic: The topic to publish the response to.
            responder_id: ID of the agent providing context.
            failed_agent_id: ID of the failed agent.
            thread_id: Thread ID of the workflow.
            interactions: List of interaction events with the failed agent.
            memory_state: Optional agent memory/state relevant to the failed agent.
            
        Returns:
            True if message was published successfully.
        """
        message = {
            "message_type": "PROVIDE_CONTEXT",
            "responder_id": responder_id,
            "failed_agent_id": failed_agent_id,
            "thread_id": thread_id,
            "interactions": interactions,
            "memory_state": memory_state or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        return await self.publish(
            topic=response_topic,
            message=message,
            key=responder_id,
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

