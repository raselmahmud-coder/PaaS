"""Background service for agents to handle context requests."""

import asyncio
import logging
from typing import Any, Dict, Optional

from src.messaging.kafka_config import get_kafka_config, KafkaConfig
from src.messaging.producer import KafkaMessageProducer
from src.messaging.consumer import KafkaMessageConsumer
from src.messaging.context_handler import ContextRequestHandler

logger = logging.getLogger(__name__)


class AgentContextService:
    """Background service that handles context requests from peer agents.
    
    This service:
    1. Subscribes to the context request topic
    2. When REQUEST_CONTEXT is received, queries local database for interactions
    3. Publishes PROVIDE_CONTEXT response with relevant data
    """
    
    def __init__(
        self,
        agent_id: str,
        config: Optional[KafkaConfig] = None,
    ):
        """Initialize the agent context service.
        
        Args:
            agent_id: The ID of the agent this service belongs to.
            config: Optional Kafka configuration.
        """
        self.agent_id = agent_id
        self.config = config or get_kafka_config()
        
        # Initialize components
        self.context_handler = ContextRequestHandler(agent_id)
        self.producer: Optional[KafkaMessageProducer] = None
        self.consumer: Optional[KafkaMessageConsumer] = None
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the context service."""
        if self._running:
            return
        
        logger.info(f"Starting context service for agent {self.agent_id}")
        
        # Initialize producer
        self.producer = KafkaMessageProducer(self.config)
        await self.producer.start()
        
        # Initialize consumer for context request topic
        self.consumer = KafkaMessageConsumer(
            topics=[self.config.context_request_topic],
            group_id=self.config.get_consumer_group(self.agent_id),
            config=self.config,
        )
        
        # Register the context request handler
        self.consumer.register_handler(
            message_type="REQUEST_CONTEXT",
            handler=self._handle_context_request,
        )
        
        await self.consumer.start()
        
        # Start consuming in background
        self._task = self.consumer.start_consuming()
        self._running = True
        
        logger.info(f"Context service started for agent {self.agent_id}")
    
    async def stop(self) -> None:
        """Stop the context service."""
        if not self._running:
            return
        
        logger.info(f"Stopping context service for agent {self.agent_id}")
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        if self.consumer:
            await self.consumer.stop()
            self.consumer = None
        
        if self.producer:
            await self.producer.stop()
            self.producer = None
        
        logger.info(f"Context service stopped for agent {self.agent_id}")
    
    async def _handle_context_request(self, message: Dict[str, Any]) -> None:
        """Handle an incoming REQUEST_CONTEXT message.
        
        Args:
            message: The REQUEST_CONTEXT message.
        """
        requester_id = message.get("requester_id")
        failed_agent_id = message.get("failed_agent_id")
        thread_id = message.get("thread_id")
        response_topic = message.get("response_topic")
        
        logger.info(
            f"Agent {self.agent_id} received context request for "
            f"failed agent {failed_agent_id} from {requester_id}"
        )
        
        # Skip if this is a request about ourselves
        if failed_agent_id == self.agent_id:
            logger.debug("Ignoring context request about self")
            return
        
        # Get context data
        context_data = await self.context_handler.handle_context_request(message)
        
        # Skip if no interactions found
        if not context_data.get("interactions") and not context_data.get("memory_state", {}).get("has_interactions"):
            logger.debug(
                f"No interactions found between {self.agent_id} and {failed_agent_id}"
            )
            return
        
        # Publish response
        if self.producer and response_topic:
            await self.producer.publish_context_response(
                response_topic=response_topic,
                responder_id=self.agent_id,
                failed_agent_id=failed_agent_id,
                thread_id=thread_id,
                interactions=context_data.get("interactions", []),
                memory_state=context_data.get("memory_state", {}),
            )
            logger.info(
                f"Agent {self.agent_id} sent context response for {failed_agent_id}"
            )
    
    @property
    def is_running(self) -> bool:
        """Check if the service is currently running."""
        return self._running
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class AgentContextServiceManager:
    """Manages multiple agent context services."""
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        """Initialize the service manager.
        
        Args:
            config: Optional Kafka configuration.
        """
        self.config = config or get_kafka_config()
        self._services: Dict[str, AgentContextService] = {}
    
    async def register_agent(self, agent_id: str) -> AgentContextService:
        """Register and start a context service for an agent.
        
        Args:
            agent_id: The agent ID to register.
            
        Returns:
            The started AgentContextService.
        """
        if agent_id in self._services:
            return self._services[agent_id]
        
        service = AgentContextService(agent_id, self.config)
        await service.start()
        self._services[agent_id] = service
        
        return service
    
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister and stop a context service for an agent.
        
        Args:
            agent_id: The agent ID to unregister.
        """
        if agent_id not in self._services:
            return
        
        service = self._services.pop(agent_id)
        await service.stop()
    
    async def stop_all(self) -> None:
        """Stop all registered context services."""
        for agent_id in list(self._services.keys()):
            await self.unregister_agent(agent_id)
    
    def get_service(self, agent_id: str) -> Optional[AgentContextService]:
        """Get the context service for an agent.
        
        Args:
            agent_id: The agent ID to get the service for.
            
        Returns:
            The AgentContextService or None if not registered.
        """
        return self._services.get(agent_id)


# Global service manager instance
_service_manager: Optional[AgentContextServiceManager] = None


def get_service_manager() -> AgentContextServiceManager:
    """Get or create the global service manager."""
    global _service_manager
    if _service_manager is None:
        _service_manager = AgentContextServiceManager()
    return _service_manager

