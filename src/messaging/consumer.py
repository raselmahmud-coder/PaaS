"""Async Kafka consumer with message handler callbacks."""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Awaitable
from datetime import datetime

from src.messaging.kafka_config import get_kafka_config, KafkaConfig

logger = logging.getLogger(__name__)

# Type for message handler callbacks
MessageHandler = Callable[[Dict[str, Any]], Awaitable[None]]


class KafkaMessageConsumer:
    """Async Kafka consumer for receiving protocol messages."""
    
    def __init__(
        self,
        topics: List[str],
        group_id: str,
        config: Optional[KafkaConfig] = None,
    ):
        """Initialize the Kafka consumer.
        
        Args:
            topics: List of topics to subscribe to.
            group_id: Consumer group ID.
            config: Optional Kafka configuration.
        """
        self.config = config or get_kafka_config()
        self.topics = topics
        self.group_id = group_id
        self._consumer = None
        self._started = False
        self._handlers: Dict[str, List[MessageHandler]] = {}
        self._running = False
        self._consume_task: Optional[asyncio.Task] = None
    
    def register_handler(
        self,
        message_type: str,
        handler: MessageHandler,
    ) -> None:
        """Register a handler for a specific message type.
        
        Args:
            message_type: The message type to handle (e.g., "REQUEST_CONTEXT").
            handler: Async callback function to handle the message.
        """
        if message_type not in self._handlers:
            self._handlers[message_type] = []
        self._handlers[message_type].append(handler)
    
    async def start(self) -> None:
        """Start the Kafka consumer."""
        if self._started:
            return
        
        try:
            from aiokafka import AIOKafkaConsumer
            
            self._consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
            )
            await self._consumer.start()
            self._started = True
            logger.info(
                f"Kafka consumer started, subscribed to {self.topics}, "
                f"group: {self.group_id}"
            )
        except ImportError:
            logger.warning("aiokafka not installed. Kafka consumer will operate in mock mode.")
            self._consumer = None
            self._started = True
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the Kafka consumer."""
        self._running = False
        
        if self._consume_task:
            self._consume_task.cancel()
            try:
                await self._consume_task
            except asyncio.CancelledError:
                pass
            self._consume_task = None
        
        if self._consumer and self._started:
            try:
                await self._consumer.stop()
            except Exception as e:
                logger.error(f"Error stopping Kafka consumer: {e}")
            finally:
                self._started = False
                self._consumer = None
    
    async def _process_message(self, message: Dict[str, Any]) -> None:
        """Process a received message by invoking registered handlers.
        
        Args:
            message: The message payload.
        """
        message_type = message.get("message_type")
        if not message_type:
            logger.warning(f"Received message without message_type: {message}")
            return
        
        handlers = self._handlers.get(message_type, [])
        if not handlers:
            logger.debug(f"No handlers registered for message type: {message_type}")
            return
        
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Error in message handler for {message_type}: {e}")
    
    async def consume_loop(self) -> None:
        """Start consuming messages in a loop."""
        if not self._started:
            await self.start()
        
        if self._consumer is None:
            # Mock mode - just wait
            logger.info("[MOCK] Consumer started in mock mode, no real messages will be received")
            while self._running:
                await asyncio.sleep(1)
            return
        
        self._running = True
        logger.info("Starting message consumption loop")
        
        try:
            async for msg in self._consumer:
                if not self._running:
                    break
                
                try:
                    await self._process_message(msg.value)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        except asyncio.CancelledError:
            logger.info("Consumer loop cancelled")
        except Exception as e:
            logger.error(f"Error in consume loop: {e}")
        finally:
            self._running = False
    
    def start_consuming(self) -> asyncio.Task:
        """Start consuming messages in a background task.
        
        Returns:
            The asyncio Task running the consumer loop.
        """
        self._consume_task = asyncio.create_task(self.consume_loop())
        return self._consume_task
    
    async def collect_messages(
        self,
        timeout: float,
        message_type: Optional[str] = None,
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Collect messages for a specified duration.
        
        This is useful for collecting PROVIDE_CONTEXT responses within a timeout.
        
        Args:
            timeout: How long to collect messages (seconds).
            message_type: Optional filter for specific message types.
            max_messages: Optional maximum number of messages to collect.
            
        Returns:
            List of collected messages.
        """
        if not self._started:
            await self.start()
        
        if self._consumer is None:
            # Mock mode - return empty list
            logger.info(f"[MOCK] Collecting messages for {timeout}s (mock mode)")
            await asyncio.sleep(min(timeout, 0.1))  # Short delay in mock mode
            return []
        
        collected: List[Dict[str, Any]] = []
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                break
            
            if max_messages and len(collected) >= max_messages:
                break
            
            remaining = timeout - elapsed
            try:
                # Poll for messages with remaining timeout
                messages = await asyncio.wait_for(
                    self._poll_once(),
                    timeout=min(remaining, self.config.consumer_poll_timeout),
                )
                
                for msg in messages:
                    if message_type is None or msg.get("message_type") == message_type:
                        collected.append(msg)
                        
                        if max_messages and len(collected) >= max_messages:
                            break
                            
            except asyncio.TimeoutError:
                continue
        
        logger.info(f"Collected {len(collected)} messages in {timeout}s")
        return collected
    
    async def _poll_once(self) -> List[Dict[str, Any]]:
        """Poll for messages once.
        
        Returns:
            List of message values.
        """
        if self._consumer is None:
            return []
        
        # Get messages from partition
        data = await self._consumer.getmany(
            timeout_ms=int(self.config.consumer_poll_timeout * 1000)
        )
        
        messages = []
        for tp, records in data.items():
            for record in records:
                messages.append(record.value)
        
        return messages

    async def consume_once(self, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """Consume messages once with a timeout.
        
        This is useful for testing - consumes available messages within timeout
        and invokes registered handlers.
        
        Args:
            timeout: Maximum time to wait for messages (seconds).
            
        Returns:
            List of consumed message values.
        """
        if not self._started:
            await self.start()
        
        if self._consumer is None:
            # Mock mode
            logger.info(f"[MOCK] consume_once called with timeout={timeout}")
            await asyncio.sleep(min(timeout, 0.1))
            return []
        
        consumed = []
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                break
            
            remaining = timeout - elapsed
            try:
                data = await asyncio.wait_for(
                    self._consumer.getmany(timeout_ms=int(min(remaining, 1.0) * 1000)),
                    timeout=remaining + 1.0,
                )
                
                for tp, records in data.items():
                    for record in records:
                        msg = record.value
                        consumed.append(msg)
                        # Invoke handlers
                        msg_type = msg.get("type") or msg.get("message_type")
                        if msg_type and msg_type in self._handlers:
                            for handler in self._handlers[msg_type]:
                                try:
                                    await handler(msg)
                                except Exception as e:
                                    logger.error(f"Handler error: {e}")
                
                # If we got messages, return them
                if consumed:
                    break
                    
            except asyncio.TimeoutError:
                continue
        
        logger.debug(f"consume_once returned {len(consumed)} messages")
        return consumed
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

