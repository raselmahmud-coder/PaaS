"""Tests for peer context retrieval functionality."""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from src.messaging.kafka_config import KafkaConfig, set_kafka_config, reset_kafka_config
from src.messaging.producer import KafkaMessageProducer
from src.messaging.consumer import KafkaMessageConsumer
from src.messaging.context_handler import ContextRequestHandler
from src.messaging.agent_context_service import AgentContextService
from src.protocol.messages import (
    RequestContextMessage,
    ProvideContextMessage,
    RequestContextPayload,
    ProvideContextPayload,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def kafka_config():
    """Provide a test Kafka configuration."""
    config = KafkaConfig(
        bootstrap_servers="localhost:9092",
        context_request_topic="test.context.request",
        context_response_topic_prefix="test.context.response",
        consumer_group_prefix="test-agent",
        context_collection_timeout=1.0,
    )
    set_kafka_config(config)
    yield config
    reset_kafka_config()


@pytest.fixture
def sample_request_message():
    """Provide a sample REQUEST_CONTEXT message."""
    return RequestContextMessage.create(
        requester_id="reconstructor-123",
        failed_agent_id="product-agent-1",
        thread_id="thread-456",
        time_window_seconds=3600,
        response_topic="test.context.response.reconstructor-123",
    )


@pytest.fixture
def sample_response_message():
    """Provide a sample PROVIDE_CONTEXT message."""
    return ProvideContextMessage.create(
        responder_id="marketing-agent-1",
        requester_id="reconstructor-123",
        failed_agent_id="product-agent-1",
        thread_id="thread-456",
        interactions=[
            {
                "event_id": 1,
                "event_type": "protocol_handoff",
                "step_name": "product_to_marketing",
                "timestamp": datetime.utcnow().isoformat(),
            }
        ],
        memory_state={
            "has_interactions": True,
            "interaction_count": 5,
            "last_interaction_type": "protocol_handoff",
        },
    )


# =============================================================================
# Protocol Message Tests
# =============================================================================


class TestRequestContextMessage:
    """Tests for RequestContextMessage."""

    def test_create_request_context_message(self):
        """Test creating a REQUEST_CONTEXT message."""
        msg = RequestContextMessage.create(
            requester_id="reconstructor-1",
            failed_agent_id="agent-failed",
            thread_id="thread-123",
            time_window_seconds=1800,
            response_topic="response.topic",
        )
        
        assert msg.message_type == "REQUEST_CONTEXT"
        assert msg.sender == "reconstructor-1"
        assert msg.receiver == "broadcast"
        assert msg.payload["failed_agent_id"] == "agent-failed"
        assert msg.payload["thread_id"] == "thread-123"
        assert msg.payload["time_window_seconds"] == 1800
        assert msg.payload["response_topic"] == "response.topic"

    def test_request_context_payload_validation(self):
        """Test RequestContextPayload validation."""
        payload = RequestContextPayload(
            failed_agent_id="agent-1",
            thread_id="thread-1",
            requester_id="requester-1",
        )
        
        assert payload.time_window_seconds == 3600  # default
        assert payload.response_topic is None


class TestProvideContextMessage:
    """Tests for ProvideContextMessage."""

    def test_create_provide_context_message(self):
        """Test creating a PROVIDE_CONTEXT message."""
        interactions = [
            {"event_type": "step_complete", "step_name": "generate_listing"}
        ]
        memory_state = {"has_interactions": True}
        
        msg = ProvideContextMessage.create(
            responder_id="agent-1",
            requester_id="reconstructor-1",
            failed_agent_id="agent-failed",
            thread_id="thread-123",
            interactions=interactions,
            memory_state=memory_state,
        )
        
        assert msg.message_type == "PROVIDE_CONTEXT"
        assert msg.sender == "agent-1"
        assert msg.receiver == "reconstructor-1"
        assert msg.get_interactions() == interactions
        assert msg.get_memory_state() == memory_state

    def test_provide_context_payload_defaults(self):
        """Test ProvideContextPayload default values."""
        payload = ProvideContextPayload(
            responder_id="agent-1",
            failed_agent_id="agent-2",
            thread_id="thread-1",
        )
        
        assert payload.interactions == []
        assert payload.memory_state == {}


# =============================================================================
# Kafka Config Tests
# =============================================================================


class TestKafkaConfig:
    """Tests for Kafka configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KafkaConfig()
        
        assert config.bootstrap_servers == "localhost:9092"
        assert config.context_request_topic == "agent.context.request"
        assert config.context_collection_timeout == 5.0

    def test_get_response_topic(self, kafka_config):
        """Test response topic generation."""
        topic = kafka_config.get_response_topic("requester-123")
        assert topic == "test.context.response.requester-123"

    def test_get_consumer_group(self, kafka_config):
        """Test consumer group ID generation."""
        group = kafka_config.get_consumer_group("agent-1")
        assert group == "test-agent-agent-1"


# =============================================================================
# Context Handler Tests
# =============================================================================


class TestContextRequestHandler:
    """Tests for ContextRequestHandler."""

    def test_handler_initialization(self):
        """Test handler initialization."""
        handler = ContextRequestHandler("agent-1")
        assert handler.agent_id == "agent-1"

    def test_data_contains_agent_ref_string(self):
        """Test finding agent references in strings."""
        handler = ContextRequestHandler("agent-1")
        
        assert handler._data_contains_agent_ref("message from agent-2", "agent-2")
        assert not handler._data_contains_agent_ref("message from agent-2", "agent-3")

    def test_data_contains_agent_ref_dict(self):
        """Test finding agent references in dictionaries."""
        handler = ContextRequestHandler("agent-1")
        
        data = {"sender": "agent-2", "content": "hello"}
        assert handler._data_contains_agent_ref(data, "agent-2")
        assert not handler._data_contains_agent_ref(data, "agent-3")

    def test_data_contains_agent_ref_nested(self):
        """Test finding agent references in nested structures."""
        handler = ContextRequestHandler("agent-1")
        
        data = {
            "outer": {
                "inner": {
                    "receiver": "agent-target"
                }
            }
        }
        assert handler._data_contains_agent_ref(data, "agent-target")

    @patch('src.messaging.context_handler.event_store')
    def test_query_interactions_empty(self, mock_event_store):
        """Test querying interactions when none exist."""
        mock_event_store.get_events.return_value = []
        
        handler = ContextRequestHandler("agent-1")
        interactions = handler.query_interactions_with_agent("agent-2")
        
        assert interactions == []
        mock_event_store.get_events.assert_called_once()

    @patch('src.messaging.context_handler.event_store')
    def test_get_memory_state_no_interactions(self, mock_event_store):
        """Test memory state when no interactions exist."""
        mock_event_store.get_events.return_value = []
        
        handler = ContextRequestHandler("agent-1")
        memory = handler.get_memory_state_for_agent("agent-2")
        
        assert memory["has_interactions"] is False
        assert memory["interaction_count"] == 0


# =============================================================================
# Producer Tests (Mock Mode)
# =============================================================================


class TestKafkaMessageProducer:
    """Tests for KafkaMessageProducer in mock mode."""

    @pytest.mark.asyncio
    async def test_producer_mock_mode(self, kafka_config):
        """Test producer operates in mock mode when aiokafka not available."""
        producer = KafkaMessageProducer(kafka_config)
        await producer.start()
        
        # Should succeed even in mock mode
        success = await producer.publish(
            topic="test.topic",
            message={"type": "test", "data": "value"},
        )
        
        assert success is True
        await producer.stop()

    @pytest.mark.asyncio
    async def test_publish_context_request(self, kafka_config):
        """Test publishing a context request message."""
        async with KafkaMessageProducer(kafka_config) as producer:
            success = await producer.publish_context_request(
                requester_id="reconstructor-1",
                failed_agent_id="agent-failed",
                thread_id="thread-123",
            )
            assert success is True

    @pytest.mark.asyncio
    async def test_publish_context_response(self, kafka_config):
        """Test publishing a context response message."""
        async with KafkaMessageProducer(kafka_config) as producer:
            success = await producer.publish_context_response(
                response_topic="test.response.topic",
                responder_id="agent-1",
                failed_agent_id="agent-failed",
                thread_id="thread-123",
                interactions=[],
            )
            assert success is True


# =============================================================================
# Consumer Tests (Mock Mode)
# =============================================================================


class TestKafkaMessageConsumer:
    """Tests for KafkaMessageConsumer in mock mode."""

    @pytest.mark.asyncio
    async def test_consumer_mock_mode(self, kafka_config):
        """Test consumer operates in mock mode."""
        consumer = KafkaMessageConsumer(
            topics=["test.topic"],
            group_id="test-group",
            config=kafka_config,
        )
        await consumer.start()
        
        # Should be started in mock mode
        assert consumer._started is True
        
        await consumer.stop()

    @pytest.mark.asyncio
    async def test_register_handler(self, kafka_config):
        """Test registering message handlers."""
        consumer = KafkaMessageConsumer(
            topics=["test.topic"],
            group_id="test-group",
            config=kafka_config,
        )
        
        handler_called = False
        
        async def test_handler(msg):
            nonlocal handler_called
            handler_called = True
        
        consumer.register_handler("TEST_TYPE", test_handler)
        
        assert "TEST_TYPE" in consumer._handlers
        assert len(consumer._handlers["TEST_TYPE"]) == 1

    @pytest.mark.asyncio
    async def test_collect_messages_mock(self, kafka_config):
        """Test collecting messages returns empty in mock mode."""
        async with KafkaMessageConsumer(
            topics=["test.topic"],
            group_id="test-group",
            config=kafka_config,
        ) as consumer:
            messages = await consumer.collect_messages(timeout=0.5)
            assert messages == []


# =============================================================================
# Agent Context Service Tests
# =============================================================================


class TestAgentContextService:
    """Tests for AgentContextService."""

    @pytest.mark.asyncio
    async def test_service_start_stop(self, kafka_config):
        """Test starting and stopping the service."""
        service = AgentContextService("agent-1", kafka_config)
        
        assert not service.is_running
        
        await service.start()
        assert service.is_running
        
        await service.stop()
        assert not service.is_running

    @pytest.mark.asyncio
    async def test_service_context_manager(self, kafka_config):
        """Test using service as async context manager."""
        async with AgentContextService("agent-1", kafka_config) as service:
            assert service.is_running
        
        # Service should be stopped after exiting context
        assert not service.is_running


# =============================================================================
# Integration Tests (Mock Mode)
# =============================================================================


class TestPeerContextIntegration:
    """Integration tests for peer context retrieval flow."""

    @pytest.mark.asyncio
    @patch('src.reconstruction.reconstructor.event_store')
    async def test_reconstruction_with_peer_context_mock(self, mock_event_store, kafka_config):
        """Test reconstruction with peer context in mock mode."""
        from src.reconstruction.reconstructor import AgentReconstructor
        
        # Setup mock event store
        mock_event_store.get_events.return_value = []
        
        # Create reconstructor with peer context enabled
        reconstructor = AgentReconstructor(
            enable_peer_context=True,
            peer_context_timeout=0.5,
        )
        
        # Query peer agents (should return empty in mock mode)
        peer_context = await reconstructor.query_peer_agents(
            failed_agent_id="agent-failed",
            thread_id="thread-123",
        )
        
        # In mock mode, we should get empty results
        assert peer_context == []

    def test_reconstruction_prompt_with_peer_context(self, kafka_config):
        """Test that reconstruction prompt includes peer context."""
        from src.reconstruction.reconstructor import AgentReconstructor
        
        reconstructor = AgentReconstructor(enable_peer_context=True)
        
        checkpoint = {
            "ts": datetime.utcnow().isoformat(),
            "channel_values": {"status": "in_progress", "current_step": 2},
        }
        events = []
        peer_context = [
            {
                "responder_id": "marketing-agent-1",
                "interactions": [{"event_type": "protocol_handoff"}],
                "memory_state": {
                    "has_interactions": True,
                    "interaction_count": 3,
                    "last_interaction_type": "protocol_handoff",
                },
            }
        ]
        
        prompt = reconstructor._build_reconstruction_prompt_with_peer_context(
            checkpoint, events, peer_context
        )
        
        # Verify peer context is included in prompt
        assert "Peer Agent Context" in prompt
        assert "marketing-agent-1" in prompt
        assert "protocol_handoff" in prompt
        assert "peer_insights" in prompt

    def test_merge_state_with_peer_context(self, kafka_config):
        """Test merging state with peer context."""
        from src.reconstruction.reconstructor import AgentReconstructor
        
        reconstructor = AgentReconstructor(enable_peer_context=True)
        
        checkpoint = {
            "channel_values": {"status": "in_progress", "current_step": 2},
        }
        events = []
        inferred_data = {"status": "in_progress", "current_step": 2}
        peer_context = [
            {
                "responder_id": "agent-1",
                "memory_state": {
                    "has_interactions": True,
                    "interaction_count": 5,
                    "last_interaction_type": "step_complete",
                },
            }
        ]
        
        merged = reconstructor._merge_state_with_peer_context(
            checkpoint, events, inferred_data, peer_context
        )
        
        assert merged["peer_context_used"] is True
        assert merged["peer_agents_count"] == 1
        assert "peer_insights" in merged

