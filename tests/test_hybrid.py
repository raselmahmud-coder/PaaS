"""Tests for hybrid reconstruction module."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from src.reconstruction.hybrid import (
    HybridReconstructor,
    HybridReconstructionResult,
    ReconstructionStrategy,
    hybrid_reconstruct,
)
from src.reconstruction.automata_reconstructor import (
    AutomataReconstructor,
    AutomataReconstructionResult,
    reconstruct_with_automata,
)
from src.automata.event_generator import generate_training_events


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def training_events():
    """Generate training events."""
    return generate_training_events(num_events=100, random_seed=42)


@pytest.fixture
def fresh_checkpoint():
    """Create a fresh checkpoint."""
    return {
        "agent_id": "agent-1",
        "thread_id": "thread-1",
        "status": "in_progress",
        "current_step": 1,
        "timestamp": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def old_checkpoint():
    """Create an old checkpoint."""
    old_time = datetime.utcnow() - timedelta(minutes=5)
    return {
        "agent_id": "agent-1",
        "thread_id": "thread-1",
        "status": "in_progress",
        "current_step": 0,
        "timestamp": old_time.isoformat(),
    }


@pytest.fixture
def sample_events():
    """Create sample events since checkpoint."""
    return [
        {
            "action_type": "validate_product_data",
            "output_data": {"status": "validated"},
            "timestamp": datetime.utcnow().isoformat(),
        },
        {
            "action_type": "generate_listing",
            "output_data": {"status": "generated"},
            "timestamp": datetime.utcnow().isoformat(),
        },
    ]


# =============================================================================
# HybridReconstructionResult Tests
# =============================================================================


class TestHybridReconstructionResult:
    """Tests for HybridReconstructionResult."""

    def test_to_dict(self):
        """Test result serialization."""
        result = HybridReconstructionResult(
            success=True,
            strategy=ReconstructionStrategy.AUTOMATA,
            reconstructed_state={"status": "completed"},
            confidence=0.9,
            agent_id="agent-1",
            thread_id="thread-1",
        )
        
        data = result.to_dict()
        
        assert data["success"] is True
        assert data["strategy"] == "automata"
        assert data["confidence"] == 0.9


# =============================================================================
# HybridReconstructor Tests
# =============================================================================


class TestHybridReconstructor:
    """Tests for HybridReconstructor."""

    def test_init_defaults(self):
        """Test default initialization."""
        reconstructor = HybridReconstructor()
        
        assert reconstructor.enable_automata is True
        assert reconstructor.enable_llm is True
        assert reconstructor.enable_peer_context is True

    def test_init_custom(self):
        """Test custom initialization."""
        reconstructor = HybridReconstructor(
            enable_automata=False,
            enable_llm=True,
            checkpoint_freshness=60,
        )
        
        assert reconstructor.enable_automata is False
        assert reconstructor.checkpoint_freshness == 60

    def test_train_automata(self, training_events):
        """Test training the automata."""
        reconstructor = HybridReconstructor()
        result = reconstructor.train_automata(training_events)
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_reconstruct_fresh_checkpoint(self, fresh_checkpoint):
        """Test reconstruction with fresh checkpoint."""
        reconstructor = HybridReconstructor()
        
        result = await reconstructor.reconstruct(
            agent_id="agent-1",
            thread_id="thread-1",
            checkpoint=fresh_checkpoint,
        )
        
        assert result.success is True
        assert result.strategy == ReconstructionStrategy.CHECKPOINT
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_reconstruct_automata(self, old_checkpoint, sample_events, training_events):
        """Test automata-based reconstruction."""
        reconstructor = HybridReconstructor(
            enable_llm=False,  # Force automata-only
            automata_confidence_threshold=0.3,  # Low threshold for test
        )
        
        # Pre-train
        reconstructor.train_automata(training_events)
        
        result = await reconstructor.reconstruct(
            agent_id="agent-1",
            thread_id="thread-1",
            checkpoint=old_checkpoint,
            events_since_checkpoint=sample_events,
            all_events=training_events,
        )
        
        assert result.success is True
        assert result.strategy in [ReconstructionStrategy.AUTOMATA, ReconstructionStrategy.FALLBACK]

    @pytest.mark.asyncio
    async def test_reconstruct_fallback(self, old_checkpoint, sample_events):
        """Test fallback reconstruction."""
        reconstructor = HybridReconstructor(
            enable_automata=False,
            enable_llm=False,  # Force fallback
        )
        
        result = await reconstructor.reconstruct(
            agent_id="agent-1",
            thread_id="thread-1",
            checkpoint=old_checkpoint,
            events_since_checkpoint=sample_events,
        )
        
        assert result.success is True
        assert result.strategy == ReconstructionStrategy.FALLBACK

    def test_reconstruct_sync(self, fresh_checkpoint):
        """Test synchronous reconstruction."""
        reconstructor = HybridReconstructor()
        
        result = reconstructor.reconstruct_sync(
            agent_id="agent-1",
            thread_id="thread-1",
            checkpoint=fresh_checkpoint,
        )
        
        assert result.success is True

    def test_is_checkpoint_fresh(self, fresh_checkpoint, old_checkpoint):
        """Test checkpoint freshness check."""
        reconstructor = HybridReconstructor(checkpoint_freshness=30)
        
        assert reconstructor._is_checkpoint_fresh(fresh_checkpoint) is True
        assert reconstructor._is_checkpoint_fresh(old_checkpoint) is False
        assert reconstructor._is_checkpoint_fresh(None) is False
        assert reconstructor._is_checkpoint_fresh({}) is False

    def test_fallback_reconstruction(self, old_checkpoint, sample_events):
        """Test fallback reconstruction logic."""
        reconstructor = HybridReconstructor()
        
        result = reconstructor._fallback_reconstruction(
            agent_id="agent-1",
            thread_id="thread-1",
            checkpoint=old_checkpoint,
            events=sample_events,
            reconstruction_time=10.0,
        )
        
        assert result.success is True
        assert result.strategy == ReconstructionStrategy.FALLBACK
        assert result.confidence == 0.3

    def test_fallback_without_checkpoint(self, sample_events):
        """Test fallback without checkpoint."""
        reconstructor = HybridReconstructor()
        
        result = reconstructor._fallback_reconstruction(
            agent_id="agent-1",
            thread_id="thread-1",
            checkpoint=None,
            events=sample_events,
            reconstruction_time=10.0,
        )
        
        assert result.success is True
        assert "agent_id" in result.reconstructed_state


# =============================================================================
# AutomataReconstructor Tests
# =============================================================================


class TestAutomataReconstructor:
    """Tests for AutomataReconstructor."""

    def test_init(self):
        """Test initialization."""
        reconstructor = AutomataReconstructor(min_events=30)
        
        assert reconstructor.min_events == 30
        assert reconstructor.is_trained is False

    def test_train(self, training_events):
        """Test training."""
        reconstructor = AutomataReconstructor()
        result = reconstructor.train(training_events)
        
        assert result.success is True
        assert reconstructor.is_trained is True

    def test_reconstruct_without_training(self):
        """Test reconstruction without training."""
        reconstructor = AutomataReconstructor()
        
        result = reconstructor.reconstruct(
            agent_id="agent-1",
            thread_id="thread-1",
        )
        
        assert result.success is False
        assert "Insufficient events" in result.error

    def test_reconstruct_with_training(self, training_events, sample_events, old_checkpoint):
        """Test reconstruction with training."""
        reconstructor = AutomataReconstructor(min_confidence=0.3)
        reconstructor.train(training_events)
        
        result = reconstructor.reconstruct(
            agent_id="agent-1",
            thread_id="thread-1",
            last_checkpoint=old_checkpoint,
            events_since_checkpoint=sample_events,
        )
        
        assert isinstance(result, AutomataReconstructionResult)
        assert result.events_used == len(sample_events)

    def test_model_accuracy(self, training_events):
        """Test model accuracy property."""
        reconstructor = AutomataReconstructor()
        
        assert reconstructor.model_accuracy == 0.0
        
        reconstructor.train(training_events)
        assert reconstructor.model_accuracy > 0.0


class TestReconstructWithAutomata:
    """Tests for reconstruct_with_automata convenience function."""

    def test_convenience_function(self, training_events, sample_events, old_checkpoint):
        """Test the convenience function."""
        result = reconstruct_with_automata(
            agent_id="agent-1",
            thread_id="thread-1",
            training_events=training_events,
            last_checkpoint=old_checkpoint,
            events_since_checkpoint=sample_events,
        )
        
        assert isinstance(result, AutomataReconstructionResult)


# =============================================================================
# Hybrid Reconstruct Function Tests
# =============================================================================


class TestHybridReconstructFunction:
    """Tests for hybrid_reconstruct convenience function."""

    @pytest.mark.asyncio
    async def test_hybrid_reconstruct(self, fresh_checkpoint):
        """Test the convenience function."""
        result = await hybrid_reconstruct(
            agent_id="agent-1",
            thread_id="thread-1",
            checkpoint=fresh_checkpoint,
        )
        
        assert isinstance(result, HybridReconstructionResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_hybrid_reconstruct_with_training(self, training_events, old_checkpoint, sample_events):
        """Test with training events."""
        result = await hybrid_reconstruct(
            agent_id="agent-1",
            thread_id="thread-1",
            checkpoint=old_checkpoint,
            events=sample_events,
            training_events=training_events,
        )
        
        assert result.success is True


# =============================================================================
# Strategy Selection Tests
# =============================================================================


class TestStrategySelection:
    """Tests for reconstruction strategy selection."""

    @pytest.mark.asyncio
    async def test_select_checkpoint_strategy(self, fresh_checkpoint):
        """Test that fresh checkpoint uses checkpoint strategy."""
        reconstructor = HybridReconstructor()
        
        result = await reconstructor.reconstruct(
            agent_id="agent-1",
            thread_id="thread-1",
            checkpoint=fresh_checkpoint,
        )
        
        assert result.strategy == ReconstructionStrategy.CHECKPOINT

    @pytest.mark.asyncio
    async def test_select_fallback_when_disabled(self, old_checkpoint, sample_events):
        """Test fallback when other strategies disabled."""
        reconstructor = HybridReconstructor(
            enable_automata=False,
            enable_llm=False,
        )
        
        result = await reconstructor.reconstruct(
            agent_id="agent-1",
            thread_id="thread-1",
            checkpoint=old_checkpoint,
            events_since_checkpoint=sample_events,
        )
        
        assert result.strategy == ReconstructionStrategy.FALLBACK

    def test_reconstruction_strategy_enum(self):
        """Test ReconstructionStrategy enum values."""
        assert ReconstructionStrategy.CHECKPOINT.value == "checkpoint"
        assert ReconstructionStrategy.AUTOMATA.value == "automata"
        assert ReconstructionStrategy.LLM.value == "llm"
        assert ReconstructionStrategy.HYBRID.value == "hybrid"
        assert ReconstructionStrategy.FALLBACK.value == "fallback"
