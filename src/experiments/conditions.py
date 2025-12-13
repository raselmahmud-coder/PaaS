"""Experimental conditions for thesis evaluation."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ConditionType(Enum):
    """Types of experimental conditions."""
    
    BASELINE = "baseline"
    RECONSTRUCTION = "reconstruction"
    FULL_SYSTEM = "full_system"
    # New comparison baselines
    SIMPLE_RETRY = "simple_retry"
    CHECKPOINT_ONLY = "checkpoint_only"
    AUTOMATA_ONLY = "automata_only"
    LLM_ONLY = "llm_only"
    # Real API validation
    REAL_API = "real_api"
    # Phase B - Industry standard baselines
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    SEMANTIC_ONLY = "semantic_only"
    # Phase C - Ablation study
    FULL_NO_SEMANTIC = "full_no_semantic"


@dataclass
class ConditionConfig:
    """Configuration for an experimental condition."""
    
    name: str
    condition_type: ConditionType
    resilience_enabled: bool
    semantic_protocol_enabled: bool
    automata_enabled: bool
    peer_context_enabled: bool
    description: str = ""
    # New fields for comparison baselines
    max_retries: int = 0  # For simple retry strategy
    use_checkpoint_restart: bool = False  # For checkpoint-only strategy
    llm_fallback_enabled: bool = True  # Whether LLM fallback is allowed
    is_real_api: bool = False  # Whether this condition uses real APIs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "condition_type": self.condition_type.value,
            "resilience_enabled": self.resilience_enabled,
            "semantic_protocol_enabled": self.semantic_protocol_enabled,
            "automata_enabled": self.automata_enabled,
            "peer_context_enabled": self.peer_context_enabled,
            "description": self.description,
            "max_retries": self.max_retries,
            "use_checkpoint_restart": self.use_checkpoint_restart,
            "llm_fallback_enabled": self.llm_fallback_enabled,
            "is_real_api": self.is_real_api,
        }


class ExperimentCondition(ABC):
    """Base class for experimental conditions.
    
    Each condition defines how the system behaves during experiments:
    - Whether resilience/reconstruction is enabled
    - Whether semantic protocol is used
    - Whether automata learning is used
    """
    
    def __init__(self):
        self.config = self._get_config()
    
    @abstractmethod
    def _get_config(self) -> ConditionConfig:
        """Get the condition configuration."""
        pass
    
    @property
    def name(self) -> str:
        """Get condition name."""
        return self.config.name
    
    @property
    def condition_type(self) -> ConditionType:
        """Get condition type."""
        return self.config.condition_type
    
    def should_attempt_recovery(self) -> bool:
        """Check if recovery should be attempted on failure."""
        return self.config.resilience_enabled
    
    def should_use_semantic_protocol(self) -> bool:
        """Check if semantic protocol should be used."""
        return self.config.semantic_protocol_enabled
    
    def should_use_automata(self) -> bool:
        """Check if automata learning should be used."""
        return self.config.automata_enabled
    
    def should_query_peers(self) -> bool:
        """Check if peer context should be queried."""
        return self.config.peer_context_enabled
    
    def get_reconstruction_strategy(self) -> str:
        """Get the reconstruction strategy for this condition."""
        if not self.config.resilience_enabled:
            # Check for semantic-only (ablation) - no recovery but semantic enabled
            if self.config.semantic_protocol_enabled:
                return "semantic_only"
            return "none"
        
        # Check for Phase B/C condition types first (explicit type matching)
        cond_type = self.config.condition_type
        
        if cond_type == ConditionType.EXPONENTIAL_BACKOFF:
            return "exponential_backoff"
        
        if cond_type == ConditionType.CIRCUIT_BREAKER:
            return "circuit_breaker"
        
        if cond_type == ConditionType.FULL_NO_SEMANTIC:
            return "hybrid"
        
        # Check for specific comparison strategies
        if self.config.max_retries > 0 and not self.config.automata_enabled:
            return "simple_retry"
        
        if self.config.use_checkpoint_restart and not self.config.automata_enabled:
            return "checkpoint"
        
        # Automata-only (no LLM fallback)
        if self.config.automata_enabled and not self.config.llm_fallback_enabled:
            return "automata_only"
        
        # LLM without peer context
        if not self.config.peer_context_enabled and not self.config.automata_enabled:
            return "llm_no_peer"
        
        # Original strategies
        if self.config.automata_enabled and self.config.semantic_protocol_enabled:
            return "hybrid"
        elif self.config.automata_enabled:
            return "automata"
        else:
            return "llm"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.config.to_dict()


class BaselineCondition(ExperimentCondition):
    """Baseline condition: No resilience enabled.
    
    When an agent fails, the entire workflow fails.
    This represents a traditional system without fault tolerance.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="baseline",
            condition_type=ConditionType.BASELINE,
            resilience_enabled=False,
            semantic_protocol_enabled=False,
            automata_enabled=False,
            peer_context_enabled=False,
            description="No resilience - agent failure causes workflow failure",
        )


class ReconstructionCondition(ExperimentCondition):
    """Reconstruction condition: LLM-based reconstruction only.
    
    When an agent fails, the system attempts to reconstruct state
    using LLM inference from checkpoints and event logs.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="reconstruction",
            condition_type=ConditionType.RECONSTRUCTION,
            resilience_enabled=True,
            semantic_protocol_enabled=False,
            automata_enabled=False,
            peer_context_enabled=True,  # LLM can use peer context
            description="LLM-based reconstruction with peer context",
        )


class FullSystemCondition(ExperimentCondition):
    """Full system condition: All features enabled.
    
    Uses semantic protocol for term alignment, automata learning
    for behavior prediction, and LLM for complex reasoning.
    This is the complete PaaS system.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="full_system",
            condition_type=ConditionType.FULL_SYSTEM,
            resilience_enabled=True,
            semantic_protocol_enabled=True,
            automata_enabled=True,
            peer_context_enabled=True,
            description="Full PaaS system with semantic protocol and automata",
        )


# =============================================================================
# Comparison Baseline Conditions (for related work comparison)
# =============================================================================


class SimpleRetryCondition(ExperimentCondition):
    """Simple retry condition: Retry N times without intelligent recovery.
    
    This represents a basic fault-tolerance strategy used in traditional
    distributed systems. When a failure occurs, the system simply retries
    the operation up to max_retries times without any state reconstruction.
    
    Comparable to: Basic retry policies in microservices, HTTP retry middleware.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="simple_retry",
            condition_type=ConditionType.SIMPLE_RETRY,
            resilience_enabled=True,  # Attempts recovery
            semantic_protocol_enabled=False,
            automata_enabled=False,
            peer_context_enabled=False,
            description="Simple retry (3 attempts) without state reconstruction",
            max_retries=3,
            use_checkpoint_restart=False,
            llm_fallback_enabled=False,
        )


class CheckpointOnlyCondition(ExperimentCondition):
    """Checkpoint-only condition: Restart from last checkpoint.
    
    When a failure occurs, the system loads the last checkpoint and
    restarts execution from that point. No LLM inference or automata
    learning is used to predict missing state.
    
    Comparable to: LangGraph native checkpointing, database transaction rollback.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="checkpoint_only",
            condition_type=ConditionType.CHECKPOINT_ONLY,
            resilience_enabled=True,
            semantic_protocol_enabled=False,
            automata_enabled=False,
            peer_context_enabled=False,
            description="Restart from last checkpoint without LLM inference",
            max_retries=0,
            use_checkpoint_restart=True,
            llm_fallback_enabled=False,
        )


class AutomataOnlyCondition(ExperimentCondition):
    """Automata-only condition: L* prediction without LLM fallback.
    
    Uses L* automata learning to predict agent behavior and reconstruct
    state, but does not fall back to LLM if automata prediction fails.
    This tests the effectiveness of formal methods alone.
    
    Comparable to: AALpy L* learning, formal verification approaches.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="automata_only",
            condition_type=ConditionType.AUTOMATA_ONLY,
            resilience_enabled=True,
            semantic_protocol_enabled=False,
            automata_enabled=True,
            peer_context_enabled=False,
            description="L* automata prediction only, no LLM fallback",
            max_retries=0,
            use_checkpoint_restart=False,
            llm_fallback_enabled=False,
        )


class LLMOnlyCondition(ExperimentCondition):
    """LLM-only condition: Pure LLM reconstruction without peer context.
    
    Uses LLM inference for state reconstruction but does not query
    peer agents for additional context. This isolates the contribution
    of peer context to the overall system performance.
    
    Comparable to: Basic GPT-4 inference for state recovery.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="llm_only",
            condition_type=ConditionType.LLM_ONLY,
            resilience_enabled=True,
            semantic_protocol_enabled=False,
            automata_enabled=False,
            peer_context_enabled=False,  # No peer context
            description="LLM reconstruction without peer context",
            max_retries=0,
            use_checkpoint_restart=False,
            llm_fallback_enabled=True,
        )


# =============================================================================
# Real API Validation Condition
# =============================================================================


class RealAPICondition(ExperimentCondition):
    """Real API condition: Full PaaS with real Shopify API.
    
    This condition uses the complete PaaS system but runs against
    a real Shopify development store, providing external validity
    evidence for the thesis.
    
    Features enabled:
    - Semantic protocol for term alignment
    - Automata learning for behavior prediction
    - LLM-based reconstruction with peer context
    - Real Shopify API operations
    
    Comparable to: Production-like e-commerce operations.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="real_api",
            condition_type=ConditionType.REAL_API,
            resilience_enabled=True,
            semantic_protocol_enabled=True,
            automata_enabled=True,
            peer_context_enabled=True,
            description="Full PaaS with real Shopify API operations",
            max_retries=0,
            use_checkpoint_restart=False,
            llm_fallback_enabled=True,
            is_real_api=True,
        )
    
    def is_real_api_condition(self) -> bool:
        """Check if this is a real API condition."""
        return True


# =============================================================================
# Phase B - Industry Standard Baselines
# =============================================================================


class ExponentialBackoffCondition(ExperimentCondition):
    """Exponential backoff retry: Industry-standard fault tolerance.
    
    Retries with exponential delays: 100ms, 200ms, 400ms, 800ms + jitter.
    No state reconstruction - just delayed retries.
    
    Literature: Google SRE Book, AWS Best Practices, Polly.NET.
    Expected success rate: ~40% (only works for transient failures)
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="exponential_backoff",
            condition_type=ConditionType.EXPONENTIAL_BACKOFF,
            resilience_enabled=True,
            semantic_protocol_enabled=False,
            automata_enabled=False,
            peer_context_enabled=False,
            description="Exponential backoff (4 retries, 100-800ms + jitter)",
            max_retries=4,
            use_checkpoint_restart=False,
            llm_fallback_enabled=False,
        )
    
    def get_retry_delays(self) -> List[float]:
        """Get retry delay sequence in seconds."""
        import random
        base = 0.1  # 100ms
        delays = []
        for i in range(4):
            delay = base * (2 ** i)  # 0.1, 0.2, 0.4, 0.8
            jitter = random.uniform(0, delay * 0.1)  # 10% jitter
            delays.append(delay + jitter)
        return delays


class CircuitBreakerState:
    """Track circuit breaker state across experiments.
    
    States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing)
    """
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    
    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0):
        self.state = self.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_failure_time: float = 0.0
    
    def record_failure(self) -> str:
        """Record a failure and return new state."""
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN
        
        return self.state
    
    def record_success(self) -> str:
        """Record success and reset if in half-open."""
        if self.state == self.HALF_OPEN:
            self.state = self.CLOSED
            self.failure_count = 0
        return self.state
    
    def can_execute(self) -> bool:
        """Check if request can proceed."""
        import time
        if self.state == self.CLOSED:
            return True
        
        if self.state == self.OPEN:
            # Check if cooldown passed
            if time.time() - self.last_failure_time > self.cooldown_seconds:
                self.state = self.HALF_OPEN
                return True  # Allow one test request
            return False
        
        # HALF_OPEN: allow request
        return True
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0


class CircuitBreakerCondition(ExperimentCondition):
    """Circuit breaker pattern: Fail-fast on repeated failures.
    
    After N consecutive failures, circuit "opens" and rejects requests
    for a cooldown period. Prevents cascade failures but doesn't recover state.
    
    Literature: Nygard (2018) "Release It!", Netflix Hystrix, Resilience4j.
    Expected success rate: ~45% (fast-fails prevent some damage but no recovery)
    """
    
    def __init__(self):
        super().__init__()
        self.circuit_state = CircuitBreakerState(failure_threshold=5, cooldown_seconds=30.0)
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="circuit_breaker",
            condition_type=ConditionType.CIRCUIT_BREAKER,
            resilience_enabled=True,
            semantic_protocol_enabled=False,
            automata_enabled=False,
            peer_context_enabled=False,
            description="Circuit breaker (5 failures -> 30s open)",
            max_retries=0,
            use_checkpoint_restart=False,
            llm_fallback_enabled=False,
        )
    
    def get_circuit_state(self) -> CircuitBreakerState:
        """Get the circuit breaker state for this condition."""
        return self.circuit_state


# =============================================================================
# Phase B - Ablation Study Condition
# =============================================================================


class SemanticOnlyCondition(ExperimentCondition):
    """Semantic-only condition: Ablation study for semantic protocol.
    
    Enables semantic handshake for term alignment but disables all
    recovery mechanisms. Tests whether semantic protocol alone
    prevents failures through proactive alignment (rather than recovering).
    
    Purpose: Isolate semantic protocol's preventive contribution.
    Expected result: Slightly better than baseline due to fewer misalignment failures.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="semantic_only",
            condition_type=ConditionType.SEMANTIC_ONLY,
            resilience_enabled=False,  # NO recovery!
            semantic_protocol_enabled=True,  # Only this enabled
            automata_enabled=False,
            peer_context_enabled=False,
            description="Semantic handshake only - prevents but doesn't recover",
            max_retries=0,
            use_checkpoint_restart=False,
            llm_fallback_enabled=False,
        )


# =============================================================================
# Phase C - Ablation Study Condition
# =============================================================================


class FullNoSemanticCondition(ExperimentCondition):
    """Full system WITHOUT semantic protocol - for ablation study.
    
    Enables all features except semantic handshake to isolate
    the semantic protocol's contribution to overall success rate.
    
    Purpose: Compare full_system vs full_no_semantic to measure
    semantic protocol's contribution.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="full_no_semantic",
            condition_type=ConditionType.FULL_NO_SEMANTIC,
            resilience_enabled=True,
            semantic_protocol_enabled=False,  # DISABLED for ablation
            automata_enabled=True,
            peer_context_enabled=True,
            description="Full system minus semantic protocol (ablation)",
            max_retries=0,
            use_checkpoint_restart=False,
            llm_fallback_enabled=True,
        )


# Registry of conditions
CONDITION_REGISTRY: Dict[str, Type[ExperimentCondition]] = {
    # Original conditions
    "baseline": BaselineCondition,
    "reconstruction": ReconstructionCondition,
    "full_system": FullSystemCondition,
    # Comparison baselines (for related work comparison)
    "simple_retry": SimpleRetryCondition,
    "checkpoint_only": CheckpointOnlyCondition,
    "automata_only": AutomataOnlyCondition,
    "llm_only": LLMOnlyCondition,
    # Real API validation
    "real_api": RealAPICondition,
    # Phase B - Industry standard baselines
    "exponential_backoff": ExponentialBackoffCondition,
    "circuit_breaker": CircuitBreakerCondition,
    # Phase B - Ablation study
    "semantic_only": SemanticOnlyCondition,
    # Phase C - Ablation study
    "full_no_semantic": FullNoSemanticCondition,
}


def get_condition(name: str) -> ExperimentCondition:
    """Get a condition by name.
    
    Args:
        name: Condition name (baseline, reconstruction, full_system).
        
    Returns:
        ExperimentCondition instance.
        
    Raises:
        KeyError: If condition not found.
    """
    if name not in CONDITION_REGISTRY:
        raise KeyError(
            f"Unknown condition: {name}. "
            f"Available: {list(CONDITION_REGISTRY.keys())}"
        )
    
    return CONDITION_REGISTRY[name]()


def list_conditions() -> List[str]:
    """List available condition names."""
    return list(CONDITION_REGISTRY.keys())


def get_all_conditions() -> List[ExperimentCondition]:
    """Get all condition instances."""
    return [cls() for cls in CONDITION_REGISTRY.values()]

