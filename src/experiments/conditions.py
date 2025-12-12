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
            return "none"
        
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

