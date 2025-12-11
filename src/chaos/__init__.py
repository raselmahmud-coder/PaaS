"""Chaos engineering framework for agent resilience testing."""

from src.chaos.exceptions import (
    ChaosException,
    AgentCrashException,
    AgentTimeoutException,
    MessageCorruptionException,
    HallucinationException,
)
from src.chaos.config import ChaosConfig, get_chaos_config
from src.chaos.decorators import (
    inject_crash,
    inject_delay,
    inject_timeout,
    inject_hallucination,
    inject_message_corruption,
    chaos_enabled,
)
from src.chaos.scenarios import (
    ChaosScenario,
    ProductCrashMidUpload,
    MarketingTimeout,
    HandoffCorruption,
    DelayedRecovery,
    CascadeFailure,
)
from src.chaos.runner import ChaosRunner
from src.chaos.metrics import ResilienceMetrics
from src.chaos.export import MetricsExporter

__all__ = [
    # Exceptions
    "ChaosException",
    "AgentCrashException",
    "AgentTimeoutException",
    "MessageCorruptionException",
    "HallucinationException",
    # Config
    "ChaosConfig",
    "get_chaos_config",
    # Decorators
    "inject_crash",
    "inject_delay",
    "inject_timeout",
    "inject_hallucination",
    "inject_message_corruption",
    "chaos_enabled",
    # Scenarios
    "ChaosScenario",
    "ProductCrashMidUpload",
    "MarketingTimeout",
    "HandoffCorruption",
    "DelayedRecovery",
    "CascadeFailure",
    # Runner
    "ChaosRunner",
    # Metrics
    "ResilienceMetrics",
    "MetricsExporter",
]

