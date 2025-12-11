"""Chaos engineering configuration via environment variables."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChaosConfig:
    """Configuration for chaos engineering framework.
    
    All settings can be overridden via environment variables.
    """
    
    # Global chaos enable/disable
    enabled: bool = field(
        default_factory=lambda: os.getenv("CHAOS_ENABLED", "false").lower() == "true"
    )
    
    # Crash fault settings
    crash_probability: float = field(
        default_factory=lambda: float(os.getenv("CHAOS_CRASH_PROBABILITY", "0.1"))
    )
    
    # Delay fault settings
    delay_enabled: bool = field(
        default_factory=lambda: os.getenv("CHAOS_DELAY_ENABLED", "true").lower() == "true"
    )
    delay_ms: int = field(
        default_factory=lambda: int(os.getenv("CHAOS_DELAY_MS", "500"))
    )
    delay_probability: float = field(
        default_factory=lambda: float(os.getenv("CHAOS_DELAY_PROBABILITY", "0.2"))
    )
    
    # Timeout fault settings
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("CHAOS_TIMEOUT_SECONDS", "30"))
    )
    timeout_probability: float = field(
        default_factory=lambda: float(os.getenv("CHAOS_TIMEOUT_PROBABILITY", "0.1"))
    )
    
    # Hallucination fault settings
    hallucination_probability: float = field(
        default_factory=lambda: float(os.getenv("CHAOS_HALLUCINATION_PROBABILITY", "0.1"))
    )
    hallucination_responses: list = field(
        default_factory=lambda: [
            "I don't know what to do.",
            '{"error": "confused"}',
            "The product is a banana.",  # Wrong output
            "",  # Empty response
            "null",
        ]
    )
    
    # Message corruption settings
    corruption_probability: float = field(
        default_factory=lambda: float(os.getenv("CHAOS_CORRUPTION_PROBABILITY", "0.1"))
    )
    corruption_fields: list = field(
        default_factory=lambda: ["payload", "sender", "receiver", "message_type"]
    )
    
    # Seed for reproducibility (optional)
    random_seed: Optional[int] = field(
        default_factory=lambda: int(os.getenv("CHAOS_RANDOM_SEED")) 
        if os.getenv("CHAOS_RANDOM_SEED") else None
    )
    
    def __post_init__(self):
        """Validate configuration values."""
        # Clamp probabilities to [0, 1]
        self.crash_probability = max(0.0, min(1.0, self.crash_probability))
        self.delay_probability = max(0.0, min(1.0, self.delay_probability))
        self.timeout_probability = max(0.0, min(1.0, self.timeout_probability))
        self.hallucination_probability = max(0.0, min(1.0, self.hallucination_probability))
        self.corruption_probability = max(0.0, min(1.0, self.corruption_probability))
        
        # Ensure positive values
        self.delay_ms = max(0, self.delay_ms)
        self.timeout_seconds = max(0.0, self.timeout_seconds)


# Global config instance
_chaos_config: Optional[ChaosConfig] = None


def get_chaos_config() -> ChaosConfig:
    """Get or create the global chaos configuration."""
    global _chaos_config
    if _chaos_config is None:
        _chaos_config = ChaosConfig()
    return _chaos_config


def set_chaos_config(config: ChaosConfig) -> None:
    """Set custom chaos configuration (for testing)."""
    global _chaos_config
    _chaos_config = config


def reset_chaos_config() -> None:
    """Reset chaos configuration to reload from environment."""
    global _chaos_config
    _chaos_config = None


def enable_chaos() -> None:
    """Enable chaos engineering globally."""
    config = get_chaos_config()
    config.enabled = True


def disable_chaos() -> None:
    """Disable chaos engineering globally."""
    config = get_chaos_config()
    config.enabled = False

