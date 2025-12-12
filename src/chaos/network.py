"""Network failure injection utilities for chaos engineering.

This module provides utilities to simulate real network issues like
delays, timeouts, and connection failures during API interactions.

These utilities work alongside the chaos decorators to provide
realistic network failure simulation for real-world API validation.
"""

import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import Optional

from src.chaos.config import get_chaos_config
from src.chaos.exceptions import AgentTimeoutException

logger = logging.getLogger(__name__)


@asynccontextmanager
async def inject_network_delay(
    delay_ms: int = 2000,
    jitter_ms: int = 500,
    probability: float = 1.0,
):
    """Context manager to inject network delay before/after operations.
    
    Usage:
        async with inject_network_delay(delay_ms=2000):
            response = await make_api_call()
    
    Args:
        delay_ms: Base delay in milliseconds
        jitter_ms: Random jitter to add (0 to jitter_ms)
        probability: Probability of applying the delay
    
    Yields:
        Control to the wrapped code
    """
    config = get_chaos_config()
    
    should_delay = config.enabled and random.random() < probability
    
    if should_delay:
        actual_delay_ms = delay_ms + random.randint(0, jitter_ms)
        delay_seconds = actual_delay_ms / 1000.0
        logger.warning(f"[NETWORK] Injecting {actual_delay_ms}ms delay")
        await asyncio.sleep(delay_seconds)
    
    try:
        yield
    finally:
        # Optional post-operation delay
        if should_delay and random.random() < 0.3:  # 30% chance of response delay
            response_delay = random.randint(100, 500) / 1000.0
            await asyncio.sleep(response_delay)


@asynccontextmanager
async def inject_connection_timeout(
    timeout_seconds: float = 30.0,
    probability: float = 0.1,
):
    """Context manager to simulate connection timeout.
    
    This blocks for the specified timeout and then raises an exception,
    simulating a network timeout scenario.
    
    Usage:
        async with inject_connection_timeout(timeout_seconds=30):
            response = await make_api_call()
    
    Args:
        timeout_seconds: How long to block before timing out
        probability: Probability of triggering the timeout
    
    Yields:
        Control to the wrapped code (never reached if timeout triggers)
        
    Raises:
        AgentTimeoutException: If timeout is triggered
    """
    config = get_chaos_config()
    
    if config.enabled and random.random() < probability:
        logger.warning(f"[NETWORK] Simulating connection timeout ({timeout_seconds}s)")
        await asyncio.sleep(timeout_seconds)
        raise AgentTimeoutException(
            f"Connection timed out after {timeout_seconds} seconds",
            timeout_seconds=timeout_seconds,
        )
    
    yield


@asynccontextmanager
async def inject_connection_reset(probability: float = 0.05):
    """Context manager to simulate connection reset.
    
    This raises a ConnectionResetError to simulate network disruption.
    
    Usage:
        async with inject_connection_reset():
            response = await make_api_call()
    
    Args:
        probability: Probability of triggering the reset
        
    Yields:
        Control to the wrapped code
        
    Raises:
        ConnectionResetError: If reset is triggered
    """
    config = get_chaos_config()
    
    if config.enabled and random.random() < probability:
        logger.warning("[NETWORK] Simulating connection reset")
        raise ConnectionResetError("Connection reset by peer (simulated)")
    
    yield


class NetworkConditioner:
    """Configurable network condition simulator.
    
    This class allows configuring various network conditions that
    can be applied to API operations during experiments.
    
    Attributes:
        latency_ms: Base latency to add (default: 0)
        jitter_ms: Random jitter range (default: 0)
        packet_loss_probability: Probability of packet loss (default: 0)
        timeout_probability: Probability of timeout (default: 0)
        timeout_seconds: Timeout duration (default: 30)
    """
    
    def __init__(
        self,
        latency_ms: int = 0,
        jitter_ms: int = 0,
        packet_loss_probability: float = 0.0,
        timeout_probability: float = 0.0,
        timeout_seconds: float = 30.0,
    ):
        """Initialize network conditioner.
        
        Args:
            latency_ms: Base latency to add in milliseconds
            jitter_ms: Random jitter to add (0 to jitter_ms)
            packet_loss_probability: Probability of simulating packet loss (0.0-1.0)
            timeout_probability: Probability of timeout (0.0-1.0)
            timeout_seconds: Timeout duration in seconds
        """
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms
        self.packet_loss_probability = packet_loss_probability
        self.timeout_probability = timeout_probability
        self.timeout_seconds = timeout_seconds
        self._enabled = True
    
    def enable(self):
        """Enable network conditioning."""
        self._enabled = True
    
    def disable(self):
        """Disable network conditioning."""
        self._enabled = False
    
    @asynccontextmanager
    async def condition(self):
        """Apply network conditions to wrapped operation.
        
        Usage:
            conditioner = NetworkConditioner(latency_ms=100, jitter_ms=50)
            async with conditioner.condition():
                response = await api_call()
        
        Yields:
            Control to the wrapped code
            
        Raises:
            TimeoutError: If timeout is triggered
            ConnectionError: If packet loss is triggered
        """
        if not self._enabled:
            yield
            return
        
        # Apply latency
        if self.latency_ms > 0:
            delay_ms = self.latency_ms
            if self.jitter_ms > 0:
                delay_ms += random.randint(0, self.jitter_ms)
            await asyncio.sleep(delay_ms / 1000.0)
        
        # Check for timeout
        if random.random() < self.timeout_probability:
            logger.warning(f"[NETWORK] Timeout triggered ({self.timeout_seconds}s)")
            await asyncio.sleep(self.timeout_seconds)
            raise TimeoutError(f"Network timeout after {self.timeout_seconds}s")
        
        # Check for packet loss
        if random.random() < self.packet_loss_probability:
            logger.warning("[NETWORK] Packet loss triggered")
            raise ConnectionError("Packet loss (simulated)")
        
        yield


# =============================================================================
# Preset Network Conditions
# =============================================================================

def get_poor_network_conditioner() -> NetworkConditioner:
    """Get a network conditioner simulating poor network conditions.
    
    Returns:
        NetworkConditioner with high latency and some packet loss
    """
    return NetworkConditioner(
        latency_ms=500,
        jitter_ms=200,
        packet_loss_probability=0.05,
        timeout_probability=0.02,
        timeout_seconds=30.0,
    )


def get_unstable_network_conditioner() -> NetworkConditioner:
    """Get a network conditioner simulating unstable network.
    
    Returns:
        NetworkConditioner with moderate latency and higher failure rate
    """
    return NetworkConditioner(
        latency_ms=200,
        jitter_ms=500,  # High variance
        packet_loss_probability=0.1,
        timeout_probability=0.05,
        timeout_seconds=15.0,
    )


def get_mobile_network_conditioner() -> NetworkConditioner:
    """Get a network conditioner simulating mobile network (3G/4G).
    
    Returns:
        NetworkConditioner with moderate latency and occasional issues
    """
    return NetworkConditioner(
        latency_ms=150,
        jitter_ms=100,
        packet_loss_probability=0.02,
        timeout_probability=0.01,
        timeout_seconds=60.0,
    )

