"""Fault injection decorators for chaos engineering."""

import random
import time
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from src.chaos.config import get_chaos_config
from src.chaos.exceptions import (
    AgentCrashException,
    AgentTimeoutException,
    MessageCorruptionException,
    HallucinationException,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def chaos_enabled(func: F) -> F:
    """Decorator that only executes if chaos is globally enabled.
    
    Use this to wrap other decorators or functions that should only
    run when chaos testing is active.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        config = get_chaos_config()
        if not config.enabled:
            return func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper


def inject_crash(
    probability: Optional[float] = None,
    message: str = "Simulated agent crash",
) -> Callable[[F], F]:
    """Decorator to randomly inject agent crashes.
    
    Args:
        probability: Probability of crash (0.0 to 1.0). Uses config default if None.
        message: Custom crash message.
        
    Returns:
        Decorated function that may raise AgentCrashException.
        
    Example:
        @inject_crash(probability=0.2)
        def agent_step(state):
            # 20% chance of crashing before execution
            return process(state)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_chaos_config()
            
            # Skip if chaos is disabled
            if not config.enabled:
                return func(*args, **kwargs)
            
            # Use provided probability or config default
            crash_prob = probability if probability is not None else config.crash_probability
            
            # Roll the dice
            if random.random() < crash_prob:
                logger.warning(f"[CHAOS] Injecting crash in {func.__name__}: {message}")
                raise AgentCrashException(message)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def inject_delay(
    delay_ms: Optional[int] = None,
    probability: Optional[float] = None,
    jitter_ms: int = 0,
) -> Callable[[F], F]:
    """Decorator to randomly inject delays/latency.
    
    Args:
        delay_ms: Delay in milliseconds. Uses config default if None.
        probability: Probability of delay (0.0 to 1.0). Uses config default if None.
        jitter_ms: Random jitter to add to delay (0 to jitter_ms).
        
    Returns:
        Decorated function that may execute with added latency.
        
    Example:
        @inject_delay(delay_ms=1000, probability=0.3)
        def slow_operation(data):
            # 30% chance of 1 second delay
            return process(data)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_chaos_config()
            
            # Skip if chaos is disabled
            if not config.enabled:
                return func(*args, **kwargs)
            
            # Use provided values or config defaults
            actual_delay_ms = delay_ms if delay_ms is not None else config.delay_ms
            delay_prob = probability if probability is not None else config.delay_probability
            
            # Roll the dice
            if random.random() < delay_prob:
                # Add jitter if specified
                actual_delay = actual_delay_ms
                if jitter_ms > 0:
                    actual_delay += random.randint(0, jitter_ms)
                
                delay_seconds = actual_delay / 1000.0
                logger.warning(f"[CHAOS] Injecting {actual_delay}ms delay in {func.__name__}")
                time.sleep(delay_seconds)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def inject_timeout(
    timeout_seconds: Optional[float] = None,
    probability: Optional[float] = None,
    block: bool = True,
) -> Callable[[F], F]:
    """Decorator to simulate agent timeouts.
    
    Args:
        timeout_seconds: How long to block before timing out. Uses config default if None.
        probability: Probability of timeout (0.0 to 1.0). Uses config default if None.
        block: If True, blocks for timeout_seconds then raises. If False, raises immediately.
        
    Returns:
        Decorated function that may raise AgentTimeoutException.
        
    Example:
        @inject_timeout(timeout_seconds=30, probability=0.1)
        def agent_step(state):
            # 10% chance of 30-second timeout
            return process(state)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_chaos_config()
            
            # Skip if chaos is disabled
            if not config.enabled:
                return func(*args, **kwargs)
            
            # Use provided values or config defaults
            actual_timeout = timeout_seconds if timeout_seconds is not None else config.timeout_seconds
            timeout_prob = probability if probability is not None else config.timeout_probability
            
            # Roll the dice
            if random.random() < timeout_prob:
                logger.warning(f"[CHAOS] Injecting timeout in {func.__name__}: {actual_timeout}s")
                
                if block:
                    # Block for the timeout duration to simulate unresponsive agent
                    time.sleep(actual_timeout)
                
                raise AgentTimeoutException(
                    f"Agent timed out after {actual_timeout} seconds",
                    timeout_seconds=actual_timeout
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def inject_hallucination(
    probability: Optional[float] = None,
    responses: Optional[List[str]] = None,
    return_instead: bool = False,
) -> Callable[[F], F]:
    """Decorator to simulate LLM hallucinations/wrong outputs.
    
    Args:
        probability: Probability of hallucination (0.0 to 1.0). Uses config default if None.
        responses: List of fake responses to return. Uses config default if None.
        return_instead: If True, returns fake response. If False, raises exception.
        
    Returns:
        Decorated function that may return wrong output or raise HallucinationException.
        
    Example:
        @inject_hallucination(probability=0.15)
        def llm_call(prompt):
            # 15% chance of hallucinated response
            return real_llm_call(prompt)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_chaos_config()
            
            # Skip if chaos is disabled
            if not config.enabled:
                return func(*args, **kwargs)
            
            # Use provided values or config defaults
            hallucination_prob = probability if probability is not None else config.hallucination_probability
            fake_responses = responses if responses is not None else config.hallucination_responses
            
            # Roll the dice
            if random.random() < hallucination_prob:
                fake_response = random.choice(fake_responses) if fake_responses else "Hallucinated response"
                logger.warning(f"[CHAOS] Injecting hallucination in {func.__name__}: {fake_response[:50]}...")
                
                if return_instead:
                    # Return a mock object with the fake response
                    class FakeResponse:
                        content = fake_response
                    return FakeResponse()
                else:
                    raise HallucinationException(f"LLM hallucinated: {fake_response}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def inject_message_corruption(
    probability: Optional[float] = None,
    fields: Optional[List[str]] = None,
    corruption_type: str = "scramble",
) -> Callable[[F], F]:
    """Decorator to simulate message/data corruption.
    
    Args:
        probability: Probability of corruption (0.0 to 1.0). Uses config default if None.
        fields: List of field names that could be corrupted. Uses config default if None.
        corruption_type: Type of corruption - "scramble", "nullify", or "randomize".
        
    Returns:
        Decorated function that may corrupt its input/output.
        
    Example:
        @inject_message_corruption(probability=0.1, fields=["payload"])
        def send_message(message):
            # 10% chance of payload corruption
            return transmit(message)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_chaos_config()
            
            # Skip if chaos is disabled
            if not config.enabled:
                return func(*args, **kwargs)
            
            # Use provided values or config defaults
            corruption_prob = probability if probability is not None else config.corruption_probability
            target_fields = fields if fields is not None else config.corruption_fields
            
            # Roll the dice
            if random.random() < corruption_prob:
                corrupted_field = random.choice(target_fields) if target_fields else "unknown"
                logger.warning(f"[CHAOS] Injecting corruption in {func.__name__}: field={corrupted_field}")
                
                # Corrupt the first dict argument if found
                corrupted_args = list(args)
                for i, arg in enumerate(corrupted_args):
                    if isinstance(arg, dict) and corrupted_field in arg:
                        corrupted_args[i] = _corrupt_dict(arg, corrupted_field, corruption_type)
                        break
                
                # Also check kwargs
                corrupted_kwargs = dict(kwargs)
                for key, value in corrupted_kwargs.items():
                    if isinstance(value, dict) and corrupted_field in value:
                        corrupted_kwargs[key] = _corrupt_dict(value, corrupted_field, corruption_type)
                        break
                
                # Execute with corrupted data
                try:
                    return func(*corrupted_args, **corrupted_kwargs)
                except Exception as e:
                    # Wrap any exception as corruption exception
                    raise MessageCorruptionException(
                        f"Message corruption caused error: {e}",
                        corrupted_field=corrupted_field
                    ) from e
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _corrupt_dict(data: Dict[str, Any], field: str, corruption_type: str) -> Dict[str, Any]:
    """Corrupt a specific field in a dictionary.
    
    Args:
        data: Dictionary to corrupt.
        field: Field name to corrupt.
        corruption_type: Type of corruption.
        
    Returns:
        Corrupted dictionary (copy).
    """
    corrupted = dict(data)
    
    if field not in corrupted:
        return corrupted
    
    original_value = corrupted[field]
    
    if corruption_type == "nullify":
        corrupted[field] = None
    elif corruption_type == "randomize":
        if isinstance(original_value, str):
            corrupted[field] = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=10))
        elif isinstance(original_value, int):
            corrupted[field] = random.randint(-1000, 1000)
        elif isinstance(original_value, float):
            corrupted[field] = random.random() * 1000
        elif isinstance(original_value, dict):
            corrupted[field] = {"corrupted": True}
        elif isinstance(original_value, list):
            corrupted[field] = ["corrupted"]
        else:
            corrupted[field] = None
    else:  # scramble
        if isinstance(original_value, str):
            chars = list(original_value)
            random.shuffle(chars)
            corrupted[field] = "".join(chars)
        elif isinstance(original_value, dict):
            # Swap keys
            keys = list(original_value.keys())
            if len(keys) >= 2:
                random.shuffle(keys)
                corrupted[field] = {keys[i]: original_value[k] for i, k in enumerate(original_value.keys())}
        else:
            corrupted[field] = None
    
    return corrupted


# Convenience function to apply multiple faults at once
def with_chaos(
    crash_prob: float = 0.0,
    delay_prob: float = 0.0,
    delay_ms: int = 500,
    timeout_prob: float = 0.0,
    timeout_seconds: float = 30.0,
) -> Callable[[F], F]:
    """Convenience decorator to apply multiple chaos faults at once.
    
    Args:
        crash_prob: Probability of crash.
        delay_prob: Probability of delay.
        delay_ms: Delay duration in milliseconds.
        timeout_prob: Probability of timeout.
        timeout_seconds: Timeout duration in seconds.
        
    Returns:
        Decorated function with multiple potential faults.
        
    Example:
        @with_chaos(crash_prob=0.1, delay_prob=0.2)
        def risky_operation(data):
            return process(data)
    """
    def decorator(func: F) -> F:
        # Apply decorators in reverse order (crash last so it's checked first)
        decorated = func
        
        if timeout_prob > 0:
            decorated = inject_timeout(
                timeout_seconds=timeout_seconds, 
                probability=timeout_prob
            )(decorated)
        
        if delay_prob > 0:
            decorated = inject_delay(
                delay_ms=delay_ms, 
                probability=delay_prob
            )(decorated)
        
        if crash_prob > 0:
            decorated = inject_crash(probability=crash_prob)(decorated)
        
        return decorated
    return decorator

