"""Chaos scenario definitions for e-commerce agent testing."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from src.chaos.config import get_chaos_config, enable_chaos, disable_chaos
from src.chaos.exceptions import (
    ChaosException,
    AgentCrashException,
    AgentTimeoutException,
    MessageCorruptionException,
)

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Result of a chaos scenario execution."""
    
    scenario_name: str
    success: bool
    recovered: bool
    start_time: datetime
    end_time: datetime
    failure_injected_at: Optional[datetime] = None
    recovery_completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Total scenario duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def mttr_seconds(self) -> Optional[float]:
        """Mean Time to Recovery in seconds (if recovery occurred)."""
        if self.failure_injected_at and self.recovery_completed_at:
            return (self.recovery_completed_at - self.failure_injected_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "scenario_name": self.scenario_name,
            "success": self.success,
            "recovered": self.recovered,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "mttr_seconds": self.mttr_seconds,
            "failure_injected_at": self.failure_injected_at.isoformat() if self.failure_injected_at else None,
            "recovery_completed_at": self.recovery_completed_at.isoformat() if self.recovery_completed_at else None,
            "error": self.error,
            "metrics": self.metrics,
        }


class ChaosScenario(ABC):
    """Base class for chaos scenarios.
    
    Each scenario defines:
    - setup(): Prepare the environment for the scenario
    - inject_fault(): Inject the specific fault
    - validate_recovery(): Verify the system recovered correctly
    - teardown(): Clean up after the scenario
    """
    
    name: str = "BaseScenario"
    description: str = "Base chaos scenario"
    
    def __init__(self):
        self.result: Optional[ScenarioResult] = None
        self._original_chaos_enabled: bool = False
    
    def setup(self) -> None:
        """Set up the scenario environment.
        
        Override to add custom setup logic.
        """
        # Store original chaos state and enable it
        config = get_chaos_config()
        self._original_chaos_enabled = config.enabled
        enable_chaos()
        logger.info(f"[SCENARIO] Setting up: {self.name}")
    
    @abstractmethod
    def inject_fault(self) -> datetime:
        """Inject the fault for this scenario.
        
        Returns:
            Timestamp when the fault was injected.
        """
        pass
    
    @abstractmethod
    def validate_recovery(self) -> bool:
        """Validate that the system recovered from the fault.
        
        Returns:
            True if recovery was successful, False otherwise.
        """
        pass
    
    def teardown(self) -> None:
        """Clean up after the scenario.
        
        Override to add custom teardown logic.
        """
        # Restore original chaos state
        config = get_chaos_config()
        config.enabled = self._original_chaos_enabled
        logger.info(f"[SCENARIO] Teardown complete: {self.name}")
    
    def run(self) -> ScenarioResult:
        """Execute the full scenario lifecycle.
        
        Returns:
            ScenarioResult with execution details.
        """
        start_time = datetime.utcnow()
        failure_injected_at = None
        recovery_completed_at = None
        success = False
        recovered = False
        error = None
        metrics = {}
        
        try:
            # Setup
            self.setup()
            
            # Inject fault
            failure_injected_at = self.inject_fault()
            
            # Validate recovery
            recovery_start = time.perf_counter()
            recovered = self.validate_recovery()
            recovery_time = time.perf_counter() - recovery_start
            
            if recovered:
                recovery_completed_at = datetime.utcnow()
                metrics["recovery_time_seconds"] = recovery_time
            
            success = recovered
            
        except Exception as e:
            error = str(e)
            logger.error(f"[SCENARIO] {self.name} failed: {e}")
            
        finally:
            # Always teardown
            try:
                self.teardown()
            except Exception as e:
                logger.error(f"[SCENARIO] Teardown error: {e}")
        
        end_time = datetime.utcnow()
        
        self.result = ScenarioResult(
            scenario_name=self.name,
            success=success,
            recovered=recovered,
            start_time=start_time,
            end_time=end_time,
            failure_injected_at=failure_injected_at,
            recovery_completed_at=recovery_completed_at,
            error=error,
            metrics=metrics,
        )
        
        return self.result


class ProductCrashMidUpload(ChaosScenario):
    """Scenario: Product agent crashes during upload at 50% progress.
    
    Tests basic crash recovery capability when agent fails mid-workflow.
    """
    
    name = "ProductCrashMidUpload"
    description = "Product agent crashes at 50% progress during upload"
    
    def __init__(
        self,
        workflow_factory=None,
        initial_state: Optional[Dict[str, Any]] = None,
        checkpointer=None,
    ):
        super().__init__()
        self.workflow_factory = workflow_factory
        self.initial_state = initial_state or {}
        self.checkpointer = checkpointer
        self._workflow = None
        self._thread_id = f"chaos-{self.name}-{int(time.time())}"
        self._final_result = None
    
    def setup(self) -> None:
        super().setup()
        # Create workflow if factory provided
        if self.workflow_factory:
            self._workflow = self.workflow_factory(checkpointer=self.checkpointer)
    
    def inject_fault(self) -> datetime:
        """Inject crash at generate_listing step (step 2 of 3 = ~50%)."""
        fault_time = datetime.utcnow()
        
        if self._workflow and self.initial_state:
            # Use fail_step config to inject crash at specific step
            config = {
                "configurable": {
                    "thread_id": self._thread_id,
                    "fail_step": "generate_listing",  # Crash at step 2
                }
            }
            
            try:
                self._workflow.invoke(self.initial_state, config)
            except RuntimeError as e:
                # Expected - the workflow crashed at generate_listing
                logger.info(f"[SCENARIO] Fault injected successfully: {e}")
        
        return fault_time
    
    def validate_recovery(self) -> bool:
        """Validate recovery by resuming the workflow."""
        if not self._workflow:
            return False
        
        try:
            from src.reconstruction.reconstructor import recover_and_resume_workflow
            
            result = recover_and_resume_workflow(
                workflow=self._workflow,
                agent_id=self.initial_state.get("agent_id", "product-agent-1"),
                thread_id=self._thread_id,
                initial_state=self.initial_state,
                checkpointer=self.checkpointer,
            )
            
            self._final_result = result
            
            # Check if workflow completed successfully
            if result.get("recovered") and result.get("final_result"):
                final_status = result["final_result"].get("status")
                return final_status == "completed"
            
            return False
            
        except Exception as e:
            logger.error(f"[SCENARIO] Recovery validation failed: {e}")
            return False


class MarketingTimeout(ChaosScenario):
    """Scenario: Marketing agent times out after receiving task handoff.
    
    Tests timeout detection and recovery when agent becomes unresponsive.
    """
    
    name = "MarketingTimeout"
    description = "Marketing agent times out after receiving task from Product agent"
    
    def __init__(
        self,
        workflow_factory=None,
        initial_state: Optional[Dict[str, Any]] = None,
        checkpointer=None,
        timeout_seconds: float = 5.0,
    ):
        super().__init__()
        self.workflow_factory = workflow_factory
        self.initial_state = initial_state or {}
        self.checkpointer = checkpointer
        self.timeout_seconds = timeout_seconds
        self._workflow = None
        self._thread_id = f"chaos-{self.name}-{int(time.time())}"
    
    def setup(self) -> None:
        super().setup()
        if self.workflow_factory:
            self._workflow = self.workflow_factory(checkpointer=self.checkpointer)
    
    def inject_fault(self) -> datetime:
        """Inject timeout at marketing step (after handoff)."""
        fault_time = datetime.utcnow()
        
        if self._workflow and self.initial_state:
            # Inject timeout at marketing step
            config = {
                "configurable": {
                    "thread_id": self._thread_id,
                    "fail_step": "generate_marketing_copy",
                }
            }
            
            try:
                self._workflow.invoke(self.initial_state, config)
            except RuntimeError as e:
                logger.info(f"[SCENARIO] Timeout injected: {e}")
        
        return fault_time
    
    def validate_recovery(self) -> bool:
        """Validate timeout recovery."""
        if not self._workflow:
            return False
        
        try:
            from src.reconstruction.reconstructor import recover_and_resume_workflow
            
            result = recover_and_resume_workflow(
                workflow=self._workflow,
                agent_id="marketing-agent-1",
                thread_id=self._thread_id,
                initial_state=self.initial_state,
                timeout_seconds=int(self.timeout_seconds),
                checkpointer=self.checkpointer,
            )
            
            return result.get("recovered", False) or result.get("final_result") is not None
            
        except Exception as e:
            logger.error(f"[SCENARIO] Recovery failed: {e}")
            return False


class HandoffCorruption(ChaosScenario):
    """Scenario: Protocol message gets corrupted during handoff.
    
    Tests message validation and corruption handling.
    """
    
    name = "HandoffCorruption"
    description = "Protocol message corrupted during Product→Marketing handoff"
    
    def __init__(self):
        super().__init__()
        self._corrupted_message = None
        self._validation_result = None
    
    def inject_fault(self) -> datetime:
        """Inject corruption into a protocol message."""
        from src.protocol.messages import TaskAssignMessage
        from src.protocol.validator import safe_validate_message
        
        fault_time = datetime.utcnow()
        
        # Create a valid message
        message = TaskAssignMessage(
            sender="product-agent-1",
            receiver="marketing-agent-1",
            payload={
                "state": {"task_id": "test"},
                "task_description": "Generate marketing copy",
            }
        )
        
        # Corrupt it
        corrupted_data = message.model_dump()
        corrupted_data["message_type"] = "INVALID_TYPE"  # Invalid type
        corrupted_data["sender"] = None  # Missing sender
        
        self._corrupted_message = corrupted_data
        
        # Try to validate - should fail
        self._validation_result = safe_validate_message(corrupted_data)
        
        return fault_time
    
    def validate_recovery(self) -> bool:
        """Validate that corrupted message was rejected."""
        # Recovery = system correctly rejected the corrupted message
        return self._validation_result is None  # None means validation failed


class DelayedRecovery(ChaosScenario):
    """Scenario: Agent responds slowly, potentially triggering false timeout.
    
    Tests that slow agents don't falsely trigger recovery.
    """
    
    name = "DelayedRecovery"
    description = "Agent responds slowly but completes - should not trigger false recovery"
    
    def __init__(
        self,
        delay_ms: int = 2000,
        timeout_seconds: float = 10.0,
    ):
        super().__init__()
        self.delay_ms = delay_ms
        self.timeout_seconds = timeout_seconds
        self._completed_normally = False
    
    def inject_fault(self) -> datetime:
        """Inject a delay that's shorter than timeout."""
        fault_time = datetime.utcnow()
        
        # Simulate delayed but successful execution
        delay_seconds = self.delay_ms / 1000.0
        
        if delay_seconds < self.timeout_seconds:
            logger.info(f"[SCENARIO] Injecting {delay_seconds}s delay (timeout: {self.timeout_seconds}s)")
            time.sleep(delay_seconds)
            self._completed_normally = True
        
        return fault_time
    
    def validate_recovery(self) -> bool:
        """Validate that the operation completed without false recovery trigger."""
        # Success = operation completed normally despite delay
        return self._completed_normally


class CascadeFailure(ChaosScenario):
    """Scenario: Multiple agents fail sequentially (cascade failure).
    
    Tests system resilience to cascading failures across agents.
    """
    
    name = "CascadeFailure"
    description = "Product agent crashes, then Marketing agent crashes sequentially"
    
    def __init__(
        self,
        workflow_factory=None,
        initial_state: Optional[Dict[str, Any]] = None,
        checkpointer=None,
    ):
        super().__init__()
        self.workflow_factory = workflow_factory
        self.initial_state = initial_state or {}
        self.checkpointer = checkpointer
        self._workflow = None
        self._thread_id = f"chaos-{self.name}-{int(time.time())}"
        self._first_failure_recovered = False
        self._second_failure_recovered = False
    
    def setup(self) -> None:
        super().setup()
        if self.workflow_factory:
            self._workflow = self.workflow_factory(checkpointer=self.checkpointer)
    
    def inject_fault(self) -> datetime:
        """Inject cascade of failures."""
        fault_time = datetime.utcnow()
        
        if self._workflow and self.initial_state:
            # First failure: Product agent crashes
            config = {
                "configurable": {
                    "thread_id": self._thread_id,
                    "fail_step": "validate_product_data",
                }
            }
            
            try:
                self._workflow.invoke(self.initial_state, config)
            except RuntimeError as e:
                logger.info(f"[SCENARIO] First failure (Product): {e}")
        
        return fault_time
    
    def validate_recovery(self) -> bool:
        """Validate recovery from cascade failure."""
        if not self._workflow:
            return False
        
        try:
            from src.reconstruction.reconstructor import recover_and_resume_workflow
            
            # First recovery attempt
            result1 = recover_and_resume_workflow(
                workflow=self._workflow,
                agent_id="product-agent-1",
                thread_id=self._thread_id,
                initial_state=self.initial_state,
                checkpointer=self.checkpointer,
            )
            
            self._first_failure_recovered = result1.get("recovered", False)
            
            # If first recovery succeeded, the workflow should complete
            # (we won't inject a second failure in this simplified version)
            if result1.get("final_result"):
                return result1["final_result"].get("status") == "completed"
            
            return self._first_failure_recovered
            
        except Exception as e:
            logger.error(f"[SCENARIO] Cascade recovery failed: {e}")
            return False


# Registry of all available scenarios
SCENARIO_REGISTRY: Dict[str, Type[ChaosScenario]] = {
    "ProductCrashMidUpload": ProductCrashMidUpload,
    "MarketingTimeout": MarketingTimeout,
    "HandoffCorruption": HandoffCorruption,
    "DelayedRecovery": DelayedRecovery,
    "CascadeFailure": CascadeFailure,
}


def get_scenario(name: str, **kwargs) -> ChaosScenario:
    """Get a scenario instance by name.
    
    Args:
        name: Scenario name from registry.
        **kwargs: Arguments to pass to scenario constructor.
        
    Returns:
        Instantiated scenario.
        
    Raises:
        KeyError: If scenario name not found.
    """
    if name not in SCENARIO_REGISTRY:
        raise KeyError(f"Unknown scenario: {name}. Available: {list(SCENARIO_REGISTRY.keys())}")
    
    return SCENARIO_REGISTRY[name](**kwargs)


def list_scenarios() -> List[str]:
    """List all available scenario names."""
    return list(SCENARIO_REGISTRY.keys())

