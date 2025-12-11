"""Chaos scenario runner for executing and managing chaos tests."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

from src.chaos.scenarios import (
    ChaosScenario,
    ScenarioResult,
    SCENARIO_REGISTRY,
    get_scenario,
    list_scenarios,
)
from src.chaos.config import get_chaos_config, enable_chaos, disable_chaos

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for the chaos runner."""
    
    # Number of times to run each scenario
    runs_per_scenario: int = 10
    
    # Delay between runs (seconds)
    delay_between_runs: float = 1.0
    
    # Whether to stop on first failure
    stop_on_failure: bool = False
    
    # Scenarios to run (None = all)
    scenarios: Optional[List[str]] = None
    
    # Workflow factory for scenarios that need it
    workflow_factory: Optional[Callable] = None
    
    # Initial state for workflow scenarios
    initial_state: Optional[Dict[str, Any]] = None
    
    # Checkpointer for workflow scenarios
    checkpointer: Any = None


@dataclass
class RunnerSummary:
    """Summary of a chaos runner execution."""
    
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    recovered_runs: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: List[ScenarioResult] = field(default_factory=list)
    scenarios_run: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_runs == 0:
            return 0.0
        return (self.successful_runs / self.total_runs) * 100
    
    @property
    def recovery_rate(self) -> float:
        """Recovery rate as a percentage."""
        if self.total_runs == 0:
            return 0.0
        return (self.recovered_runs / self.total_runs) * 100
    
    @property
    def average_mttr_seconds(self) -> Optional[float]:
        """Average MTTR across all runs that had recovery."""
        mttr_values = [r.mttr_seconds for r in self.results if r.mttr_seconds is not None]
        if not mttr_values:
            return None
        return sum(mttr_values) / len(mttr_values)
    
    @property
    def duration_seconds(self) -> float:
        """Total runner duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "recovered_runs": self.recovered_runs,
            "success_rate": self.success_rate,
            "recovery_rate": self.recovery_rate,
            "average_mttr_seconds": self.average_mttr_seconds,
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "scenarios_run": self.scenarios_run,
            "results": [r.to_dict() for r in self.results],
        }
    
    def by_scenario(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics grouped by scenario."""
        by_scenario: Dict[str, List[ScenarioResult]] = {}
        
        for result in self.results:
            if result.scenario_name not in by_scenario:
                by_scenario[result.scenario_name] = []
            by_scenario[result.scenario_name].append(result)
        
        summary = {}
        for scenario_name, results in by_scenario.items():
            successful = sum(1 for r in results if r.success)
            recovered = sum(1 for r in results if r.recovered)
            mttr_values = [r.mttr_seconds for r in results if r.mttr_seconds]
            
            summary[scenario_name] = {
                "runs": len(results),
                "successful": successful,
                "recovered": recovered,
                "success_rate": (successful / len(results)) * 100 if results else 0,
                "recovery_rate": (recovered / len(results)) * 100 if results else 0,
                "average_mttr": sum(mttr_values) / len(mttr_values) if mttr_values else None,
                "min_mttr": min(mttr_values) if mttr_values else None,
                "max_mttr": max(mttr_values) if mttr_values else None,
            }
        
        return summary


class ChaosRunner:
    """Runner for executing chaos scenarios.
    
    The runner manages the execution of multiple chaos scenarios,
    collects results, and generates summary statistics.
    """
    
    def __init__(self, config: Optional[RunnerConfig] = None):
        """Initialize the chaos runner.
        
        Args:
            config: Runner configuration. Uses defaults if None.
        """
        self.config = config or RunnerConfig()
        self.summary = RunnerSummary()
        self._running = False
    
    def run_scenario(
        self,
        scenario: ChaosScenario,
        runs: Optional[int] = None,
    ) -> List[ScenarioResult]:
        """Run a single scenario multiple times.
        
        Args:
            scenario: The scenario to run.
            runs: Number of runs. Uses config default if None.
            
        Returns:
            List of results from each run.
        """
        num_runs = runs or self.config.runs_per_scenario
        results: List[ScenarioResult] = []
        
        logger.info(f"[RUNNER] Running scenario: {scenario.name} ({num_runs} runs)")
        
        for i in range(num_runs):
            logger.info(f"[RUNNER] {scenario.name} run {i + 1}/{num_runs}")
            
            try:
                result = scenario.run()
                results.append(result)
                
                # Update summary
                self.summary.total_runs += 1
                if result.success:
                    self.summary.successful_runs += 1
                else:
                    self.summary.failed_runs += 1
                if result.recovered:
                    self.summary.recovered_runs += 1
                
                self.summary.results.append(result)
                
                # Log result
                status = "SUCCESS" if result.success else "FAILED"
                mttr = f"MTTR: {result.mttr_seconds:.2f}s" if result.mttr_seconds else "No MTTR"
                logger.info(f"[RUNNER] {scenario.name} run {i + 1}: {status} ({mttr})")
                
                # Check stop on failure
                if self.config.stop_on_failure and not result.success:
                    logger.warning(f"[RUNNER] Stopping due to failure in {scenario.name}")
                    break
                
            except Exception as e:
                logger.error(f"[RUNNER] Error in {scenario.name} run {i + 1}: {e}")
                # Create failure result
                error_result = ScenarioResult(
                    scenario_name=scenario.name,
                    success=False,
                    recovered=False,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    error=str(e),
                )
                results.append(error_result)
                self.summary.results.append(error_result)
                self.summary.total_runs += 1
                self.summary.failed_runs += 1
            
            # Delay between runs
            if i < num_runs - 1 and self.config.delay_between_runs > 0:
                time.sleep(self.config.delay_between_runs)
        
        return results
    
    def run_all(
        self,
        scenarios: Optional[List[str]] = None,
    ) -> RunnerSummary:
        """Run all configured scenarios.
        
        Args:
            scenarios: List of scenario names to run. Uses config if None.
            
        Returns:
            Summary of all runs.
        """
        self._running = True
        self.summary = RunnerSummary()
        self.summary.start_time = datetime.utcnow()
        
        # Determine which scenarios to run
        scenario_names = scenarios or self.config.scenarios or list_scenarios()
        
        logger.info(f"[RUNNER] Starting chaos run with {len(scenario_names)} scenarios")
        
        # Enable chaos globally
        enable_chaos()
        
        try:
            for scenario_name in scenario_names:
                if not self._running:
                    logger.info("[RUNNER] Run cancelled")
                    break
                
                # Skip if stop on failure triggered
                if self.config.stop_on_failure and self.summary.failed_runs > 0:
                    logger.warning(f"[RUNNER] Skipping {scenario_name} due to previous failure")
                    continue
                
                try:
                    # Create scenario instance with workflow config if available
                    kwargs = {}
                    if self.config.workflow_factory:
                        kwargs["workflow_factory"] = self.config.workflow_factory
                    if self.config.initial_state:
                        kwargs["initial_state"] = self.config.initial_state
                    if self.config.checkpointer:
                        kwargs["checkpointer"] = self.config.checkpointer
                    
                    scenario = get_scenario(scenario_name, **kwargs)
                    self.summary.scenarios_run.append(scenario_name)
                    
                    # Run the scenario
                    self.run_scenario(scenario)
                    
                except KeyError as e:
                    logger.error(f"[RUNNER] Unknown scenario: {e}")
                except Exception as e:
                    logger.error(f"[RUNNER] Error running {scenario_name}: {e}")
                
        finally:
            # Disable chaos
            disable_chaos()
            self._running = False
        
        self.summary.end_time = datetime.utcnow()
        
        # Log summary
        logger.info(f"[RUNNER] Completed: {self.summary.total_runs} runs, "
                   f"{self.summary.success_rate:.1f}% success rate, "
                   f"{self.summary.recovery_rate:.1f}% recovery rate")
        
        if self.summary.average_mttr_seconds:
            logger.info(f"[RUNNER] Average MTTR: {self.summary.average_mttr_seconds:.2f}s")
        
        return self.summary
    
    def stop(self) -> None:
        """Stop the runner gracefully."""
        self._running = False
        logger.info("[RUNNER] Stop requested")
    
    @property
    def is_running(self) -> bool:
        """Check if the runner is currently executing."""
        return self._running


def run_quick_chaos_test(
    workflow_factory: Optional[Callable] = None,
    initial_state: Optional[Dict[str, Any]] = None,
    checkpointer: Any = None,
    scenarios: Optional[List[str]] = None,
    runs_per_scenario: int = 3,
) -> RunnerSummary:
    """Quick helper to run chaos tests with minimal setup.
    
    Args:
        workflow_factory: Factory function to create workflow.
        initial_state: Initial state for workflow.
        checkpointer: Checkpointer instance.
        scenarios: Scenarios to run (None = simple scenarios only).
        runs_per_scenario: Number of runs per scenario.
        
    Returns:
        Summary of test runs.
        
    Example:
        >>> from src.workflows.product_workflow import create_product_upload_workflow
        >>> summary = run_quick_chaos_test(
        ...     workflow_factory=create_product_upload_workflow,
        ...     initial_state={...},
        ...     runs_per_scenario=5,
        ... )
        >>> print(f"Success rate: {summary.success_rate}%")
    """
    # Use simple scenarios by default (no workflow required)
    default_scenarios = ["HandoffCorruption", "DelayedRecovery"]
    
    config = RunnerConfig(
        runs_per_scenario=runs_per_scenario,
        delay_between_runs=0.5,
        scenarios=scenarios or default_scenarios,
        workflow_factory=workflow_factory,
        initial_state=initial_state,
        checkpointer=checkpointer,
    )
    
    runner = ChaosRunner(config)
    return runner.run_all()

