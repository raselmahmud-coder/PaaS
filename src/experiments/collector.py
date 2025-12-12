"""Metrics collector for experiment results."""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetrics:
    """Aggregated metrics from experiments."""
    
    # Counts
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    failures_injected: int = 0
    recoveries_attempted: int = 0
    recoveries_successful: int = 0
    
    # Rates
    success_rate: float = 0.0
    failure_rate: float = 0.0
    recovery_success_rate: float = 0.0
    
    # MTTR (Mean Time To Recovery - Agentic)
    mttr_mean: Optional[float] = None
    mttr_std: Optional[float] = None
    mttr_p50: Optional[float] = None
    mttr_p95: Optional[float] = None
    mttr_p99: Optional[float] = None
    
    # Duration
    duration_mean_ms: float = 0.0
    duration_std_ms: float = 0.0
    
    # Semantic Protocol Metrics (Gap 4)
    semantic_conflicts_total: int = 0
    semantic_conflicts_resolved: int = 0
    semantic_resolution_rate: float = 0.0
    semantic_negotiation_time_ms: float = 0.0
    
    # By condition
    metrics_by_condition: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # By scenario
    metrics_by_scenario: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "failures_injected": self.failures_injected,
            "recoveries_attempted": self.recoveries_attempted,
            "recoveries_successful": self.recoveries_successful,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "recovery_success_rate": self.recovery_success_rate,
            "mttr_mean": self.mttr_mean,
            "mttr_std": self.mttr_std,
            "mttr_p50": self.mttr_p50,
            "mttr_p95": self.mttr_p95,
            "mttr_p99": self.mttr_p99,
            "duration_mean_ms": self.duration_mean_ms,
            "duration_std_ms": self.duration_std_ms,
            "semantic_conflicts_total": self.semantic_conflicts_total,
            "semantic_conflicts_resolved": self.semantic_conflicts_resolved,
            "semantic_resolution_rate": self.semantic_resolution_rate,
            "semantic_negotiation_time_ms": self.semantic_negotiation_time_ms,
            "metrics_by_condition": self.metrics_by_condition,
            "metrics_by_scenario": self.metrics_by_scenario,
            "timestamp": self.timestamp,
        }


class MetricsCollector:
    """Collects and aggregates experiment metrics."""
    
    def __init__(self):
        """Initialize the collector."""
        self._results: List[Any] = []  # ExperimentResult objects
        self._mttr_values: List[float] = []
        self._durations: List[float] = []
        self._by_condition: Dict[str, List[Any]] = {}
        self._by_scenario: Dict[str, List[Any]] = {}
    
    def record_result(self, result: Any) -> None:
        """Record an experiment result.
        
        Args:
            result: ExperimentResult object.
        """
        self._results.append(result)
        self._durations.append(result.total_duration_ms)
        
        # Track MTTR if recovery occurred
        if result.recovery_success and result.recovery_time_ms > 0:
            self._mttr_values.append(result.recovery_time_ms / 1000.0)  # Convert to seconds
        
        # Group by condition
        if result.condition_name not in self._by_condition:
            self._by_condition[result.condition_name] = []
        self._by_condition[result.condition_name].append(result)
        
        # Group by scenario
        if result.scenario_name not in self._by_scenario:
            self._by_scenario[result.scenario_name] = []
        self._by_scenario[result.scenario_name].append(result)
    
    def get_metrics(self) -> ExperimentMetrics:
        """Get aggregated metrics.
        
        Returns:
            ExperimentMetrics with all computed values.
        """
        if not self._results:
            return ExperimentMetrics()
        
        # Count totals
        total_runs = len(self._results)
        successful_runs = sum(1 for r in self._results if r.success)
        failed_runs = total_runs - successful_runs
        failures_injected = sum(1 for r in self._results if r.failure_occurred)
        recoveries_attempted = sum(1 for r in self._results if r.recovery_attempted)
        recoveries_successful = sum(1 for r in self._results if r.recovery_success)
        
        # Calculate rates
        success_rate = successful_runs / total_runs if total_runs > 0 else 0.0
        failure_rate = failures_injected / total_runs if total_runs > 0 else 0.0
        recovery_success_rate = (
            recoveries_successful / recoveries_attempted
            if recoveries_attempted > 0
            else 0.0
        )
        
        # Calculate MTTR statistics
        mttr_mean = None
        mttr_std = None
        mttr_p50 = None
        mttr_p95 = None
        mttr_p99 = None
        
        if self._mttr_values:
            mttr_mean = statistics.mean(self._mttr_values)
            if len(self._mttr_values) > 1:
                mttr_std = statistics.stdev(self._mttr_values)
            
            sorted_mttr = sorted(self._mttr_values)
            mttr_p50 = self._percentile(sorted_mttr, 50)
            mttr_p95 = self._percentile(sorted_mttr, 95)
            mttr_p99 = self._percentile(sorted_mttr, 99)
        
        # Calculate duration statistics
        duration_mean_ms = statistics.mean(self._durations) if self._durations else 0.0
        duration_std_ms = (
            statistics.stdev(self._durations)
            if len(self._durations) > 1
            else 0.0
        )
        
        # Calculate semantic protocol metrics (Gap 4)
        semantic_conflicts_total = sum(
            getattr(r, 'semantic_conflicts', 0) for r in self._results
        )
        semantic_conflicts_resolved = sum(
            getattr(r, 'semantic_resolved', 0) for r in self._results
        )
        semantic_resolution_rate = (
            semantic_conflicts_resolved / semantic_conflicts_total
            if semantic_conflicts_total > 0
            else 0.0
        )
        semantic_negotiation_times = [
            getattr(r, 'semantic_negotiation_ms', 0) 
            for r in self._results 
            if getattr(r, 'semantic_negotiation_ms', 0) > 0
        ]
        semantic_negotiation_time_ms = (
            statistics.mean(semantic_negotiation_times)
            if semantic_negotiation_times
            else 0.0
        )
        
        # Calculate by-condition metrics
        metrics_by_condition = {}
        for condition_name, results in self._by_condition.items():
            metrics_by_condition[condition_name] = self._calculate_group_metrics(
                results
            )
        
        # Calculate by-scenario metrics
        metrics_by_scenario = {}
        for scenario_name, results in self._by_scenario.items():
            metrics_by_scenario[scenario_name] = self._calculate_group_metrics(
                results
            )
        
        return ExperimentMetrics(
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            failures_injected=failures_injected,
            recoveries_attempted=recoveries_attempted,
            recoveries_successful=recoveries_successful,
            success_rate=success_rate,
            failure_rate=failure_rate,
            recovery_success_rate=recovery_success_rate,
            mttr_mean=mttr_mean,
            mttr_std=mttr_std,
            mttr_p50=mttr_p50,
            mttr_p95=mttr_p95,
            mttr_p99=mttr_p99,
            duration_mean_ms=duration_mean_ms,
            duration_std_ms=duration_std_ms,
            semantic_conflicts_total=semantic_conflicts_total,
            semantic_conflicts_resolved=semantic_conflicts_resolved,
            semantic_resolution_rate=semantic_resolution_rate,
            semantic_negotiation_time_ms=semantic_negotiation_time_ms,
            metrics_by_condition=metrics_by_condition,
            metrics_by_scenario=metrics_by_scenario,
        )
    
    def _calculate_group_metrics(self, results: List[Any]) -> Dict[str, Any]:
        """Calculate metrics for a group of results."""
        if not results:
            return {}
        
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failures = sum(1 for r in results if r.failure_occurred)
        recoveries = sum(1 for r in results if r.recovery_success)
        
        mttr_values = [
            r.recovery_time_ms / 1000.0
            for r in results
            if r.recovery_success and r.recovery_time_ms > 0
        ]
        
        # Semantic metrics (Gap 4)
        semantic_conflicts = sum(
            getattr(r, 'semantic_conflicts', 0) for r in results
        )
        semantic_resolved = sum(
            getattr(r, 'semantic_resolved', 0) for r in results
        )
        semantic_resolution_rate = (
            semantic_resolved / semantic_conflicts
            if semantic_conflicts > 0
            else 0.0
        )
        
        return {
            "total_runs": total,
            "successful_runs": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "failures_injected": failures,
            "recoveries_successful": recoveries,
            "recovery_rate": recoveries / failures if failures > 0 else 0.0,
            "mttr_mean": statistics.mean(mttr_values) if mttr_values else None,
            "mttr_p50": self._percentile(sorted(mttr_values), 50) if mttr_values else None,
            "semantic_conflicts": semantic_conflicts,
            "semantic_resolved": semantic_resolved,
            "semantic_resolution_rate": semantic_resolution_rate,
        }
    
    @staticmethod
    def _percentile(sorted_data: List[float], percentile: int) -> float:
        """Calculate percentile of sorted data."""
        if not sorted_data:
            return 0.0
        
        k = (len(sorted_data) - 1) * (percentile / 100.0)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        
        if f == c:
            return sorted_data[f]
        
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
    
    def get_results_for_condition(self, condition_name: str) -> List[Any]:
        """Get results for a specific condition."""
        return self._by_condition.get(condition_name, [])
    
    def get_results_for_scenario(self, scenario_name: str) -> List[Any]:
        """Get results for a specific scenario."""
        return self._by_scenario.get(scenario_name, [])
    
    def clear(self) -> None:
        """Clear all collected data."""
        self._results.clear()
        self._mttr_values.clear()
        self._durations.clear()
        self._by_condition.clear()
        self._by_scenario.clear()


def aggregate_metrics(results: List[Any]) -> ExperimentMetrics:
    """Aggregate metrics from a list of results.
    
    Args:
        results: List of ExperimentResult objects.
        
    Returns:
        ExperimentMetrics with aggregated values.
    """
    collector = MetricsCollector()
    for result in results:
        collector.record_result(result)
    return collector.get_metrics()

