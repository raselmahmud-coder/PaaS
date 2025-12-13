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
    
    # Reconstruction Accuracy Metrics (Phase A - Ground Truth Validation)
    reconstruction_accuracy_mean: float = 0.0
    reconstruction_accuracy_std: float = 0.0
    reconstruction_accuracy_by_condition: Dict[str, float] = field(default_factory=dict)
    
    # Semantic Protocol Metrics (Gap 4)
    semantic_conflicts_total: int = 0
    semantic_conflicts_resolved: int = 0
    semantic_resolution_rate: float = 0.0
    semantic_negotiation_time_ms: float = 0.0
    
    # Recovery Timing Breakdown (Phase D - Gap #10)
    timing_breakdown_mean: Dict[str, float] = field(default_factory=dict)
    timing_breakdown_by_condition: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
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
            "reconstruction_accuracy_mean": self.reconstruction_accuracy_mean,
            "reconstruction_accuracy_std": self.reconstruction_accuracy_std,
            "reconstruction_accuracy_by_condition": self.reconstruction_accuracy_by_condition,
            "semantic_conflicts_total": self.semantic_conflicts_total,
            "semantic_conflicts_resolved": self.semantic_conflicts_resolved,
            "semantic_resolution_rate": self.semantic_resolution_rate,
            "semantic_negotiation_time_ms": self.semantic_negotiation_time_ms,
            "timing_breakdown_mean": self.timing_breakdown_mean,
            "timing_breakdown_by_condition": self.timing_breakdown_by_condition,
            "metrics_by_condition": self.metrics_by_condition,
            "metrics_by_scenario": self.metrics_by_scenario,
            "timestamp": self.timestamp,
        }


@dataclass
class AblationMetrics:
    """Metrics for ablation study analysis (Phase C).
    
    Calculates each component's contribution to overall success rate
    by comparing full system against variants with components removed.
    """
    
    semantic_contribution: float = 0.0
    automata_contribution: float = 0.0
    peer_context_contribution: float = 0.0
    llm_contribution: float = 0.0
    semantic_automata_synergy: float = 0.0
    
    @classmethod
    def calculate(cls, metrics_by_condition: Dict[str, Dict]) -> "AblationMetrics":
        """Calculate contributions from ablation experiment results.
        
        Args:
            metrics_by_condition: Dict mapping condition name to metrics dict
                                  with 'success_rate' key.
                                  
        Returns:
            AblationMetrics with component contributions in percentage points.
        """
        full = metrics_by_condition.get("full_system", {}).get("success_rate", 0)
        
        # Get other conditions, defaulting to 0 if missing (not full)
        no_semantic = metrics_by_condition.get("full_no_semantic", {}).get("success_rate", 0)
        reconstruction = metrics_by_condition.get("reconstruction", {}).get("success_rate", 0)
        llm_only = metrics_by_condition.get("llm_only", {}).get("success_rate", 0)
        automata_only = metrics_by_condition.get("automata_only", {}).get("success_rate", 0)
        
        # Calculate contributions only if we have full_system
        if full == 0:
            return cls()
        
        # If no_semantic is missing, use full (semantic adds nothing)
        if no_semantic == 0:
            no_semantic = full
        
        semantic_contrib = (full - no_semantic) * 100
        automata_contrib = (full - reconstruction) * 100 if reconstruction > 0 else 0
        peer_contrib = (reconstruction - llm_only) * 100 if reconstruction > 0 and llm_only > 0 else 0
        llm_contrib = (full - automata_only) * 100 if automata_only > 0 else 0
        
        synergy = (full - (no_semantic + automata_contrib / 100)) * 100 if no_semantic > 0 else 0
        
        return cls(
            semantic_contribution=semantic_contrib,
            automata_contribution=automata_contrib,
            peer_context_contribution=peer_contrib,
            llm_contribution=llm_contrib,
            semantic_automata_synergy=synergy,
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "semantic_contribution_pp": self.semantic_contribution,
            "automata_contribution_pp": self.automata_contribution,
            "peer_context_contribution_pp": self.peer_context_contribution,
            "llm_contribution_pp": self.llm_contribution,
            "semantic_automata_synergy_pp": self.semantic_automata_synergy,
        }


class MetricsCollector:
    """Collects and aggregates experiment metrics."""
    
    def __init__(self):
        """Initialize the collector."""
        self._results: List[Any] = []  # ExperimentResult objects
        self._mttr_values: List[float] = []
        self._durations: List[float] = []
        self._reconstruction_accuracies: List[float] = []  # Phase A - Ground Truth
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
        
        # Track reconstruction accuracy (Phase A - Ground Truth)
        mean_accuracy = getattr(result, 'mean_reconstruction_accuracy', 0.0)
        if mean_accuracy > 0:
            self._reconstruction_accuracies.append(mean_accuracy)
        
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
        
        # Calculate reconstruction accuracy (Phase A - Ground Truth)
        reconstruction_accuracy_mean = (
            statistics.mean(self._reconstruction_accuracies) 
            if self._reconstruction_accuracies else 0.0
        )
        reconstruction_accuracy_std = (
            statistics.stdev(self._reconstruction_accuracies)
            if len(self._reconstruction_accuracies) > 1 else 0.0
        )
        
        # Calculate accuracy by condition
        reconstruction_accuracy_by_condition = {}
        for condition_name, results in self._by_condition.items():
            accuracies = [
                getattr(r, 'mean_reconstruction_accuracy', 0.0) 
                for r in results 
                if getattr(r, 'mean_reconstruction_accuracy', 0.0) > 0
            ]
            if accuracies:
                reconstruction_accuracy_by_condition[condition_name] = statistics.mean(accuracies)
        
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
        
        # Calculate recovery timing breakdown (Phase D - Gap #10)
        timing_breakdown_mean = self._aggregate_timing_breakdown(self._results)
        timing_breakdown_by_condition = {}
        for condition_name, results in self._by_condition.items():
            timing_breakdown_by_condition[condition_name] = self._aggregate_timing_breakdown(results)
        
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
            reconstruction_accuracy_mean=reconstruction_accuracy_mean,
            reconstruction_accuracy_std=reconstruction_accuracy_std,
            reconstruction_accuracy_by_condition=reconstruction_accuracy_by_condition,
            semantic_conflicts_total=semantic_conflicts_total,
            semantic_conflicts_resolved=semantic_conflicts_resolved,
            semantic_resolution_rate=semantic_resolution_rate,
            semantic_negotiation_time_ms=semantic_negotiation_time_ms,
            timing_breakdown_mean=timing_breakdown_mean,
            timing_breakdown_by_condition=timing_breakdown_by_condition,
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
        
        # Reconstruction accuracy (Phase A - Ground Truth)
        accuracy_values = [
            getattr(r, 'mean_reconstruction_accuracy', 0.0)
            for r in results
            if getattr(r, 'mean_reconstruction_accuracy', 0.0) > 0
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
        
        # Timing breakdown (Phase D - Gap #10)
        timing_breakdown = self._aggregate_timing_breakdown(results)
        
        return {
            "total_runs": total,
            "successful_runs": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "failures_injected": failures,
            "recoveries_successful": recoveries,
            "recovery_rate": recoveries / failures if failures > 0 else 0.0,
            "mttr_mean": statistics.mean(mttr_values) if mttr_values else None,
            "mttr_p50": self._percentile(sorted(mttr_values), 50) if mttr_values else None,
            "reconstruction_accuracy_mean": statistics.mean(accuracy_values) if accuracy_values else None,
            "semantic_conflicts": semantic_conflicts,
            "semantic_resolved": semantic_resolved,
            "semantic_resolution_rate": semantic_resolution_rate,
            "timing_breakdown": timing_breakdown,
        }
    
    def _aggregate_timing_breakdown(self, results: List[Any]) -> Dict[str, float]:
        """Aggregate recovery timing breakdown from results (Phase D - Gap #10).
        
        Args:
            results: List of ExperimentResult objects.
            
        Returns:
            Dict with mean timing values for each operation.
        """
        timing_keys = [
            "checkpoint_load_ms",
            "event_query_ms", 
            "peer_context_ms",
            "automata_predict_ms",
            "llm_inference_ms",
            "state_merge_ms",
            "accuracy_calc_ms",
            "total_ms",
        ]
        
        timing_values: Dict[str, List[float]] = {k: [] for k in timing_keys}
        
        for result in results:
            timing = getattr(result, 'recovery_timing_breakdown', None)
            if timing is not None:
                for key in timing_keys:
                    value = getattr(timing, key, 0.0)
                    if value > 0:
                        timing_values[key].append(value)
        
        # Calculate means
        timing_means = {}
        for key, values in timing_values.items():
            if values:
                timing_means[key] = statistics.mean(values)
        
        return timing_means
    
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
        self._reconstruction_accuracies.clear()
        self._by_condition.clear()
        self._by_scenario.clear()


