"""Resilience metrics collection for chaos engineering."""

import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from src.chaos.scenarios import ScenarioResult


@dataclass
class RecoveryEvent:
    """Record of a single recovery event."""
    
    agent_id: str
    thread_id: str
    failure_type: str
    failure_time: datetime
    recovery_time: Optional[datetime] = None
    success: bool = False
    reconstruction_accuracy: Optional[float] = None
    peer_context_used: bool = False
    peer_agents_queried: int = 0
    
    @property
    def mttr_seconds(self) -> Optional[float]:
        """Mean Time to Recovery in seconds."""
        if self.failure_time and self.recovery_time:
            return (self.recovery_time - self.failure_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "failure_type": self.failure_type,
            "failure_time": self.failure_time.isoformat(),
            "recovery_time": self.recovery_time.isoformat() if self.recovery_time else None,
            "mttr_seconds": self.mttr_seconds,
            "success": self.success,
            "reconstruction_accuracy": self.reconstruction_accuracy,
            "peer_context_used": self.peer_context_used,
            "peer_agents_queried": self.peer_agents_queried,
        }


@dataclass
class MetricsSummary:
    """Summary statistics for resilience metrics."""
    
    # MTTR-A (Mean Time to Recovery - Agentic)
    mttr_avg: Optional[float] = None
    mttr_min: Optional[float] = None
    mttr_max: Optional[float] = None
    mttr_p50: Optional[float] = None
    mttr_p95: Optional[float] = None
    mttr_p99: Optional[float] = None
    mttr_std: Optional[float] = None
    
    # Recovery rates
    total_failures: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    recovery_success_rate: float = 0.0
    
    # Reconstruction accuracy
    avg_reconstruction_accuracy: Optional[float] = None
    
    # Task completion
    total_tasks: int = 0
    completed_tasks: int = 0
    task_completion_rate: float = 0.0
    
    # Peer context usage
    peer_context_usage_rate: float = 0.0
    avg_peers_queried: float = 0.0
    
    # Time period
    measurement_start: Optional[datetime] = None
    measurement_end: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "mttr": {
                "avg_seconds": self.mttr_avg,
                "min_seconds": self.mttr_min,
                "max_seconds": self.mttr_max,
                "p50_seconds": self.mttr_p50,
                "p95_seconds": self.mttr_p95,
                "p99_seconds": self.mttr_p99,
                "std_seconds": self.mttr_std,
            },
            "recovery": {
                "total_failures": self.total_failures,
                "successful": self.successful_recoveries,
                "failed": self.failed_recoveries,
                "success_rate": self.recovery_success_rate,
            },
            "reconstruction": {
                "avg_accuracy": self.avg_reconstruction_accuracy,
            },
            "tasks": {
                "total": self.total_tasks,
                "completed": self.completed_tasks,
                "completion_rate": self.task_completion_rate,
            },
            "peer_context": {
                "usage_rate": self.peer_context_usage_rate,
                "avg_peers_queried": self.avg_peers_queried,
            },
            "measurement_period": {
                "start": self.measurement_start.isoformat() if self.measurement_start else None,
                "end": self.measurement_end.isoformat() if self.measurement_end else None,
            },
        }


class ResilienceMetrics:
    """Collector for resilience and recovery metrics.
    
    Tracks MTTR-A (Mean Time to Recovery - Agentic), recovery success rates,
    reconstruction accuracy, and other resilience metrics for thesis evaluation.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self._recovery_events: List[RecoveryEvent] = []
        self._scenario_results: List[ScenarioResult] = []
        self._task_completions: List[Dict[str, Any]] = []
        self._start_time: Optional[datetime] = None
        self._current_recovery: Optional[RecoveryEvent] = None
    
    def start_measurement(self) -> None:
        """Start a new measurement period."""
        self._start_time = datetime.utcnow()
        self._recovery_events = []
        self._scenario_results = []
        self._task_completions = []
    
    def record_failure(
        self,
        agent_id: str,
        thread_id: str,
        failure_type: str,
    ) -> RecoveryEvent:
        """Record a failure event.
        
        Args:
            agent_id: ID of the failed agent.
            thread_id: Thread/workflow ID.
            failure_type: Type of failure (crash, timeout, etc.).
            
        Returns:
            RecoveryEvent to track recovery.
        """
        event = RecoveryEvent(
            agent_id=agent_id,
            thread_id=thread_id,
            failure_type=failure_type,
            failure_time=datetime.utcnow(),
        )
        self._recovery_events.append(event)
        self._current_recovery = event
        return event
    
    def record_recovery(
        self,
        event: Optional[RecoveryEvent] = None,
        success: bool = True,
        reconstruction_accuracy: Optional[float] = None,
        peer_context_used: bool = False,
        peer_agents_queried: int = 0,
    ) -> None:
        """Record a recovery completion.
        
        Args:
            event: The recovery event to update (uses current if None).
            success: Whether recovery was successful.
            reconstruction_accuracy: Accuracy of state reconstruction (0.0 to 1.0).
            peer_context_used: Whether peer context was used in recovery.
            peer_agents_queried: Number of peer agents queried.
        """
        target_event = event or self._current_recovery
        if target_event is None:
            return
        
        target_event.recovery_time = datetime.utcnow()
        target_event.success = success
        target_event.reconstruction_accuracy = reconstruction_accuracy
        target_event.peer_context_used = peer_context_used
        target_event.peer_agents_queried = peer_agents_queried
        
        self._current_recovery = None
    
    @contextmanager
    def measure_recovery(
        self,
        agent_id: str,
        thread_id: str,
        failure_type: str,
    ):
        """Context manager to measure recovery time.
        
        Args:
            agent_id: ID of the failed agent.
            thread_id: Thread/workflow ID.
            failure_type: Type of failure.
            
        Yields:
            RecoveryEvent being measured.
            
        Example:
            >>> with metrics.measure_recovery("agent-1", "thread-1", "crash") as event:
            ...     # Perform recovery
            ...     result = reconstructor.reconstruct(...)
            ...     event.reconstruction_accuracy = calculate_accuracy(result)
        """
        event = self.record_failure(agent_id, thread_id, failure_type)
        try:
            yield event
            self.record_recovery(event, success=True)
        except Exception:
            self.record_recovery(event, success=False)
            raise
    
    def record_scenario_result(self, result: ScenarioResult) -> None:
        """Record a chaos scenario result.
        
        Args:
            result: Result from a chaos scenario run.
        """
        self._scenario_results.append(result)
    
    def record_task_completion(
        self,
        task_id: str,
        completed: bool,
        had_failure: bool = False,
        recovered: bool = False,
    ) -> None:
        """Record a task completion event.
        
        Args:
            task_id: ID of the task.
            completed: Whether the task completed successfully.
            had_failure: Whether the task experienced a failure.
            recovered: Whether the task recovered from failure.
        """
        self._task_completions.append({
            "task_id": task_id,
            "completed": completed,
            "had_failure": had_failure,
            "recovered": recovered,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def get_summary(self) -> MetricsSummary:
        """Calculate summary statistics from collected metrics.
        
        Returns:
            MetricsSummary with all calculated statistics.
        """
        summary = MetricsSummary()
        summary.measurement_start = self._start_time
        summary.measurement_end = datetime.utcnow()
        
        # Calculate MTTR statistics
        mttr_values = [
            e.mttr_seconds for e in self._recovery_events 
            if e.mttr_seconds is not None
        ]
        
        if mttr_values:
            summary.mttr_avg = statistics.mean(mttr_values)
            summary.mttr_min = min(mttr_values)
            summary.mttr_max = max(mttr_values)
            summary.mttr_p50 = self._percentile(mttr_values, 50)
            summary.mttr_p95 = self._percentile(mttr_values, 95)
            summary.mttr_p99 = self._percentile(mttr_values, 99)
            
            if len(mttr_values) >= 2:
                summary.mttr_std = statistics.stdev(mttr_values)
        
        # Calculate recovery rates
        summary.total_failures = len(self._recovery_events)
        summary.successful_recoveries = sum(
            1 for e in self._recovery_events if e.success
        )
        summary.failed_recoveries = summary.total_failures - summary.successful_recoveries
        
        if summary.total_failures > 0:
            summary.recovery_success_rate = (
                summary.successful_recoveries / summary.total_failures
            ) * 100
        
        # Calculate reconstruction accuracy
        accuracy_values = [
            e.reconstruction_accuracy for e in self._recovery_events
            if e.reconstruction_accuracy is not None
        ]
        if accuracy_values:
            summary.avg_reconstruction_accuracy = statistics.mean(accuracy_values)
        
        # Calculate task completion rates
        summary.total_tasks = len(self._task_completions)
        summary.completed_tasks = sum(
            1 for t in self._task_completions if t["completed"]
        )
        if summary.total_tasks > 0:
            summary.task_completion_rate = (
                summary.completed_tasks / summary.total_tasks
            ) * 100
        
        # Calculate peer context usage
        peer_context_events = [
            e for e in self._recovery_events if e.peer_context_used
        ]
        if self._recovery_events:
            summary.peer_context_usage_rate = (
                len(peer_context_events) / len(self._recovery_events)
            ) * 100
        
        peers_queried = [e.peer_agents_queried for e in self._recovery_events]
        if peers_queried:
            summary.avg_peers_queried = statistics.mean(peers_queried)
        
        return summary
    
    def get_events(self) -> List[RecoveryEvent]:
        """Get all recorded recovery events."""
        return list(self._recovery_events)
    
    def get_scenario_results(self) -> List[ScenarioResult]:
        """Get all recorded scenario results."""
        return list(self._scenario_results)
    
    def clear(self) -> None:
        """Clear all recorded metrics."""
        self._recovery_events = []
        self._scenario_results = []
        self._task_completions = []
        self._start_time = None
        self._current_recovery = None
    
    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        """Calculate percentile of a list of values.
        
        Args:
            values: List of numeric values.
            p: Percentile to calculate (0-100).
            
        Returns:
            The percentile value.
        """
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        
        if f == c:
            return sorted_values[f]
        
        return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics data as dictionary.
        
        Returns:
            Dictionary with all metrics data.
        """
        return {
            "summary": self.get_summary().to_dict(),
            "events": [e.to_dict() for e in self._recovery_events],
            "scenario_results": [r.to_dict() for r in self._scenario_results],
            "task_completions": self._task_completions,
        }


# Global metrics instance
_global_metrics: Optional[ResilienceMetrics] = None


def get_metrics() -> ResilienceMetrics:
    """Get or create the global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = ResilienceMetrics()
    return _global_metrics


def reset_metrics() -> None:
    """Reset the global metrics instance."""
    global _global_metrics
    _global_metrics = None

