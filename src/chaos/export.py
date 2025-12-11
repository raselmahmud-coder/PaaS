"""Export utilities for resilience metrics - JSON and CSV formats."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.chaos.metrics import ResilienceMetrics, MetricsSummary, RecoveryEvent
from src.chaos.scenarios import ScenarioResult
from src.chaos.runner import RunnerSummary


class MetricsExporter:
    """Export metrics to various formats for thesis evaluation."""
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "data/metrics",
        prefix: str = "chaos",
    ):
        """Initialize the exporter.
        
        Args:
            output_dir: Directory to write output files.
            prefix: Prefix for output filenames.
        """
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        
        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_filename(self, name: str, extension: str) -> Path:
        """Generate a timestamped filename.
        
        Args:
            name: Base name for the file.
            extension: File extension (without dot).
            
        Returns:
            Full path to the file.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.prefix}_{name}_{timestamp}.{extension}"
        return self.output_dir / filename
    
    # =========================================================================
    # JSON Export
    # =========================================================================
    
    def export_summary_json(
        self,
        summary: Union[MetricsSummary, RunnerSummary],
        filename: Optional[str] = None,
    ) -> Path:
        """Export summary statistics to JSON.
        
        Args:
            summary: MetricsSummary or RunnerSummary to export.
            filename: Optional custom filename.
            
        Returns:
            Path to the created file.
        """
        filepath = Path(filename) if filename else self._get_filename("summary", "json")
        
        data = summary.to_dict()
        data["exported_at"] = datetime.utcnow().isoformat()
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def export_events_json(
        self,
        events: List[RecoveryEvent],
        filename: Optional[str] = None,
    ) -> Path:
        """Export recovery events to JSON.
        
        Args:
            events: List of RecoveryEvent objects.
            filename: Optional custom filename.
            
        Returns:
            Path to the created file.
        """
        filepath = Path(filename) if filename else self._get_filename("events", "json")
        
        data = {
            "exported_at": datetime.utcnow().isoformat(),
            "event_count": len(events),
            "events": [e.to_dict() for e in events],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def export_scenarios_json(
        self,
        results: List[ScenarioResult],
        filename: Optional[str] = None,
    ) -> Path:
        """Export scenario results to JSON.
        
        Args:
            results: List of ScenarioResult objects.
            filename: Optional custom filename.
            
        Returns:
            Path to the created file.
        """
        filepath = Path(filename) if filename else self._get_filename("scenarios", "json")
        
        data = {
            "exported_at": datetime.utcnow().isoformat(),
            "scenario_count": len(results),
            "results": [r.to_dict() for r in results],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    def export_full_metrics_json(
        self,
        metrics: ResilienceMetrics,
        filename: Optional[str] = None,
    ) -> Path:
        """Export all metrics data to a single JSON file.
        
        Args:
            metrics: ResilienceMetrics instance with collected data.
            filename: Optional custom filename.
            
        Returns:
            Path to the created file.
        """
        filepath = Path(filename) if filename else self._get_filename("full_metrics", "json")
        
        data = metrics.to_dict()
        data["exported_at"] = datetime.utcnow().isoformat()
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        return filepath
    
    # =========================================================================
    # CSV Export
    # =========================================================================
    
    def export_events_csv(
        self,
        events: List[RecoveryEvent],
        filename: Optional[str] = None,
    ) -> Path:
        """Export recovery events to CSV.
        
        Args:
            events: List of RecoveryEvent objects.
            filename: Optional custom filename.
            
        Returns:
            Path to the created file.
        """
        filepath = Path(filename) if filename else self._get_filename("events", "csv")
        
        fieldnames = [
            "agent_id",
            "thread_id",
            "failure_type",
            "failure_time",
            "recovery_time",
            "mttr_seconds",
            "success",
            "reconstruction_accuracy",
            "peer_context_used",
            "peer_agents_queried",
        ]
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in events:
                writer.writerow(event.to_dict())
        
        return filepath
    
    def export_scenarios_csv(
        self,
        results: List[ScenarioResult],
        filename: Optional[str] = None,
    ) -> Path:
        """Export scenario results to CSV.
        
        Args:
            results: List of ScenarioResult objects.
            filename: Optional custom filename.
            
        Returns:
            Path to the created file.
        """
        filepath = Path(filename) if filename else self._get_filename("scenarios", "csv")
        
        fieldnames = [
            "scenario_name",
            "success",
            "recovered",
            "start_time",
            "end_time",
            "duration_seconds",
            "mttr_seconds",
            "error",
        ]
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = result.to_dict()
                # Flatten for CSV
                row = {k: row[k] for k in fieldnames if k in row}
                writer.writerow(row)
        
        return filepath
    
    def export_mttr_csv(
        self,
        events: List[RecoveryEvent],
        filename: Optional[str] = None,
    ) -> Path:
        """Export MTTR data specifically for thesis analysis.
        
        Args:
            events: List of RecoveryEvent objects.
            filename: Optional custom filename.
            
        Returns:
            Path to the created file.
        """
        filepath = Path(filename) if filename else self._get_filename("mttr", "csv")
        
        fieldnames = [
            "failure_type",
            "mttr_seconds",
            "success",
            "peer_context_used",
        ]
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in events:
                if event.mttr_seconds is not None:
                    writer.writerow({
                        "failure_type": event.failure_type,
                        "mttr_seconds": event.mttr_seconds,
                        "success": event.success,
                        "peer_context_used": event.peer_context_used,
                    })
        
        return filepath
    
    def export_summary_csv(
        self,
        summary: MetricsSummary,
        filename: Optional[str] = None,
    ) -> Path:
        """Export summary statistics to CSV (single row).
        
        Args:
            summary: MetricsSummary to export.
            filename: Optional custom filename.
            
        Returns:
            Path to the created file.
        """
        filepath = Path(filename) if filename else self._get_filename("summary", "csv")
        
        fieldnames = [
            "mttr_avg",
            "mttr_min",
            "mttr_max",
            "mttr_p50",
            "mttr_p95",
            "mttr_p99",
            "mttr_std",
            "total_failures",
            "successful_recoveries",
            "failed_recoveries",
            "recovery_success_rate",
            "avg_reconstruction_accuracy",
            "total_tasks",
            "completed_tasks",
            "task_completion_rate",
            "peer_context_usage_rate",
            "avg_peers_queried",
        ]
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            row = {
                "mttr_avg": summary.mttr_avg,
                "mttr_min": summary.mttr_min,
                "mttr_max": summary.mttr_max,
                "mttr_p50": summary.mttr_p50,
                "mttr_p95": summary.mttr_p95,
                "mttr_p99": summary.mttr_p99,
                "mttr_std": summary.mttr_std,
                "total_failures": summary.total_failures,
                "successful_recoveries": summary.successful_recoveries,
                "failed_recoveries": summary.failed_recoveries,
                "recovery_success_rate": summary.recovery_success_rate,
                "avg_reconstruction_accuracy": summary.avg_reconstruction_accuracy,
                "total_tasks": summary.total_tasks,
                "completed_tasks": summary.completed_tasks,
                "task_completion_rate": summary.task_completion_rate,
                "peer_context_usage_rate": summary.peer_context_usage_rate,
                "avg_peers_queried": summary.avg_peers_queried,
            }
            writer.writerow(row)
        
        return filepath
    
    # =========================================================================
    # Batch Export
    # =========================================================================
    
    def export_all(
        self,
        metrics: ResilienceMetrics,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """Export all metrics data in multiple formats.
        
        Args:
            metrics: ResilienceMetrics instance with collected data.
            formats: List of formats to export ("json", "csv"). Defaults to both.
            
        Returns:
            Dictionary mapping format/type to output file paths.
        """
        formats = formats or ["json", "csv"]
        outputs: Dict[str, Path] = {}
        
        summary = metrics.get_summary()
        events = metrics.get_events()
        scenarios = metrics.get_scenario_results()
        
        if "json" in formats:
            outputs["full_metrics_json"] = self.export_full_metrics_json(metrics)
            outputs["summary_json"] = self.export_summary_json(summary)
            
            if events:
                outputs["events_json"] = self.export_events_json(events)
            
            if scenarios:
                outputs["scenarios_json"] = self.export_scenarios_json(scenarios)
        
        if "csv" in formats:
            outputs["summary_csv"] = self.export_summary_csv(summary)
            
            if events:
                outputs["events_csv"] = self.export_events_csv(events)
                outputs["mttr_csv"] = self.export_mttr_csv(events)
            
            if scenarios:
                outputs["scenarios_csv"] = self.export_scenarios_csv(scenarios)
        
        return outputs
    
    def export_runner_results(
        self,
        summary: RunnerSummary,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """Export chaos runner results.
        
        Args:
            summary: RunnerSummary from chaos runner.
            formats: List of formats to export.
            
        Returns:
            Dictionary mapping format/type to output file paths.
        """
        formats = formats or ["json", "csv"]
        outputs: Dict[str, Path] = {}
        
        if "json" in formats:
            outputs["runner_summary_json"] = self.export_summary_json(summary)
            
            if summary.results:
                outputs["runner_scenarios_json"] = self.export_scenarios_json(summary.results)
        
        if "csv" in formats:
            if summary.results:
                outputs["runner_scenarios_csv"] = self.export_scenarios_csv(summary.results)
        
        return outputs


def quick_export(
    metrics: ResilienceMetrics,
    output_dir: str = "data/metrics",
) -> Dict[str, Path]:
    """Quick helper to export metrics with default settings.
    
    Args:
        metrics: ResilienceMetrics to export.
        output_dir: Output directory.
        
    Returns:
        Dictionary of exported file paths.
        
    Example:
        >>> metrics = get_metrics()
        >>> # ... run chaos tests ...
        >>> files = quick_export(metrics)
        >>> print(f"Exported to: {files}")
    """
    exporter = MetricsExporter(output_dir=output_dir)
    return exporter.export_all(metrics)

