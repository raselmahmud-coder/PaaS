"""Export utilities for experiment results."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.experiments.collector import ExperimentMetrics

logger = logging.getLogger(__name__)


class ExperimentExporter:
    """Exports experiment results to various formats."""
    
    def __init__(self, output_dir: Path):
        """Initialize the exporter.
        
        Args:
            output_dir: Base output directory.
        """
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.summary_dir = self.output_dir / "summary"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
    
    def export_results_csv(
        self,
        results: List[Any],
        filename: str,
    ) -> Path:
        """Export results to CSV.
        
        Args:
            results: List of ExperimentResult objects.
            filename: Output filename (without extension).
            
        Returns:
            Path to created file.
        """
        if not results:
            logger.warning("No results to export")
            return None
        
        file_path = self.raw_dir / f"{filename}.csv"
        
        # Define columns
        columns = [
            "run_id",
            "scenario_name",
            "condition_name",
            "success",
            "total_duration_ms",
            "steps_completed",
            "total_steps",
            "failure_occurred",
            "failure_step",
            "recovery_attempted",
            "recovery_success",
            "recovery_time_ms",
            "mttr_seconds",
            "timestamp",
        ]
        
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for result in results:
                row = {
                    "run_id": result.run_id,
                    "scenario_name": result.scenario_name,
                    "condition_name": result.condition_name,
                    "success": result.success,
                    "total_duration_ms": result.total_duration_ms,
                    "steps_completed": result.steps_completed,
                    "total_steps": result.total_steps,
                    "failure_occurred": result.failure_occurred,
                    "failure_step": result.failure_step or "",
                    "recovery_attempted": result.recovery_attempted,
                    "recovery_success": result.recovery_success,
                    "recovery_time_ms": result.recovery_time_ms,
                    "mttr_seconds": result.mttr_seconds or "",
                    "timestamp": result.timestamp,
                }
                writer.writerow(row)
        
        logger.info(f"Exported {len(results)} results to {file_path}")
        return file_path
    
    def export_results_json(
        self,
        results: List[Any],
        filename: str,
    ) -> Path:
        """Export results to JSON.
        
        Args:
            results: List of ExperimentResult objects.
            filename: Output filename (without extension).
            
        Returns:
            Path to created file.
        """
        file_path = self.raw_dir / f"{filename}.json"
        
        data = [r.to_dict() for r in results]
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(results)} results to {file_path}")
        return file_path
    
    def export_metrics_json(
        self,
        metrics: ExperimentMetrics,
        filename: str = "metrics_summary",
    ) -> Path:
        """Export aggregated metrics to JSON.
        
        Args:
            metrics: ExperimentMetrics object.
            filename: Output filename (without extension).
            
        Returns:
            Path to created file.
        """
        file_path = self.summary_dir / f"{filename}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        logger.info(f"Exported metrics to {file_path}")
        return file_path
    
    def export_comparison_csv(
        self,
        results_by_condition: Dict[str, List[Any]],
        filename: str = "condition_comparison",
    ) -> Path:
        """Export condition comparison to CSV.
        
        Args:
            results_by_condition: Dict mapping condition name to results.
            filename: Output filename (without extension).
            
        Returns:
            Path to created file.
        """
        from src.experiments.collector import aggregate_metrics
        
        file_path = self.summary_dir / f"{filename}.csv"
        
        columns = [
            "condition",
            "total_runs",
            "success_rate",
            "recovery_rate",
            "mttr_mean",
            "mttr_p50",
            "mttr_p95",
        ]
        
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for condition_name, results in results_by_condition.items():
                metrics = aggregate_metrics(results)
                
                row = {
                    "condition": condition_name,
                    "total_runs": metrics.total_runs,
                    "success_rate": f"{metrics.success_rate:.4f}",
                    "recovery_rate": f"{metrics.recovery_success_rate:.4f}",
                    "mttr_mean": f"{metrics.mttr_mean:.4f}" if metrics.mttr_mean else "",
                    "mttr_p50": f"{metrics.mttr_p50:.4f}" if metrics.mttr_p50 else "",
                    "mttr_p95": f"{metrics.mttr_p95:.4f}" if metrics.mttr_p95 else "",
                }
                writer.writerow(row)
        
        logger.info(f"Exported comparison to {file_path}")
        return file_path
    
    def export_statistical_tests(
        self,
        results_by_condition: Dict[str, List[Any]],
        filename: str = "statistical_tests",
    ) -> Path:
        """Export statistical test results.
        
        Args:
            results_by_condition: Dict mapping condition name to results.
            filename: Output filename (without extension).
            
        Returns:
            Path to created file.
        """
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available for statistical tests")
            return None
        
        file_path = self.summary_dir / f"{filename}.json"
        
        # Extract success rates for each condition
        condition_names = list(results_by_condition.keys())
        success_rates = {}
        mttr_values = {}
        
        for name, results in results_by_condition.items():
            success_rates[name] = [1 if r.success else 0 for r in results]
            mttr_values[name] = [
                r.recovery_time_ms / 1000.0
                for r in results
                if r.recovery_success and r.recovery_time_ms > 0
            ]
        
        tests = {
            "comparisons": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        # Pairwise t-tests for success rates
        for i, name1 in enumerate(condition_names):
            for name2 in condition_names[i + 1:]:
                if success_rates[name1] and success_rates[name2]:
                    t_stat, p_value = stats.ttest_ind(
                        success_rates[name1],
                        success_rates[name2],
                    )
                    
                    tests["comparisons"].append({
                        "test": "t-test",
                        "metric": "success_rate",
                        "condition_1": name1,
                        "condition_2": name2,
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                    })
                
                # MTTR comparison
                if mttr_values.get(name1) and mttr_values.get(name2):
                    t_stat, p_value = stats.ttest_ind(
                        mttr_values[name1],
                        mttr_values[name2],
                    )
                    
                    tests["comparisons"].append({
                        "test": "t-test",
                        "metric": "mttr",
                        "condition_1": name1,
                        "condition_2": name2,
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                    })
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tests, f, indent=2)
        
        logger.info(f"Exported statistical tests to {file_path}")
        return file_path
    
    def export_all(
        self,
        results_by_condition: Dict[str, List[Any]],
        metrics: ExperimentMetrics,
    ) -> Dict[str, Path]:
        """Export all data.
        
        Args:
            results_by_condition: Dict mapping condition name to results.
            metrics: Aggregated metrics.
            
        Returns:
            Dict mapping export type to file path.
        """
        paths = {}
        
        # Export raw results for each condition
        for condition_name, results in results_by_condition.items():
            path = self.export_results_csv(results, f"{condition_name}_runs")
            if path:
                paths[f"{condition_name}_csv"] = path
            
            path = self.export_results_json(results, f"{condition_name}_runs")
            if path:
                paths[f"{condition_name}_json"] = path
        
        # Export summary metrics
        paths["metrics_summary"] = self.export_metrics_json(metrics)
        
        # Export comparison
        paths["comparison"] = self.export_comparison_csv(results_by_condition)
        
        # Export statistical tests
        path = self.export_statistical_tests(results_by_condition)
        if path:
            paths["statistical_tests"] = path
        
        return paths


def export_to_pandas(results: List[Any]):
    """Convert results to pandas DataFrame.
    
    Args:
        results: List of ExperimentResult objects.
        
    Returns:
        pandas DataFrame.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed")
        return None
    
    data = [r.to_dict() for r in results]
    return pd.DataFrame(data)

