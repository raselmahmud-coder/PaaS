"""Tests for the experiment framework."""

import os
import tempfile
from pathlib import Path

import pytest

# Test scenario loader
class TestScenarioLoader:
    """Tests for YAML scenario loading."""
    
    def test_load_vendor_onboarding(self):
        """Test loading vendor onboarding scenario."""
        from src.experiments.scenario_loader import load_scenario
        
        scenario = load_scenario("vendor_onboarding")
        
        assert scenario.name == "Vendor Onboarding"
        assert scenario.complexity == "medium"
        assert len(scenario.steps) == 5
        assert scenario.failure_injection.enabled is True
        assert scenario.failure_injection.probability == 0.3
    
    def test_load_product_launch(self):
        """Test loading product launch scenario."""
        from src.experiments.scenario_loader import load_scenario
        
        scenario = load_scenario("product_launch")
        
        assert scenario.name == "Product Launch Campaign"
        assert scenario.complexity == "high"
        assert len(scenario.steps) == 7
    
    def test_load_customer_feedback(self):
        """Test loading customer feedback scenario."""
        from src.experiments.scenario_loader import load_scenario
        
        scenario = load_scenario("customer_feedback")
        
        assert scenario.name == "Customer Feedback Loop"
        assert len(scenario.steps) == 7
    
    def test_load_inventory_crisis(self):
        """Test loading inventory crisis scenario."""
        from src.experiments.scenario_loader import load_scenario
        
        scenario = load_scenario("inventory_crisis")
        
        assert scenario.name == "Inventory Crisis Response"
        assert scenario.complexity == "high"
        assert len(scenario.steps) == 7
    
    def test_load_all_scenarios(self):
        """Test loading all scenarios."""
        from src.experiments.scenario_loader import load_all_scenarios
        
        scenarios = load_all_scenarios()
        
        assert len(scenarios) >= 4
        names = [s.name for s in scenarios]
        assert "Vendor Onboarding" in names
        assert "Product Launch Campaign" in names
    
    def test_scenario_step_structure(self):
        """Test scenario step structure."""
        from src.experiments.scenario_loader import load_scenario
        
        scenario = load_scenario("vendor_onboarding")
        
        step = scenario.steps[0]
        assert step.name == "validate_product_data"
        assert step.agent == "product-agent"
        assert step.expected_status == "validated"
        assert step.timeout_seconds == 30
    
    def test_scenario_get_initial_state(self):
        """Test initial state with run_id substitution."""
        from src.experiments.scenario_loader import load_scenario
        
        scenario = load_scenario("vendor_onboarding")
        state = scenario.get_initial_state("test-123")
        
        assert "task_id" in state
        assert "test-123" in state["task_id"]
    
    def test_scenario_agent_ids(self):
        """Test getting agent IDs."""
        from src.experiments.scenario_loader import load_scenario
        
        scenario = load_scenario("vendor_onboarding")
        
        assert "product-agent" in scenario.agent_ids
        assert "marketing-agent" in scenario.agent_ids
    
    def test_scenario_not_found(self):
        """Test loading non-existent scenario."""
        from src.experiments.scenario_loader import load_scenario
        
        with pytest.raises(FileNotFoundError):
            load_scenario("nonexistent_scenario")


class TestExperimentConditions:
    """Tests for experimental conditions."""
    
    def test_baseline_condition(self):
        """Test baseline condition configuration."""
        from src.experiments.conditions import BaselineCondition
        
        condition = BaselineCondition()
        
        assert condition.name == "baseline"
        assert condition.should_attempt_recovery() is False
        assert condition.should_use_semantic_protocol() is False
        assert condition.should_use_automata() is False
        assert condition.get_reconstruction_strategy() == "none"
    
    def test_reconstruction_condition(self):
        """Test reconstruction condition configuration."""
        from src.experiments.conditions import ReconstructionCondition
        
        condition = ReconstructionCondition()
        
        assert condition.name == "reconstruction"
        assert condition.should_attempt_recovery() is True
        assert condition.should_use_semantic_protocol() is False
        assert condition.should_use_automata() is False
        assert condition.get_reconstruction_strategy() == "llm"
    
    def test_full_system_condition(self):
        """Test full system condition configuration."""
        from src.experiments.conditions import FullSystemCondition
        
        condition = FullSystemCondition()
        
        assert condition.name == "full_system"
        assert condition.should_attempt_recovery() is True
        assert condition.should_use_semantic_protocol() is True
        assert condition.should_use_automata() is True
        assert condition.get_reconstruction_strategy() == "hybrid"
    
    def test_get_condition(self):
        """Test getting condition by name."""
        from src.experiments.conditions import get_condition
        
        baseline = get_condition("baseline")
        assert baseline.name == "baseline"
        
        reconstruction = get_condition("reconstruction")
        assert reconstruction.name == "reconstruction"
        
        full_system = get_condition("full_system")
        assert full_system.name == "full_system"
    
    def test_get_condition_invalid(self):
        """Test getting invalid condition."""
        from src.experiments.conditions import get_condition
        
        with pytest.raises(KeyError):
            get_condition("invalid_condition")
    
    def test_list_conditions(self):
        """Test listing available conditions."""
        from src.experiments.conditions import list_conditions
        
        conditions = list_conditions()
        
        assert "baseline" in conditions
        assert "reconstruction" in conditions
        assert "full_system" in conditions
    
    def test_condition_to_dict(self):
        """Test condition serialization."""
        from src.experiments.conditions import BaselineCondition
        
        condition = BaselineCondition()
        data = condition.to_dict()
        
        assert data["name"] == "baseline"
        assert data["resilience_enabled"] is False


class TestComparisonBaselineConditions:
    """Tests for comparison baseline conditions (related work comparison)."""
    
    def test_simple_retry_condition(self):
        """Test simple retry condition configuration."""
        from src.experiments.conditions import SimpleRetryCondition
        
        condition = SimpleRetryCondition()
        
        assert condition.name == "simple_retry"
        assert condition.should_attempt_recovery() is True
        assert condition.should_use_automata() is False
        assert condition.should_use_semantic_protocol() is False
        assert condition.should_query_peers() is False
        assert condition.config.max_retries == 3
        assert condition.get_reconstruction_strategy() == "simple_retry"
    
    def test_checkpoint_only_condition(self):
        """Test checkpoint-only condition configuration."""
        from src.experiments.conditions import CheckpointOnlyCondition
        
        condition = CheckpointOnlyCondition()
        
        assert condition.name == "checkpoint_only"
        assert condition.should_attempt_recovery() is True
        assert condition.should_use_automata() is False
        assert condition.should_use_semantic_protocol() is False
        assert condition.config.use_checkpoint_restart is True
        assert condition.config.llm_fallback_enabled is False
        assert condition.get_reconstruction_strategy() == "checkpoint"
    
    def test_automata_only_condition(self):
        """Test automata-only condition configuration."""
        from src.experiments.conditions import AutomataOnlyCondition
        
        condition = AutomataOnlyCondition()
        
        assert condition.name == "automata_only"
        assert condition.should_attempt_recovery() is True
        assert condition.should_use_automata() is True
        assert condition.should_use_semantic_protocol() is False
        assert condition.should_query_peers() is False
        assert condition.config.llm_fallback_enabled is False
        assert condition.get_reconstruction_strategy() == "automata_only"
    
    def test_llm_only_condition(self):
        """Test LLM-only condition configuration."""
        from src.experiments.conditions import LLMOnlyCondition
        
        condition = LLMOnlyCondition()
        
        assert condition.name == "llm_only"
        assert condition.should_attempt_recovery() is True
        assert condition.should_use_automata() is False
        assert condition.should_use_semantic_protocol() is False
        assert condition.should_query_peers() is False  # No peer context
        assert condition.config.llm_fallback_enabled is True
        assert condition.get_reconstruction_strategy() == "llm_no_peer"
    
    def test_get_comparison_conditions(self):
        """Test getting all comparison baseline conditions by name."""
        from src.experiments.conditions import get_condition
        
        simple_retry = get_condition("simple_retry")
        assert simple_retry.name == "simple_retry"
        
        checkpoint_only = get_condition("checkpoint_only")
        assert checkpoint_only.name == "checkpoint_only"
        
        automata_only = get_condition("automata_only")
        assert automata_only.name == "automata_only"
        
        llm_only = get_condition("llm_only")
        assert llm_only.name == "llm_only"
    
    def test_comparison_conditions_in_list(self):
        """Test comparison conditions are listed."""
        from src.experiments.conditions import list_conditions
        
        conditions = list_conditions()
        
        assert "simple_retry" in conditions
        assert "checkpoint_only" in conditions
        assert "automata_only" in conditions
        assert "llm_only" in conditions
        assert len(conditions) == 7  # 3 original + 4 comparison
    
    def test_comparison_condition_to_dict(self):
        """Test comparison condition serialization includes new fields."""
        from src.experiments.conditions import SimpleRetryCondition
        
        condition = SimpleRetryCondition()
        data = condition.to_dict()
        
        assert data["name"] == "simple_retry"
        assert data["max_retries"] == 3
        assert data["use_checkpoint_restart"] is False
        assert data["llm_fallback_enabled"] is False


class TestComparisonRecoveryStrategies:
    """Tests for recovery behavior of comparison baseline conditions."""
    
    def test_simple_retry_recovery(self):
        """Test simple retry has low recovery rate."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import SimpleRetryCondition
        
        runner = ExperimentRunner(failure_probability=0.9, seed=42)
        condition = SimpleRetryCondition()
        
        results = runner.run_batch("vendor_onboarding", condition, num_runs=50)
        
        # Should have recovery attempts
        recoveries = [r for r in results if r.recovery_attempted]
        assert len(recoveries) > 0
        
        # Simple retry should have lower success rate than hybrid
        successful = [r for r in recoveries if r.recovery_success]
        if recoveries:
            recovery_rate = len(successful) / len(recoveries)
            # Simple retry typically around 35%
            assert recovery_rate < 0.6
    
    def test_checkpoint_only_recovery(self):
        """Test checkpoint-only has moderate recovery rate."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import CheckpointOnlyCondition
        
        runner = ExperimentRunner(failure_probability=0.9, seed=42)
        condition = CheckpointOnlyCondition()
        
        results = runner.run_batch("vendor_onboarding", condition, num_runs=50)
        
        recoveries = [r for r in results if r.recovery_attempted]
        successful = [r for r in recoveries if r.recovery_success]
        
        if recoveries:
            recovery_rate = len(successful) / len(recoveries)
            # Checkpoint typically around 55%
            assert recovery_rate > 0.3
            assert recovery_rate < 0.8
    
    def test_automata_only_recovery(self):
        """Test automata-only has good recovery rate for structured workflows."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import AutomataOnlyCondition
        
        runner = ExperimentRunner(failure_probability=0.9, seed=42)
        condition = AutomataOnlyCondition()
        
        results = runner.run_batch("vendor_onboarding", condition, num_runs=50)
        
        recoveries = [r for r in results if r.recovery_attempted]
        successful = [r for r in recoveries if r.recovery_success]
        
        if recoveries:
            recovery_rate = len(successful) / len(recoveries)
            # Automata typically around 70%
            assert recovery_rate > 0.4
    
    def test_llm_only_recovery(self):
        """Test LLM-only has good but not optimal recovery rate."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import LLMOnlyCondition
        
        runner = ExperimentRunner(failure_probability=0.9, seed=42)
        condition = LLMOnlyCondition()
        
        results = runner.run_batch("vendor_onboarding", condition, num_runs=50)
        
        recoveries = [r for r in results if r.recovery_attempted]
        successful = [r for r in recoveries if r.recovery_success]
        
        if recoveries:
            recovery_rate = len(successful) / len(recoveries)
            # LLM without peer typically around 68%
            assert recovery_rate > 0.4
    
    def test_recovery_rate_ordering(self):
        """Test that recovery rates follow expected ordering."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import (
            SimpleRetryCondition,
            CheckpointOnlyCondition,
            AutomataOnlyCondition,
            FullSystemCondition,
        )
        
        runner = ExperimentRunner(failure_probability=0.8, seed=42)
        
        # Run experiments for each condition
        conditions = [
            SimpleRetryCondition(),
            CheckpointOnlyCondition(),
            AutomataOnlyCondition(),
            FullSystemCondition(),
        ]
        
        recovery_rates = {}
        for condition in conditions:
            results = runner.run_batch("vendor_onboarding", condition, num_runs=50)
            recoveries = [r for r in results if r.recovery_attempted]
            successful = [r for r in recoveries if r.recovery_success]
            if recoveries:
                recovery_rates[condition.name] = len(successful) / len(recoveries)
            else:
                recovery_rates[condition.name] = 0.0
        
        # Full system should be best
        if recovery_rates.get("full_system", 0) > 0:
            assert recovery_rates["full_system"] >= recovery_rates.get("automata_only", 0)
        
        # Simple retry should be worst (or close to it)
        if recovery_rates.get("simple_retry", 0) > 0 and recovery_rates.get("full_system", 0) > 0:
            assert recovery_rates["simple_retry"] <= recovery_rates["full_system"]


class TestExperimentRunner:
    """Tests for experiment runner."""
    
    def test_run_single_experiment(self):
        """Test running a single experiment."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import BaselineCondition
        
        runner = ExperimentRunner(seed=42)
        condition = BaselineCondition()
        
        result = runner.run_single("vendor_onboarding", condition)
        
        assert result.scenario_name == "Vendor Onboarding"
        assert result.condition_name == "baseline"
        assert result.total_steps == 5
        assert result.run_id is not None
    
    def test_run_batch_experiments(self):
        """Test running batch experiments."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import BaselineCondition
        
        runner = ExperimentRunner(seed=42)
        condition = BaselineCondition()
        
        results = runner.run_batch("vendor_onboarding", condition, num_runs=10)
        
        assert len(results) == 10
        assert all(r.scenario_name == "Vendor Onboarding" for r in results)
    
    def test_run_with_failure_injection(self):
        """Test running with failure injection."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import BaselineCondition
        
        # High failure probability
        runner = ExperimentRunner(failure_probability=0.9, seed=42)
        condition = BaselineCondition()
        
        results = runner.run_batch("vendor_onboarding", condition, num_runs=20)
        
        # Some should have failures
        failures = [r for r in results if r.failure_occurred]
        assert len(failures) > 0
    
    def test_recovery_with_reconstruction(self):
        """Test recovery under reconstruction condition."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import ReconstructionCondition
        
        runner = ExperimentRunner(failure_probability=0.9, seed=42)
        condition = ReconstructionCondition()
        
        results = runner.run_batch("vendor_onboarding", condition, num_runs=30)
        
        # Should have recovery attempts
        recoveries = [r for r in results if r.recovery_attempted]
        assert len(recoveries) > 0
        
        # Some recoveries should succeed
        successful = [r for r in recoveries if r.recovery_success]
        assert len(successful) > 0
    
    def test_recovery_with_full_system(self):
        """Test recovery under full system condition."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import FullSystemCondition
        
        runner = ExperimentRunner(failure_probability=0.9, seed=42)
        condition = FullSystemCondition()
        
        results = runner.run_batch("vendor_onboarding", condition, num_runs=30)
        
        # Full system should have high recovery rate
        recoveries = [r for r in results if r.recovery_attempted]
        successful = [r for r in recoveries if r.recovery_success]
        
        if recoveries:
            recovery_rate = len(successful) / len(recoveries)
            # Full system should have better recovery rate than baseline
            assert recovery_rate > 0.5
    
    def test_experiment_result_structure(self):
        """Test experiment result structure."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import BaselineCondition
        
        runner = ExperimentRunner(seed=42)
        condition = BaselineCondition()
        
        result = runner.run_single("vendor_onboarding", condition)
        
        assert hasattr(result, "run_id")
        assert hasattr(result, "scenario_name")
        assert hasattr(result, "condition_name")
        assert hasattr(result, "success")
        assert hasattr(result, "total_duration_ms")
        assert hasattr(result, "steps_completed")
        assert hasattr(result, "failure_occurred")
        assert hasattr(result, "recovery_attempted")
        assert hasattr(result, "timestamp")
    
    def test_result_to_dict(self):
        """Test result serialization."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import BaselineCondition
        
        runner = ExperimentRunner(seed=42)
        condition = BaselineCondition()
        
        result = runner.run_single("vendor_onboarding", condition)
        data = result.to_dict()
        
        assert "run_id" in data
        assert "scenario_name" in data
        assert "success" in data
    
    def test_run_all_scenarios(self):
        """Test running all scenarios for a condition."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import BaselineCondition
        
        runner = ExperimentRunner(seed=42)
        condition = BaselineCondition()
        
        results = runner.run_all_scenarios(condition, runs_per_scenario=5)
        
        # Should have results for all 4 scenarios
        scenario_names = set(r.scenario_name for r in results)
        assert len(scenario_names) == 4
    
    def test_get_metrics(self):
        """Test getting aggregated metrics."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import BaselineCondition
        
        runner = ExperimentRunner(seed=42)
        condition = BaselineCondition()
        
        runner.run_batch("vendor_onboarding", condition, num_runs=10)
        
        metrics = runner.get_metrics()
        
        assert metrics.total_runs == 10


class TestMetricsCollector:
    """Tests for metrics collection."""
    
    def test_record_result(self):
        """Test recording experiment result."""
        from src.experiments.collector import MetricsCollector
        from src.experiments.runner import ExperimentResult
        
        collector = MetricsCollector()
        
        result = ExperimentResult(
            run_id="test-001",
            scenario_name="Test Scenario",
            condition_name="baseline",
            success=True,
            total_duration_ms=100.0,
            steps_completed=5,
            total_steps=5,
            failure_occurred=False,
        )
        
        collector.record_result(result)
        
        metrics = collector.get_metrics()
        assert metrics.total_runs == 1
        assert metrics.successful_runs == 1
    
    def test_aggregate_mttr(self):
        """Test MTTR aggregation."""
        from src.experiments.collector import MetricsCollector
        from src.experiments.runner import ExperimentResult
        
        collector = MetricsCollector()
        
        # Add results with recovery times
        for i in range(10):
            result = ExperimentResult(
                run_id=f"test-{i:03d}",
                scenario_name="Test",
                condition_name="reconstruction",
                success=True,
                total_duration_ms=100.0,
                steps_completed=5,
                total_steps=5,
                failure_occurred=True,
                recovery_attempted=True,
                recovery_success=True,
                recovery_time_ms=100.0 + i * 10,  # 100-190ms
            )
            collector.record_result(result)
        
        metrics = collector.get_metrics()
        
        assert metrics.mttr_mean is not None
        assert metrics.mttr_p50 is not None
        assert metrics.mttr_p95 is not None
    
    def test_metrics_by_condition(self):
        """Test metrics grouping by condition."""
        from src.experiments.collector import MetricsCollector
        from src.experiments.runner import ExperimentResult
        
        collector = MetricsCollector()
        
        # Add results for different conditions
        for condition in ["baseline", "reconstruction", "full_system"]:
            for i in range(5):
                result = ExperimentResult(
                    run_id=f"{condition}-{i}",
                    scenario_name="Test",
                    condition_name=condition,
                    success=condition != "baseline",  # Baseline fails
                    total_duration_ms=100.0,
                    steps_completed=5 if condition != "baseline" else 3,
                    total_steps=5,
                    failure_occurred=True,
                    recovery_success=condition != "baseline",
                )
                collector.record_result(result)
        
        metrics = collector.get_metrics()
        
        assert "baseline" in metrics.metrics_by_condition
        assert "reconstruction" in metrics.metrics_by_condition
        assert "full_system" in metrics.metrics_by_condition
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        from src.experiments.collector import MetricsCollector
        from src.experiments.runner import ExperimentResult
        
        collector = MetricsCollector()
        
        # 7 successes, 3 failures
        for i in range(10):
            result = ExperimentResult(
                run_id=f"test-{i}",
                scenario_name="Test",
                condition_name="test",
                success=i < 7,
                total_duration_ms=100.0,
                steps_completed=5,
                total_steps=5,
                failure_occurred=False,
            )
            collector.record_result(result)
        
        metrics = collector.get_metrics()
        
        assert abs(metrics.success_rate - 0.7) < 0.01


class TestExperimentExport:
    """Tests for experiment data export."""
    
    def test_export_csv(self):
        """Test exporting to CSV."""
        from src.experiments.export import ExperimentExporter
        from src.experiments.runner import ExperimentResult
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ExperimentExporter(Path(tmpdir))
            
            results = [
                ExperimentResult(
                    run_id=f"test-{i}",
                    scenario_name="Test",
                    condition_name="baseline",
                    success=True,
                    total_duration_ms=100.0,
                    steps_completed=5,
                    total_steps=5,
                    failure_occurred=False,
                )
                for i in range(5)
            ]
            
            path = exporter.export_results_csv(results, "test_results")
            
            assert path.exists()
            assert path.suffix == ".csv"
    
    def test_export_json(self):
        """Test exporting to JSON."""
        from src.experiments.export import ExperimentExporter
        from src.experiments.runner import ExperimentResult
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ExperimentExporter(Path(tmpdir))
            
            results = [
                ExperimentResult(
                    run_id=f"test-{i}",
                    scenario_name="Test",
                    condition_name="baseline",
                    success=True,
                    total_duration_ms=100.0,
                    steps_completed=5,
                    total_steps=5,
                    failure_occurred=False,
                )
                for i in range(5)
            ]
            
            path = exporter.export_results_json(results, "test_results")
            
            assert path.exists()
            assert path.suffix == ".json"
    
    def test_export_comparison(self):
        """Test exporting condition comparison."""
        from src.experiments.export import ExperimentExporter
        from src.experiments.runner import ExperimentResult
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ExperimentExporter(Path(tmpdir))
            
            results_by_condition = {
                "baseline": [
                    ExperimentResult(
                        run_id=f"base-{i}",
                        scenario_name="Test",
                        condition_name="baseline",
                        success=i % 2 == 0,
                        total_duration_ms=100.0,
                        steps_completed=5,
                        total_steps=5,
                        failure_occurred=False,
                    )
                    for i in range(10)
                ],
                "reconstruction": [
                    ExperimentResult(
                        run_id=f"recon-{i}",
                        scenario_name="Test",
                        condition_name="reconstruction",
                        success=True,
                        total_duration_ms=100.0,
                        steps_completed=5,
                        total_steps=5,
                        failure_occurred=True,
                        recovery_success=True,
                        recovery_time_ms=150.0,
                    )
                    for i in range(10)
                ],
            }
            
            path = exporter.export_comparison_csv(results_by_condition)
            
            assert path.exists()
    
    def test_export_pandas(self):
        """Test converting to pandas DataFrame."""
        pytest.importorskip("pandas")
        
        from src.experiments.export import export_to_pandas
        from src.experiments.runner import ExperimentResult
        
        results = [
            ExperimentResult(
                run_id=f"test-{i}",
                scenario_name="Test",
                condition_name="baseline",
                success=True,
                total_duration_ms=100.0,
                steps_completed=5,
                total_steps=5,
                failure_occurred=False,
            )
            for i in range(5)
        ]
        
        df = export_to_pandas(results)
        
        assert len(df) == 5
        assert "run_id" in df.columns
        assert "success" in df.columns


class TestIntegration:
    """Integration tests for full experiment workflow."""
    
    def test_full_experiment_workflow(self):
        """Test complete experiment workflow."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import get_all_conditions
        
        runner = ExperimentRunner(seed=42)
        
        # Run small batch for each condition
        results_by_condition = {}
        for condition in get_all_conditions():
            results = runner.run_batch(
                "vendor_onboarding", condition, num_runs=10
            )
            results_by_condition[condition.name] = results
        
        # Verify we have results for all conditions (7 total now)
        assert "baseline" in results_by_condition
        assert "reconstruction" in results_by_condition
        assert "full_system" in results_by_condition
        # New comparison baselines
        assert "simple_retry" in results_by_condition
        assert "checkpoint_only" in results_by_condition
        assert "automata_only" in results_by_condition
        assert "llm_only" in results_by_condition
        
        # Verify metrics
        metrics = runner.get_metrics()
        assert metrics.total_runs == 70  # 10 * 7 conditions
    
    def test_experiment_reproducibility(self):
        """Test that experiments are reproducible with seed."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import BaselineCondition
        
        runner1 = ExperimentRunner(seed=12345)
        runner2 = ExperimentRunner(seed=12345)
        
        condition = BaselineCondition()
        
        results1 = runner1.run_batch("vendor_onboarding", condition, num_runs=10)
        results2 = runner2.run_batch("vendor_onboarding", condition, num_runs=10)
        
        # Results should be identical
        for r1, r2 in zip(results1, results2):
            assert r1.success == r2.success
            assert r1.failure_occurred == r2.failure_occurred

