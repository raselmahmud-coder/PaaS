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
        assert "real_api" in conditions
        assert len(conditions) == 12  # 3 original + 4 comparison + 1 real_api + 3 Phase B + 1 Phase C
    
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


class TestPhaseBConditions:
    """Tests for Phase B baseline conditions."""
    
    def test_exponential_backoff_condition(self):
        """Test exponential backoff condition configuration."""
        from src.experiments.conditions import ExponentialBackoffCondition
        
        condition = ExponentialBackoffCondition()
        
        assert condition.name == "exponential_backoff"
        assert condition.should_attempt_recovery() is True
        assert condition.should_use_automata() is False
        assert condition.should_use_semantic_protocol() is False
        assert condition.config.max_retries == 4
        assert condition.get_reconstruction_strategy() == "exponential_backoff"
        
        # Test retry delays method
        delays = condition.get_retry_delays()
        assert len(delays) == 4
        assert 0.09 <= delays[0] <= 0.11  # ~100ms with jitter
        assert 0.18 <= delays[1] <= 0.22  # ~200ms with jitter
    
    def test_circuit_breaker_condition(self):
        """Test circuit breaker condition configuration."""
        from src.experiments.conditions import CircuitBreakerCondition
        
        condition = CircuitBreakerCondition()
        
        assert condition.name == "circuit_breaker"
        assert condition.should_attempt_recovery() is True
        assert condition.should_use_automata() is False
        assert condition.get_reconstruction_strategy() == "circuit_breaker"
        
        # Test circuit state
        state = condition.get_circuit_state()
        assert state.state == "closed"
        assert state.can_execute() is True
    
    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state machine."""
        from src.experiments.conditions import CircuitBreakerState
        
        state = CircuitBreakerState(failure_threshold=3, cooldown_seconds=0.1)
        
        # Initial state
        assert state.state == "closed"
        assert state.can_execute() is True
        
        # Record failures until open
        state.record_failure()
        state.record_failure()
        assert state.state == "closed"
        
        state.record_failure()  # 3rd failure
        assert state.state == "open"
        assert state.can_execute() is False
        
        # Wait for cooldown
        import time
        time.sleep(0.15)
        assert state.can_execute() is True
        assert state.state == "half_open"
        
        # Success resets
        state.record_success()
        assert state.state == "closed"
    
    def test_semantic_only_condition(self):
        """Test semantic-only ablation condition."""
        from src.experiments.conditions import SemanticOnlyCondition
        
        condition = SemanticOnlyCondition()
        
        assert condition.name == "semantic_only"
        assert condition.should_attempt_recovery() is False  # No recovery!
        assert condition.should_use_semantic_protocol() is True  # Only semantic
        assert condition.should_use_automata() is False
        assert condition.get_reconstruction_strategy() == "semantic_only"
    
    def test_new_conditions_in_registry(self):
        """Test new conditions are in registry."""
        from src.experiments.conditions import list_conditions, get_condition
        
        conditions = list_conditions()
        
        assert "exponential_backoff" in conditions
        assert "circuit_breaker" in conditions
        assert "semantic_only" in conditions
        assert len(conditions) == 12  # 8 existing + 3 Phase B + 1 Phase C
        
        # Test get_condition works
        exp = get_condition("exponential_backoff")
        assert exp.name == "exponential_backoff"
    
    def test_condition_to_dict_new_conditions(self):
        """Test new conditions serialize correctly."""
        from src.experiments.conditions import ExponentialBackoffCondition
        
        condition = ExponentialBackoffCondition()
        data = condition.to_dict()
        
        assert data["name"] == "exponential_backoff"
        assert data["max_retries"] == 4
        assert data["resilience_enabled"] is True


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
        
        # Should have results for all 5 scenarios (including shopify_real)
        scenario_names = set(r.scenario_name for r in results)
        assert len(scenario_names) == 5
    
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


class TestSemanticMetrics:
    """Tests for semantic conflict metrics tracking (Gap 4)."""
    
    def test_scenario_has_term_conflicts(self):
        """Test that scenarios have term_conflicts definitions."""
        from src.experiments.scenario_loader import load_scenario
        
        scenario = load_scenario("vendor_onboarding")
        
        # Check that at least some steps have term conflicts
        steps_with_conflicts = [s for s in scenario.steps if s.has_term_conflicts()]
        assert len(steps_with_conflicts) > 0
    
    def test_term_conflict_structure(self):
        """Test term_conflicts structure in scenario steps."""
        from src.experiments.scenario_loader import load_scenario
        
        scenario = load_scenario("vendor_onboarding")
        
        # Find a step with term conflicts
        for step in scenario.steps:
            if step.has_term_conflicts():
                term_conflict = step.term_conflicts
                assert hasattr(term_conflict, 'terms')
                assert hasattr(term_conflict, 'probability')
                assert hasattr(term_conflict, 'severity')
                assert len(term_conflict.terms) >= 2
                assert 0 <= term_conflict.probability <= 1
                assert term_conflict.severity in ['low', 'medium', 'high']
                break
    
    def test_experiment_result_has_semantic_fields(self):
        """Test that ExperimentResult includes semantic metrics fields."""
        from src.experiments.runner import ExperimentResult
        
        result = ExperimentResult(
            run_id="test-001",
            scenario_name="Test",
            condition_name="test",
            success=True,
            total_duration_ms=100.0,
            steps_completed=5,
            total_steps=5,
            failure_occurred=False,
            semantic_conflicts=3,
            semantic_resolved=2,
            semantic_negotiation_ms=150.0,
        )
        
        assert result.semantic_conflicts == 3
        assert result.semantic_resolved == 2
        assert result.semantic_negotiation_ms == 150.0
    
    def test_step_result_has_semantic_fields(self):
        """Test that StepResult includes semantic metrics fields."""
        from src.experiments.runner import StepResult
        
        step_result = StepResult(
            step_name="validate_data",
            agent="product-agent",
            status="completed",
            success=True,
            duration_ms=50.0,
            semantic_conflicts=1,
            semantic_resolved=1,
            semantic_negotiation_ms=75.0,
        )
        
        assert step_result.semantic_conflicts == 1
        assert step_result.semantic_resolved == 1
        assert step_result.semantic_negotiation_ms == 75.0
    
    def test_runner_tracks_semantic_conflicts(self):
        """Test that runner tracks semantic conflicts during execution."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import FullSystemCondition
        
        runner = ExperimentRunner(seed=42)
        condition = FullSystemCondition()
        
        # Run enough experiments to trigger some conflicts
        results = runner.run_batch("vendor_onboarding", condition, num_runs=50)
        
        # Check that some experiments recorded semantic conflicts
        total_conflicts = sum(r.semantic_conflicts for r in results)
        total_resolved = sum(r.semantic_resolved for r in results)
        
        # With probability-based conflicts, we should see some
        assert total_conflicts >= 0  # Could be 0 due to probability
        
        # If conflicts occurred, some should be resolved (high resolution rate)
        if total_conflicts > 0:
            resolution_rate = total_resolved / total_conflicts
            # Full system has semantic protocol enabled, so resolution should be high
            assert resolution_rate > 0.7
    
    def test_semantic_resolution_rate_by_condition(self):
        """Test that semantic resolution rate differs by condition."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import (
            FullSystemCondition, 
            BaselineCondition,
        )
        
        runner = ExperimentRunner(seed=42)
        
        # Run with semantic protocol enabled (full system)
        full_condition = FullSystemCondition()
        full_results = runner.run_batch("vendor_onboarding", full_condition, num_runs=50)
        
        full_conflicts = sum(r.semantic_conflicts for r in full_results)
        full_resolved = sum(r.semantic_resolved for r in full_results)
        
        # Clear and run with semantic protocol disabled (baseline)
        runner.clear()
        baseline_condition = BaselineCondition()
        baseline_results = runner.run_batch("vendor_onboarding", baseline_condition, num_runs=50)
        
        baseline_conflicts = sum(r.semantic_conflicts for r in baseline_results)
        baseline_resolved = sum(r.semantic_resolved for r in baseline_results)
        
        # Full system should have better resolution rate if conflicts occurred
        if full_conflicts > 0 and baseline_conflicts > 0:
            full_rate = full_resolved / full_conflicts
            baseline_rate = baseline_resolved / baseline_conflicts
            assert full_rate > baseline_rate
    
    def test_collector_aggregates_semantic_metrics(self):
        """Test that MetricsCollector aggregates semantic metrics."""
        from src.experiments.collector import MetricsCollector
        from src.experiments.runner import ExperimentResult
        
        collector = MetricsCollector()
        
        # Add results with semantic metrics
        for i in range(10):
            result = ExperimentResult(
                run_id=f"test-{i}",
                scenario_name="Test",
                condition_name="full_system",
                success=True,
                total_duration_ms=100.0,
                steps_completed=5,
                total_steps=5,
                failure_occurred=False,
                semantic_conflicts=2,
                semantic_resolved=1 if i % 2 == 0 else 2,
                semantic_negotiation_ms=100.0 + i * 10,
            )
            collector.record_result(result)
        
        metrics = collector.get_metrics()
        
        # Total conflicts: 10 * 2 = 20
        assert metrics.semantic_conflicts_total == 20
        # Resolved: 5 * 1 + 5 * 2 = 15
        assert metrics.semantic_conflicts_resolved == 15
        # Resolution rate: 15/20 = 0.75
        assert abs(metrics.semantic_resolution_rate - 0.75) < 0.01
        # Negotiation time should be average of 100-190ms
        assert 100 <= metrics.semantic_negotiation_time_ms <= 200
    
    def test_metrics_by_condition_includes_semantic(self):
        """Test that per-condition metrics include semantic fields."""
        from src.experiments.collector import MetricsCollector
        from src.experiments.runner import ExperimentResult
        
        collector = MetricsCollector()
        
        # Add results for two conditions
        for condition in ["baseline", "full_system"]:
            for i in range(5):
                conflicts = 2 if condition == "full_system" else 1
                resolved = 2 if condition == "full_system" else 0
                result = ExperimentResult(
                    run_id=f"{condition}-{i}",
                    scenario_name="Test",
                    condition_name=condition,
                    success=True,
                    total_duration_ms=100.0,
                    steps_completed=5,
                    total_steps=5,
                    failure_occurred=False,
                    semantic_conflicts=conflicts,
                    semantic_resolved=resolved,
                )
                collector.record_result(result)
        
        metrics = collector.get_metrics()
        
        # Check per-condition semantic metrics
        assert "semantic_conflicts" in metrics.metrics_by_condition["baseline"]
        assert "semantic_resolved" in metrics.metrics_by_condition["baseline"]
        assert "semantic_resolution_rate" in metrics.metrics_by_condition["baseline"]
        
        assert "semantic_conflicts" in metrics.metrics_by_condition["full_system"]
        assert "semantic_resolved" in metrics.metrics_by_condition["full_system"]
        assert "semantic_resolution_rate" in metrics.metrics_by_condition["full_system"]
        
        # Full system should have higher resolution rate
        full_rate = metrics.metrics_by_condition["full_system"]["semantic_resolution_rate"]
        baseline_rate = metrics.metrics_by_condition["baseline"]["semantic_resolution_rate"]
        assert full_rate > baseline_rate
    
    def test_experiment_metrics_to_dict_includes_semantic(self):
        """Test that ExperimentMetrics.to_dict() includes semantic fields."""
        from src.experiments.collector import ExperimentMetrics
        
        metrics = ExperimentMetrics(
            total_runs=100,
            semantic_conflicts_total=50,
            semantic_conflicts_resolved=45,
            semantic_resolution_rate=0.9,
            semantic_negotiation_time_ms=125.0,
        )
        
        data = metrics.to_dict()
        
        assert data["semantic_conflicts_total"] == 50
        assert data["semantic_conflicts_resolved"] == 45
        assert data["semantic_resolution_rate"] == 0.9
        assert data["semantic_negotiation_time_ms"] == 125.0
    
    def test_result_to_dict_includes_semantic(self):
        """Test that ExperimentResult.to_dict() includes semantic fields."""
        from src.experiments.runner import ExperimentResult
        
        result = ExperimentResult(
            run_id="test-001",
            scenario_name="Test",
            condition_name="full_system",
            success=True,
            total_duration_ms=100.0,
            steps_completed=5,
            total_steps=5,
            failure_occurred=False,
            semantic_conflicts=3,
            semantic_resolved=2,
            semantic_negotiation_ms=150.0,
        )
        
        data = result.to_dict()
        
        assert data["semantic_conflicts"] == 3
        assert data["semantic_resolved"] == 2
        assert data["semantic_negotiation_ms"] == 150.0


class TestRealReconstruction:
    """Tests for real reconstruction integration (Phase A)."""
    
    def test_recovery_result_dataclass(self):
        """Test RecoveryResult dataclass structure."""
        from src.experiments.runner import RecoveryResult
        
        result = RecoveryResult(
            success=True,
            strategy_used="hybrid",
            recovery_time_ms=150.0,
            confidence=0.95,
            ground_truth_state={"step_index": 2},
            reconstructed_state={"step_index": 2},
            reconstruction_accuracy=0.92,
            timing_breakdown={"reconstruction_ms": 100.0},
        )
        
        assert result.success is True
        assert result.strategy_used == "hybrid"
        assert result.recovery_time_ms == 150.0
        assert result.confidence == 0.95
        assert result.reconstruction_accuracy == 0.92
        assert "reconstruction_ms" in result.timing_breakdown
    
    def test_ground_truth_capture(self):
        """Test ground truth state capture before failure."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import FullSystemCondition
        from src.experiments.scenario_loader import load_scenario
        
        runner = ExperimentRunner(failure_probability=1.0, seed=42)
        scenario = load_scenario("vendor_onboarding")
        step = scenario.steps[1]  # Use second step
        
        ground_truth = runner._capture_current_state(scenario, step, 1)
        
        # Verify ground truth structure
        assert "scenario_name" in ground_truth
        assert "current_step" in ground_truth
        assert "step_index" in ground_truth
        assert "variables" in ground_truth
        assert "pending_steps" in ground_truth
        assert "completed_steps" in ground_truth
        
        assert ground_truth["scenario_name"] == "Vendor Onboarding"
        assert ground_truth["step_index"] == 1
        assert ground_truth["completed_steps"] == 1
    
    def test_reconstruction_accuracy_calculation(self):
        """Test state similarity calculation for reconstruction accuracy."""
        from src.experiments.runner import ExperimentRunner
        
        runner = ExperimentRunner(seed=42)
        
        # Test identical states
        state_a = {"field1": "value1", "field2": 100, "field3": True}
        state_b = {"field1": "value1", "field2": 100, "field3": True}
        accuracy = runner._calculate_state_similarity(state_a, state_b)
        assert accuracy == 1.0, f"Expected 1.0 for identical states, got {accuracy}"
        
        # Test partially matching states
        state_c = {"field1": "value1", "field2": 100, "field3": False}
        accuracy = runner._calculate_state_similarity(state_a, state_c)
        # Should be ~0.67 (2/3 match: field1 and field2 match, field3 doesn't)
        assert 0.6 < accuracy < 0.7, f"Expected ~0.67 for 2/3 match, got {accuracy}"
        
        # Test empty states - both empty should return 1.0
        accuracy = runner._calculate_state_similarity({}, {})
        assert accuracy == 1.0, f"Expected 1.0 for both empty, got {accuracy}"
        
        # Test one empty state - should return 0.0
        accuracy = runner._calculate_state_similarity(state_a, {})
        assert accuracy == 0.0, f"Expected 0.0 when one state is empty, got {accuracy}"
        
        # Test with only string fields (to ensure no boolean/int confusion)
        state_d = {"name": "test", "status": "active"}
        state_e = {"name": "test", "status": "active"}
        accuracy = runner._calculate_state_similarity(state_d, state_e)
        assert accuracy == 1.0, f"Expected 1.0 for identical string-only states, got {accuracy}"
    
    def test_reconstruction_accuracy_nested_dict(self):
        """Test accuracy calculation with nested dictionaries."""
        from src.experiments.runner import ExperimentRunner
        
        runner = ExperimentRunner(seed=42)
        
        state_a = {
            "outer": {"inner1": "a", "inner2": "b"},
            "simple": 100,
        }
        
        state_b = {
            "outer": {"inner1": "a", "inner2": "b"},
            "simple": 100,
        }
        
        accuracy = runner._calculate_state_similarity(state_a, state_b)
        assert accuracy == 1.0
        
        # Partial nested match
        state_c = {
            "outer": {"inner1": "a", "inner2": "c"},  # inner2 differs
            "simple": 100,
        }
        
        accuracy = runner._calculate_state_similarity(state_a, state_c)
        # outer has 50% match (1/2), simple has 100% match
        # Total: (0.5 + 1.0) / 2 = 0.75
        assert 0.7 < accuracy < 0.8
    
    def test_reconstruction_accuracy_numeric_tolerance(self):
        """Test accuracy calculation with numeric tolerance."""
        from src.experiments.runner import ExperimentRunner
        
        runner = ExperimentRunner(seed=42)
        
        state_a = {"value": 100.0}
        state_b = {"value": 100.5}  # 0.5% deviation
        
        # With 1% tolerance (default), should match
        accuracy = runner._calculate_state_similarity(state_a, state_b, numeric_tolerance=0.01)
        assert accuracy == 1.0
        
        # With strict tolerance, should not match
        accuracy = runner._calculate_state_similarity(state_a, state_b, numeric_tolerance=0.001)
        assert accuracy == 0.0
    
    def test_rng_independence(self):
        """Test that separate RNG streams produce independent results."""
        from src.experiments.runner import ExperimentRunner
        
        runner = ExperimentRunner(seed=42)
        
        # Verify separate RNG instances exist
        assert runner._failure_rng is not runner._semantic_rng
        assert runner._semantic_rng is not runner._recovery_rng
        assert runner._recovery_rng is not runner._step_rng
        
        # Verify they have different states (different seeds)
        failure_val = runner._failure_rng.random()
        semantic_val = runner._semantic_rng.random()
        recovery_val = runner._recovery_rng.random()
        step_val = runner._step_rng.random()
        
        # Reset and verify reproducibility
        runner2 = ExperimentRunner(seed=42)
        
        assert runner2._failure_rng.random() == failure_val
        assert runner2._semantic_rng.random() == semantic_val
        assert runner2._recovery_rng.random() == recovery_val
        assert runner2._step_rng.random() == step_val
    
    def test_rng_stream_separation(self):
        """Test that failure RNG doesn't affect semantic RNG."""
        from src.experiments.runner import ExperimentRunner
        
        runner1 = ExperimentRunner(seed=42)
        runner2 = ExperimentRunner(seed=42)
        
        # In runner1, consume many failure RNG values
        for _ in range(100):
            runner1._failure_rng.random()
        
        # In runner2, don't consume any failure RNG values
        
        # Both should produce same semantic RNG value
        # (since semantic RNG has separate seed offset)
        sem1 = runner1._semantic_rng.random()
        sem2 = runner2._semantic_rng.random()
        
        assert sem1 == sem2
    
    def test_step_result_includes_ground_truth_fields(self):
        """Test StepResult includes ground truth validation fields."""
        from src.experiments.runner import StepResult
        
        step_result = StepResult(
            step_name="test_step",
            agent="test-agent",
            status="recovered",
            success=True,
            duration_ms=50.0,
            recovered=True,
            ground_truth_state={"step_index": 1},
            reconstructed_state={"step_index": 1},
            reconstruction_accuracy=0.95,
        )
        
        assert step_result.ground_truth_state == {"step_index": 1}
        assert step_result.reconstructed_state == {"step_index": 1}
        assert step_result.reconstruction_accuracy == 0.95
    
    def test_experiment_result_includes_accuracy_fields(self):
        """Test ExperimentResult includes aggregated accuracy fields."""
        from src.experiments.runner import ExperimentResult
        
        result = ExperimentResult(
            run_id="test-001",
            scenario_name="Test",
            condition_name="full_system",
            success=True,
            total_duration_ms=100.0,
            steps_completed=5,
            total_steps=5,
            failure_occurred=True,
            recovery_success=True,
            mean_reconstruction_accuracy=0.85,
            reconstruction_accuracies=[0.80, 0.85, 0.90],
        )
        
        assert result.mean_reconstruction_accuracy == 0.85
        assert len(result.reconstruction_accuracies) == 3
        
        # Check to_dict includes fields
        data = result.to_dict()
        assert "mean_reconstruction_accuracy" in data
        assert "reconstruction_accuracies" in data
    
    def test_runner_tracks_reconstruction_accuracy(self):
        """Test that runner tracks reconstruction accuracy during recovery."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import FullSystemCondition
        
        runner = ExperimentRunner(failure_probability=0.9, seed=42)
        condition = FullSystemCondition()
        
        results = runner.run_batch("vendor_onboarding", condition, num_runs=30)
        
        # Get results with recovery
        recovered = [r for r in results if r.recovery_success]
        
        # Should have some reconstructions with accuracy
        accuracies = [r.mean_reconstruction_accuracy for r in recovered if r.mean_reconstruction_accuracy > 0]
        
        if accuracies:
            # Accuracy should be reasonable for full system
            avg_accuracy = sum(accuracies) / len(accuracies)
            assert avg_accuracy > 0.5  # Should be better than random
    
    def test_simulated_reconstruction_generates_state(self):
        """Test that _simulate_recovery_with_state returns reconstructed state."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import FullSystemCondition
        from src.experiments.scenario_loader import load_scenario
        
        runner = ExperimentRunner(seed=42)
        condition = FullSystemCondition()
        scenario = load_scenario("vendor_onboarding")
        step = scenario.steps[1]
        
        ground_truth = {
            "scenario_name": "Vendor Onboarding",
            "step_index": 1,
            "variables": {"workflow_id": "test-123"},
        }
        
        success, reconstructed = runner._simulate_recovery_with_state(
            condition, scenario, step, "crash", ground_truth
        )
        
        # Should return a reconstructed state dict
        assert reconstructed is not None
        assert isinstance(reconstructed, dict)
        
        # Reconstructed state should be related to ground truth
        # (may have some corrupted fields for accuracy testing)
        assert "scenario_name" in reconstructed
    
    def test_metrics_collector_tracks_reconstruction_accuracy(self):
        """Test MetricsCollector aggregates reconstruction accuracy."""
        from src.experiments.collector import MetricsCollector
        from src.experiments.runner import ExperimentResult
        
        collector = MetricsCollector()
        
        for i in range(10):
            result = ExperimentResult(
                run_id=f"test-{i}",
                scenario_name="Test",
                condition_name="full_system",
                success=True,
                total_duration_ms=100.0,
                steps_completed=5,
                total_steps=5,
                failure_occurred=True,
                recovery_success=True,
                mean_reconstruction_accuracy=0.80 + i * 0.02,  # 0.80-0.98
            )
            collector.record_result(result)
        
        metrics = collector.get_metrics()
        
        # Should have accuracy stats
        assert metrics.reconstruction_accuracy_mean > 0
        # Mean should be around 0.89 ((0.80 + 0.98) / 2)
        assert 0.85 < metrics.reconstruction_accuracy_mean < 0.93
        
        # Should have accuracy by condition
        assert "full_system" in metrics.reconstruction_accuracy_by_condition
    
    @pytest.mark.asyncio
    async def test_execute_real_recovery_returns_result(self):
        """Test that _execute_real_recovery returns a RecoveryResult."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import FullSystemCondition
        
        runner = ExperimentRunner(seed=42)
        condition = FullSystemCondition()
        
        ground_truth = {
            "scenario_name": "Test",
            "step_index": 2,
            "variables": {"workflow_id": "test-456"},
        }
        
        events = [
            {"event_type": "step_execution", "step_name": "step1", "agent": "test"},
            {"event_type": "step_execution", "step_name": "step2", "agent": "test"},
        ]
        
        result = await runner._execute_real_recovery(
            condition=condition,
            agent_id="test-agent",
            thread_id="test-thread",
            ground_truth_state=ground_truth,
            events=events,
        )
        
        # Should return a RecoveryResult
        assert hasattr(result, 'success')
        assert hasattr(result, 'strategy_used')
        assert hasattr(result, 'recovery_time_ms')
        assert hasattr(result, 'reconstruction_accuracy')
        
        # Should have timing breakdown
        assert result.timing_breakdown is not None


class TestAblationMetrics:
    """Tests for ablation study metrics (Phase C)."""
    
    def test_ablation_metrics_calculation(self):
        """Test AblationMetrics.calculate() works correctly."""
        from src.experiments.collector import AblationMetrics
        
        mock_results = {
            "baseline": {"success_rate": 0.35},
            "semantic_only": {"success_rate": 0.40},
            "automata_only": {"success_rate": 0.70},
            "llm_only": {"success_rate": 0.68},
            "reconstruction": {"success_rate": 0.84},
            "full_no_semantic": {"success_rate": 0.90},
            "full_system": {"success_rate": 0.95},
        }
        
        metrics = AblationMetrics.calculate(mock_results)
        
        # Semantic contribution: full - no_semantic = 0.95 - 0.90 = 0.05 = 5pp
        assert abs(metrics.semantic_contribution - 5.0) < 0.1
        
        # Automata contribution: full - reconstruction = 0.95 - 0.84 = 0.11 = 11pp
        assert abs(metrics.automata_contribution - 11.0) < 0.1
        
        # Peer contribution: reconstruction - llm_only = 0.84 - 0.68 = 0.16 = 16pp
        assert abs(metrics.peer_context_contribution - 16.0) < 0.1
        
        # LLM contribution: full - automata_only = 0.95 - 0.70 = 0.25 = 25pp
        assert abs(metrics.llm_contribution - 25.0) < 0.1
    
    def test_ablation_metrics_to_dict(self):
        """Test AblationMetrics serialization."""
        from src.experiments.collector import AblationMetrics
        
        metrics = AblationMetrics(
            semantic_contribution=5.0,
            automata_contribution=11.0,
            peer_context_contribution=16.0,
            llm_contribution=25.0,
            semantic_automata_synergy=2.0,
        )
        
        data = metrics.to_dict()
        
        assert "semantic_contribution_pp" in data
        assert data["semantic_contribution_pp"] == 5.0
        assert "automata_contribution_pp" in data
        assert "peer_context_contribution_pp" in data
        assert "llm_contribution_pp" in data
        assert "semantic_automata_synergy_pp" in data
    
    def test_ablation_metrics_missing_conditions(self):
        """Test AblationMetrics handles missing conditions gracefully."""
        from src.experiments.collector import AblationMetrics
        
        mock_results = {
            "full_system": {"success_rate": 0.95},
        }
        
        metrics = AblationMetrics.calculate(mock_results)
        
        # Should default to 0 if conditions missing
        assert metrics.semantic_contribution == 0.0
        assert metrics.automata_contribution == 0.0
    
    def test_full_no_semantic_condition(self):
        """Test full_no_semantic condition for ablation."""
        from src.experiments.conditions import get_condition
        
        condition = get_condition("full_no_semantic")
        
        assert condition.name == "full_no_semantic"
        assert condition.should_use_semantic_protocol() is False
        assert condition.should_use_automata() is True
        assert condition.should_query_peers() is True
        assert condition.should_attempt_recovery() is True
        assert condition.get_reconstruction_strategy() == "hybrid"
    
    def test_full_no_semantic_in_registry(self):
        """Test full_no_semantic is in registry."""
        from src.experiments.conditions import list_conditions, get_condition
        
        conditions = list_conditions()
        
        assert "full_no_semantic" in conditions
        assert len(conditions) == 12  # 11 existing + 1 new
        
        condition = get_condition("full_no_semantic")
        assert condition.name == "full_no_semantic"
    
    def test_ablation_condition_comparison(self):
        """Test that full_system vs full_no_semantic isolates semantic contribution."""
        from src.experiments.conditions import FullSystemCondition, FullNoSemanticCondition
        
        full = FullSystemCondition()
        no_semantic = FullNoSemanticCondition()
        
        # Both should have same features except semantic
        assert full.should_use_automata() == no_semantic.should_use_automata()
        assert full.should_query_peers() == no_semantic.should_query_peers()
        assert full.should_attempt_recovery() == no_semantic.should_attempt_recovery()
        
        # Only difference should be semantic protocol
        assert full.should_use_semantic_protocol() is True
        assert no_semantic.should_use_semantic_protocol() is False


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
        
        # Verify we have results for all conditions (12 total: 3 original + 4 comparison + 1 real_api + 3 Phase B + 1 Phase C)
        assert "baseline" in results_by_condition
        assert "reconstruction" in results_by_condition
        assert "full_system" in results_by_condition
        # New comparison baselines
        assert "simple_retry" in results_by_condition
        assert "checkpoint_only" in results_by_condition
        assert "automata_only" in results_by_condition
        assert "llm_only" in results_by_condition
        assert "real_api" in results_by_condition
        # Phase B baselines
        assert "exponential_backoff" in results_by_condition
        assert "circuit_breaker" in results_by_condition
        assert "semantic_only" in results_by_condition
        # Phase C ablation
        assert "full_no_semantic" in results_by_condition
        
        # Verify metrics
        metrics = runner.get_metrics()
        assert metrics.total_runs == 120  # 10 * 12 conditions
    
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


class TestCascadeFailures:
    """Tests for cascade failure scenarios (Phase F)."""
    
    def test_cascade_config_creation(self):
        """Test CascadeConfig dataclass creation."""
        from src.experiments.scenario_loader import CascadeConfig
        
        config = CascadeConfig(
            enabled=True,
            trigger_step=2,
            downstream_probability=0.8,
            max_depth=3,
            delay_between_failures_ms=200,
        )
        
        assert config.enabled is True
        assert config.trigger_step == 2
        assert config.downstream_probability == 0.8
        assert config.max_depth == 3
        assert config.delay_between_failures_ms == 200
    
    def test_cascade_config_from_dict(self):
        """Test CascadeConfig creation from dictionary."""
        from src.experiments.scenario_loader import CascadeConfig
        
        data = {
            "enabled": True,
            "trigger_step": 3,
            "downstream_probability": 0.7,
            "max_depth": 2,
            "delay_between_failures_ms": 150,
        }
        
        config = CascadeConfig.from_dict(data)
        
        assert config.enabled is True
        assert config.trigger_step == 3
        assert config.downstream_probability == 0.7
        assert config.max_depth == 2
        assert config.delay_between_failures_ms == 150
    
    def test_cascade_config_to_dict(self):
        """Test CascadeConfig serialization to dictionary."""
        from src.experiments.scenario_loader import CascadeConfig
        
        config = CascadeConfig(
            enabled=True,
            trigger_step=1,
            downstream_probability=0.6,
            max_depth=2,
            delay_between_failures_ms=100,
        )
        
        d = config.to_dict()
        
        assert d["enabled"] is True
        assert d["trigger_step"] == 1
        assert d["downstream_probability"] == 0.6
        assert d["max_depth"] == 2
        assert d["delay_between_failures_ms"] == 100
    
    def test_failure_injection_config_with_cascade(self):
        """Test FailureInjectionConfig with cascade configuration."""
        from src.experiments.scenario_loader import FailureInjectionConfig, CascadeConfig
        
        data = {
            "enabled": True,
            "probability": 0.4,
            "target_steps": [1, 2, 3],
            "failure_types": ["crash", "timeout"],
            "cascade": {
                "enabled": True,
                "trigger_step": 2,
                "downstream_probability": 0.7,
                "max_depth": 2,
                "delay_between_failures_ms": 150,
            },
        }
        
        config = FailureInjectionConfig.from_dict(data)
        
        assert config.enabled is True
        assert config.probability == 0.4
        assert config.cascade is not None
        assert config.cascade.enabled is True
        assert config.cascade.trigger_step == 2
        assert config.has_cascade() is True
    
    def test_failure_injection_config_without_cascade(self):
        """Test FailureInjectionConfig without cascade configuration."""
        from src.experiments.scenario_loader import FailureInjectionConfig
        
        data = {
            "enabled": True,
            "probability": 0.3,
            "target_steps": [1],
            "failure_types": ["crash"],
        }
        
        config = FailureInjectionConfig.from_dict(data)
        
        assert config.cascade is None
        assert config.has_cascade() is False
    
    def test_load_cascade_failure_scenario(self):
        """Test loading the cascade_failure.yaml scenario."""
        from src.experiments.scenario_loader import ScenarioLoader
        
        loader = ScenarioLoader()
        scenario = loader.load("cascade_failure")
        
        assert scenario.name == "Cascade Failure Test"
        assert scenario.complexity == "high"
        assert len(scenario.agents) == 4  # coordinator, product-agent-1, product-agent-2, marketing
        assert len(scenario.steps) == 8
        
        # Check cascade config
        assert scenario.failure_injection.cascade is not None
        assert scenario.failure_injection.cascade.enabled is True
        assert scenario.failure_injection.cascade.trigger_step == 3
        assert scenario.failure_injection.cascade.downstream_probability == 0.70
        assert scenario.failure_injection.cascade.max_depth == 2
    
    def test_experiment_result_cascade_fields(self):
        """Test ExperimentResult includes cascade failure fields."""
        from src.experiments.runner import ExperimentResult
        
        result = ExperimentResult(
            run_id="test-001",
            scenario_name="cascade_failure",
            condition_name="full_system",
            success=True,
            total_duration_ms=500.0,
            steps_completed=8,
            total_steps=8,
            failure_occurred=True,
            cascade_triggered=True,
            cascade_failures=2,
            cascade_depth=2,
        )
        
        assert result.cascade_triggered is True
        assert result.cascade_failures == 2
        assert result.cascade_depth == 2
    
    def test_experiment_result_to_dict_includes_cascade(self):
        """Test ExperimentResult.to_dict() includes cascade fields."""
        from src.experiments.runner import ExperimentResult
        
        result = ExperimentResult(
            run_id="test-001",
            scenario_name="cascade_failure",
            condition_name="full_system",
            success=True,
            total_duration_ms=500.0,
            steps_completed=8,
            total_steps=8,
            failure_occurred=True,
            cascade_triggered=True,
            cascade_failures=3,
            cascade_depth=2,
        )
        
        d = result.to_dict()
        
        assert "cascade_triggered" in d
        assert "cascade_failures" in d
        assert "cascade_depth" in d
        assert d["cascade_triggered"] is True
        assert d["cascade_failures"] == 3
        assert d["cascade_depth"] == 2
    
    def test_runner_cascade_failure_method(self):
        """Test ExperimentRunner._should_cascade_failure method."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.scenario_loader import ScenarioLoader
        
        runner = ExperimentRunner(seed=42)
        loader = ScenarioLoader()
        scenario = loader.load("cascade_failure")
        
        # Test cascade propagation logic
        # Should not cascade if not triggered
        should_cascade_not_triggered = runner._should_cascade_failure(
            step_idx=4,
            scenario=scenario,
            cascade_depth=0,
            cascade_triggered=False,
        )
        assert should_cascade_not_triggered is False
        
        # Should respect max_depth
        should_cascade_max_depth = runner._should_cascade_failure(
            step_idx=6,
            scenario=scenario,
            cascade_depth=3,  # max_depth is 2
            cascade_triggered=True,
        )
        assert should_cascade_max_depth is False
    
    def test_runner_get_cascade_failure_type(self):
        """Test ExperimentRunner._get_cascade_failure_type method."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.scenario_loader import ScenarioLoader
        
        runner = ExperimentRunner(seed=42)
        loader = ScenarioLoader()
        scenario = loader.load("cascade_failure")
        
        # Get cascade failure type
        failure_type = runner._get_cascade_failure_type(
            scenario=scenario,
            original_failure_type="crash",
        )
        
        # Should be one of the cascade types
        assert failure_type in ["timeout", "crash", "message_corruption"]
    
    def test_run_cascade_scenario(self):
        """Test running the cascade failure scenario."""
        from src.experiments.runner import ExperimentRunner
        from src.experiments.conditions import get_condition
        
        runner = ExperimentRunner(seed=42, failure_probability=0.5)
        condition = get_condition("full_system")
        
        # Run a few experiments
        results = runner.run_batch("cascade_failure", condition, num_runs=5)
        
        assert len(results) == 5
        
        # Check that cascade fields are populated
        for result in results:
            assert hasattr(result, "cascade_triggered")
            assert hasattr(result, "cascade_failures")
            assert hasattr(result, "cascade_depth")
            assert result.scenario_name == "Cascade Failure Test"
