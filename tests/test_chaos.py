"""Tests for chaos engineering framework."""

import os
import time
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.chaos.exceptions import (
    ChaosException,
    AgentCrashException,
    AgentTimeoutException,
    MessageCorruptionException,
    HallucinationException,
)
from src.chaos.config import (
    ChaosConfig,
    get_chaos_config,
    set_chaos_config,
    reset_chaos_config,
    enable_chaos,
    disable_chaos,
)
from src.chaos.decorators import (
    inject_crash,
    inject_delay,
    inject_timeout,
    inject_hallucination,
    inject_message_corruption,
    with_chaos,
)
from src.chaos.scenarios import (
    ChaosScenario,
    ScenarioResult,
    HandoffCorruption,
    DelayedRecovery,
    list_scenarios,
    get_scenario,
)
from src.chaos.runner import ChaosRunner, RunnerConfig, RunnerSummary
from src.chaos.metrics import (
    ResilienceMetrics,
    RecoveryEvent,
    MetricsSummary,
    get_metrics,
    reset_metrics,
)
from src.chaos.export import MetricsExporter, quick_export


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def chaos_config():
    """Provide a test chaos configuration."""
    config = ChaosConfig(
        enabled=True,
        crash_probability=0.5,
        delay_ms=100,
        delay_probability=0.5,
        timeout_seconds=1.0,
        timeout_probability=0.5,
    )
    set_chaos_config(config)
    yield config
    reset_chaos_config()


@pytest.fixture
def disabled_chaos_config():
    """Provide a disabled chaos configuration."""
    config = ChaosConfig(enabled=False)
    set_chaos_config(config)
    yield config
    reset_chaos_config()


@pytest.fixture
def metrics():
    """Provide a fresh metrics instance."""
    reset_metrics()
    m = get_metrics()
    m.start_measurement()
    yield m
    reset_metrics()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary output directory."""
    output_dir = tmp_path / "metrics"
    output_dir.mkdir()
    return output_dir


# =============================================================================
# Exception Tests
# =============================================================================


class TestChaosExceptions:
    """Tests for chaos exception types."""

    def test_base_exception(self):
        """Test base ChaosException."""
        exc = ChaosException("test error", fault_type="test")
        assert str(exc) == "test error"
        assert exc.fault_type == "test"

    def test_crash_exception(self):
        """Test AgentCrashException."""
        exc = AgentCrashException()
        assert "crash" in exc.message.lower()
        assert exc.fault_type == "crash"

    def test_timeout_exception(self):
        """Test AgentTimeoutException."""
        exc = AgentTimeoutException(timeout_seconds=30)
        assert exc.fault_type == "timeout"
        assert exc.timeout_seconds == 30

    def test_corruption_exception(self):
        """Test MessageCorruptionException."""
        exc = MessageCorruptionException(corrupted_field="payload")
        assert exc.fault_type == "corruption"
        assert exc.corrupted_field == "payload"

    def test_hallucination_exception(self):
        """Test HallucinationException."""
        exc = HallucinationException()
        assert exc.fault_type == "hallucination"


# =============================================================================
# Configuration Tests
# =============================================================================


class TestChaosConfig:
    """Tests for chaos configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        reset_chaos_config()
        config = get_chaos_config()
        
        assert config.enabled is False  # Disabled by default
        assert 0 <= config.crash_probability <= 1
        assert config.delay_ms >= 0

    def test_probability_clamping(self):
        """Test probability values are clamped to [0, 1]."""
        config = ChaosConfig(
            crash_probability=2.0,  # Over 1
            delay_probability=-0.5,  # Under 0
        )
        
        assert config.crash_probability == 1.0
        assert config.delay_probability == 0.0

    def test_enable_disable_chaos(self, disabled_chaos_config):
        """Test enabling and disabling chaos."""
        assert disabled_chaos_config.enabled is False
        
        enable_chaos()
        assert get_chaos_config().enabled is True
        
        disable_chaos()
        assert get_chaos_config().enabled is False


# =============================================================================
# Decorator Tests
# =============================================================================


class TestFaultDecorators:
    """Tests for fault injection decorators."""

    def test_crash_decorator_disabled(self, disabled_chaos_config):
        """Test crash decorator doesn't trigger when chaos disabled."""
        @inject_crash(probability=1.0)
        def should_not_crash():
            return "success"
        
        result = should_not_crash()
        assert result == "success"

    def test_crash_decorator_enabled(self, chaos_config):
        """Test crash decorator triggers with high probability."""
        chaos_config.crash_probability = 1.0
        
        @inject_crash(probability=1.0)
        def will_crash():
            return "success"
        
        with pytest.raises(AgentCrashException):
            will_crash()

    def test_delay_decorator_disabled(self, disabled_chaos_config):
        """Test delay decorator doesn't add delay when disabled."""
        @inject_delay(delay_ms=1000, probability=1.0)
        def fast_function():
            return "done"
        
        start = time.time()
        result = fast_function()
        elapsed = time.time() - start
        
        assert result == "done"
        assert elapsed < 0.5  # Should be fast

    def test_delay_decorator_enabled(self, chaos_config):
        """Test delay decorator adds latency."""
        @inject_delay(delay_ms=100, probability=1.0)
        def slow_function():
            return "done"
        
        start = time.time()
        result = slow_function()
        elapsed = time.time() - start
        
        assert result == "done"
        assert elapsed >= 0.1  # Should have delay

    def test_timeout_decorator_raises(self, chaos_config):
        """Test timeout decorator raises exception."""
        @inject_timeout(timeout_seconds=0.1, probability=1.0, block=False)
        def will_timeout():
            return "success"
        
        with pytest.raises(AgentTimeoutException) as exc_info:
            will_timeout()
        
        assert exc_info.value.timeout_seconds == 0.1

    def test_hallucination_decorator_raises(self, chaos_config):
        """Test hallucination decorator raises exception."""
        @inject_hallucination(probability=1.0)
        def llm_call():
            return "real response"
        
        with pytest.raises(HallucinationException):
            llm_call()

    def test_hallucination_decorator_returns_fake(self, chaos_config):
        """Test hallucination decorator returns fake response."""
        @inject_hallucination(
            probability=1.0,
            responses=["fake"],
            return_instead=True,
        )
        def llm_call():
            return "real response"
        
        result = llm_call()
        assert result.content == "fake"

    def test_corruption_decorator_corrupts_data(self, chaos_config):
        """Test message corruption decorator."""
        @inject_message_corruption(
            probability=1.0,
            fields=["value"],
            corruption_type="nullify",
        )
        def process_message(msg):
            return msg
        
        result = process_message({"value": "original", "other": "keep"})
        assert result["value"] is None
        assert result["other"] == "keep"

    def test_with_chaos_convenience_decorator(self, chaos_config):
        """Test convenience decorator with multiple faults."""
        @with_chaos(crash_prob=1.0)
        def risky_operation():
            return "success"
        
        with pytest.raises(AgentCrashException):
            risky_operation()


# =============================================================================
# Scenario Tests
# =============================================================================


class TestChaosScenarios:
    """Tests for chaos scenarios."""

    def test_list_scenarios(self):
        """Test listing available scenarios."""
        scenarios = list_scenarios()
        
        assert "HandoffCorruption" in scenarios
        assert "DelayedRecovery" in scenarios
        assert len(scenarios) == 5

    def test_get_scenario(self):
        """Test getting scenario by name."""
        scenario = get_scenario("HandoffCorruption")
        assert isinstance(scenario, HandoffCorruption)

    def test_get_unknown_scenario(self):
        """Test getting unknown scenario raises KeyError."""
        with pytest.raises(KeyError):
            get_scenario("UnknownScenario")

    def test_scenario_result_to_dict(self):
        """Test ScenarioResult serialization."""
        result = ScenarioResult(
            scenario_name="TestScenario",
            success=True,
            recovered=True,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow(),
        )
        
        data = result.to_dict()
        assert data["scenario_name"] == "TestScenario"
        assert data["success"] is True

    def test_handoff_corruption_scenario(self, chaos_config):
        """Test HandoffCorruption scenario."""
        scenario = HandoffCorruption()
        result = scenario.run()
        
        assert result.scenario_name == "HandoffCorruption"
        # Corrupted message should be rejected (success = True means validation caught it)
        assert result.success is True

    def test_delayed_recovery_scenario(self, chaos_config):
        """Test DelayedRecovery scenario."""
        scenario = DelayedRecovery(
            delay_ms=100,  # Short delay
            timeout_seconds=10.0,  # Long timeout
        )
        result = scenario.run()
        
        assert result.scenario_name == "DelayedRecovery"
        assert result.success is True  # Should complete without triggering recovery


# =============================================================================
# Runner Tests
# =============================================================================


class TestChaosRunner:
    """Tests for chaos runner."""

    def test_runner_config_defaults(self):
        """Test runner config defaults."""
        config = RunnerConfig()
        
        assert config.runs_per_scenario == 10
        assert config.delay_between_runs == 1.0
        assert config.stop_on_failure is False

    def test_runner_summary_calculations(self):
        """Test runner summary calculations."""
        summary = RunnerSummary(
            total_runs=10,
            successful_runs=8,
            failed_runs=2,
            recovered_runs=7,
        )
        
        assert summary.success_rate == 80.0
        assert summary.recovery_rate == 70.0

    def test_runner_run_simple_scenario(self, chaos_config):
        """Test running a simple scenario."""
        config = RunnerConfig(
            runs_per_scenario=2,
            delay_between_runs=0.1,
            scenarios=["HandoffCorruption"],
        )
        
        runner = ChaosRunner(config)
        summary = runner.run_all()
        
        assert summary.total_runs == 2
        assert "HandoffCorruption" in summary.scenarios_run

    def test_runner_summary_by_scenario(self, chaos_config):
        """Test summary grouped by scenario."""
        config = RunnerConfig(
            runs_per_scenario=2,
            delay_between_runs=0.1,
            scenarios=["HandoffCorruption", "DelayedRecovery"],
        )
        
        runner = ChaosRunner(config)
        summary = runner.run_all()
        
        by_scenario = summary.by_scenario()
        assert "HandoffCorruption" in by_scenario
        assert "DelayedRecovery" in by_scenario


# =============================================================================
# Metrics Tests
# =============================================================================


class TestResilienceMetrics:
    """Tests for resilience metrics collection."""

    def test_record_failure(self, metrics):
        """Test recording a failure event."""
        event = metrics.record_failure(
            agent_id="agent-1",
            thread_id="thread-1",
            failure_type="crash",
        )
        
        assert event.agent_id == "agent-1"
        assert event.failure_type == "crash"
        assert event.failure_time is not None

    def test_record_recovery(self, metrics):
        """Test recording recovery."""
        event = metrics.record_failure("agent-1", "thread-1", "crash")
        time.sleep(0.1)  # Simulate recovery time
        metrics.record_recovery(
            event,
            success=True,
            reconstruction_accuracy=0.95,
        )
        
        assert event.success is True
        assert event.mttr_seconds is not None
        assert event.mttr_seconds >= 0.1
        assert event.reconstruction_accuracy == 0.95

    def test_measure_recovery_context_manager(self, metrics):
        """Test recovery measurement context manager."""
        with metrics.measure_recovery("agent-1", "thread-1", "timeout") as event:
            time.sleep(0.05)
            event.reconstruction_accuracy = 0.9
        
        assert event.success is True
        assert event.mttr_seconds >= 0.05

    def test_summary_calculations(self, metrics):
        """Test summary statistics calculation."""
        # Record some events
        for i in range(5):
            event = metrics.record_failure(f"agent-{i}", f"thread-{i}", "crash")
            time.sleep(0.02)
            metrics.record_recovery(event, success=(i < 4), reconstruction_accuracy=0.8 + i * 0.02)
        
        summary = metrics.get_summary()
        
        assert summary.total_failures == 5
        assert summary.successful_recoveries == 4
        assert summary.recovery_success_rate == 80.0
        assert summary.mttr_avg is not None
        assert summary.avg_reconstruction_accuracy is not None

    def test_percentile_calculation(self, metrics):
        """Test MTTR percentile calculations."""
        # Record events with varying MTTR
        for delay in [0.01, 0.02, 0.05, 0.1, 0.2]:
            event = metrics.record_failure("agent", "thread", "test")
            time.sleep(delay)
            metrics.record_recovery(event, success=True)
        
        summary = metrics.get_summary()
        
        assert summary.mttr_p50 is not None
        assert summary.mttr_p95 is not None
        assert summary.mttr_p95 >= summary.mttr_p50


# =============================================================================
# Export Tests
# =============================================================================


class TestMetricsExporter:
    """Tests for metrics export functionality."""

    def test_export_summary_json(self, metrics, temp_output_dir):
        """Test exporting summary to JSON."""
        event = metrics.record_failure("agent-1", "thread-1", "crash")
        metrics.record_recovery(event, success=True)
        
        exporter = MetricsExporter(output_dir=temp_output_dir)
        filepath = exporter.export_summary_json(metrics.get_summary())
        
        assert filepath.exists()
        assert filepath.suffix == ".json"

    def test_export_events_csv(self, metrics, temp_output_dir):
        """Test exporting events to CSV."""
        event = metrics.record_failure("agent-1", "thread-1", "crash")
        metrics.record_recovery(event, success=True)
        
        exporter = MetricsExporter(output_dir=temp_output_dir)
        filepath = exporter.export_events_csv(metrics.get_events())
        
        assert filepath.exists()
        assert filepath.suffix == ".csv"

    def test_export_all(self, metrics, temp_output_dir):
        """Test exporting all metrics."""
        event = metrics.record_failure("agent-1", "thread-1", "crash")
        metrics.record_recovery(event, success=True)
        
        exporter = MetricsExporter(output_dir=temp_output_dir)
        outputs = exporter.export_all(metrics)
        
        assert "full_metrics_json" in outputs
        assert "summary_json" in outputs
        assert "summary_csv" in outputs

    def test_quick_export(self, metrics, temp_output_dir):
        """Test quick export helper."""
        event = metrics.record_failure("agent-1", "thread-1", "crash")
        metrics.record_recovery(event, success=True)
        
        outputs = quick_export(metrics, output_dir=str(temp_output_dir))
        
        assert len(outputs) > 0
        for path in outputs.values():
            assert path.exists()


# =============================================================================
# Integration Tests
# =============================================================================


class TestChaosIntegration:
    """Integration tests for the chaos framework."""

    def test_full_chaos_workflow(self, chaos_config, temp_output_dir):
        """Test complete chaos testing workflow."""
        # Setup metrics
        reset_metrics()
        metrics = get_metrics()
        metrics.start_measurement()
        
        # Configure runner
        config = RunnerConfig(
            runs_per_scenario=3,
            delay_between_runs=0.1,
            scenarios=["HandoffCorruption", "DelayedRecovery"],
        )
        
        # Run chaos tests
        runner = ChaosRunner(config)
        runner_summary = runner.run_all()
        
        # Record results in metrics
        for result in runner_summary.results:
            metrics.record_scenario_result(result)
        
        # Export results
        exporter = MetricsExporter(output_dir=temp_output_dir)
        outputs = exporter.export_runner_results(runner_summary)
        
        # Verify
        assert runner_summary.total_runs == 6  # 3 runs x 2 scenarios
        assert len(outputs) > 0

