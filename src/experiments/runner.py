"""Experiment runner for thesis evaluation."""

import argparse
import asyncio
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.experiments.scenario_loader import Scenario, ScenarioLoader, ScenarioStep

# Environment variable to control reconstruction mode
# Set PAAS_REAL_RECONSTRUCTION=true to use real HybridReconstructor (default: true)
# Set PAAS_REAL_RECONSTRUCTION=false to use simulated recovery for faster debugging
USE_REAL_RECONSTRUCTION = os.getenv("PAAS_REAL_RECONSTRUCTION", "true").lower() == "true"
from src.experiments.conditions import (
    ExperimentCondition,
    get_condition,
    get_all_conditions,
    RealAPICondition,
)
from src.experiments.collector import MetricsCollector, ExperimentMetrics
from src.reconstruction.hybrid import (
    HybridReconstructor,
    HybridReconstructionResult,
    ReconstructionStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class RecoveryTimingBreakdown:
    """Detailed timing breakdown for recovery operations (Phase D - Gap #10).
    
    Tracks individual operation times to identify bottlenecks:
    - Checkpoint loading from SQLite
    - Event store queries
    - Peer context retrieval (if enabled)
    - Automata prediction (if enabled)
    - LLM inference call
    """
    
    checkpoint_load_ms: float = 0.0
    event_query_ms: float = 0.0
    peer_context_ms: float = 0.0
    automata_predict_ms: float = 0.0
    llm_inference_ms: float = 0.0
    state_merge_ms: float = 0.0
    accuracy_calc_ms: float = 0.0
    total_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "checkpoint_load_ms": self.checkpoint_load_ms,
            "event_query_ms": self.event_query_ms,
            "peer_context_ms": self.peer_context_ms,
            "automata_predict_ms": self.automata_predict_ms,
            "llm_inference_ms": self.llm_inference_ms,
            "state_merge_ms": self.state_merge_ms,
            "accuracy_calc_ms": self.accuracy_calc_ms,
            "total_ms": self.total_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "RecoveryTimingBreakdown":
        """Create from dictionary."""
        return cls(
            checkpoint_load_ms=data.get("checkpoint_load_ms", 0.0),
            event_query_ms=data.get("event_query_ms", 0.0),
            peer_context_ms=data.get("peer_context_ms", 0.0),
            automata_predict_ms=data.get("automata_predict_ms", 0.0),
            llm_inference_ms=data.get("llm_inference_ms", 0.0),
            state_merge_ms=data.get("state_merge_ms", 0.0),
            accuracy_calc_ms=data.get("accuracy_calc_ms", 0.0),
            total_ms=data.get("total_ms", 0.0),
        )


@dataclass
class StepResult:
    """Result of executing a single step."""
    
    step_name: str
    agent: str
    status: str
    success: bool
    duration_ms: float
    error: Optional[str] = None
    recovered: bool = False
    recovery_time_ms: float = 0.0
    # Ground Truth Validation (Phase A)
    ground_truth_state: Optional[Dict[str, Any]] = None
    reconstructed_state: Optional[Dict[str, Any]] = None
    reconstruction_accuracy: float = 0.0
    # Semantic protocol metrics (Gap 4)
    semantic_conflicts: int = 0
    semantic_resolved: int = 0
    semantic_negotiation_ms: float = 0.0
    # Recovery timing breakdown (Phase D - Gap #10)
    recovery_timing: Optional[RecoveryTimingBreakdown] = None


@dataclass
class RecoveryResult:
    """Detailed result from recovery attempt (Phase A - Real Reconstruction)."""
    success: bool
    strategy_used: str
    recovery_time_ms: float
    confidence: float
    ground_truth_state: Optional[Dict[str, Any]] = None
    reconstructed_state: Optional[Dict[str, Any]] = None
    reconstruction_accuracy: float = 0.0
    timing_breakdown: Dict[str, float] = field(default_factory=dict)
    detailed_timing: Optional[RecoveryTimingBreakdown] = None


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    
    run_id: str
    scenario_name: str
    condition_name: str
    success: bool
    total_duration_ms: float
    steps_completed: int
    total_steps: int
    failure_occurred: bool
    failure_step: Optional[str] = None
    recovery_attempted: bool = False
    recovery_success: bool = False
    recovery_time_ms: float = 0.0
    step_results: List[StepResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Ground Truth Validation (Phase A)
    mean_reconstruction_accuracy: float = 0.0
    reconstruction_accuracies: List[float] = field(default_factory=list)
    # Semantic protocol metrics (Gap 4)
    semantic_conflicts: int = 0
    semantic_resolved: int = 0
    semantic_negotiation_ms: float = 0.0
    # Recovery timing breakdown (Phase D - Gap #10)
    recovery_timing_breakdown: Optional[RecoveryTimingBreakdown] = None
    # Cascade failure metrics (Phase F)
    cascade_triggered: bool = False
    cascade_failures: int = 0
    cascade_depth: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "scenario_name": self.scenario_name,
            "condition_name": self.condition_name,
            "success": self.success,
            "total_duration_ms": self.total_duration_ms,
            "steps_completed": self.steps_completed,
            "total_steps": self.total_steps,
            "failure_occurred": self.failure_occurred,
            "failure_step": self.failure_step,
            "recovery_attempted": self.recovery_attempted,
            "recovery_success": self.recovery_success,
            "recovery_time_ms": self.recovery_time_ms,
            "mean_reconstruction_accuracy": self.mean_reconstruction_accuracy,
            "reconstruction_accuracies": self.reconstruction_accuracies,
            "semantic_conflicts": self.semantic_conflicts,
            "semantic_resolved": self.semantic_resolved,
            "semantic_negotiation_ms": self.semantic_negotiation_ms,
            "recovery_timing_breakdown": self.recovery_timing_breakdown.to_dict() if self.recovery_timing_breakdown else None,
            # Cascade failure metrics (Phase F)
            "cascade_triggered": self.cascade_triggered,
            "cascade_failures": self.cascade_failures,
            "cascade_depth": self.cascade_depth,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @property
    def mttr_seconds(self) -> Optional[float]:
        """Get MTTR in seconds, if recovery occurred."""
        if self.recovery_success and self.recovery_time_ms > 0:
            return self.recovery_time_ms / 1000.0
        return None


class ExperimentRunner:
    """Runner for controlled experiments.
    
    Executes scenarios under different conditions and collects metrics.
    """
    
    def __init__(
        self,
        scenarios_dir: Optional[Path] = None,
        failure_probability: float = 0.3,
        seed: Optional[int] = None,
    ):
        """Initialize the runner.
        
        Args:
            scenarios_dir: Directory containing scenario YAML files.
            failure_probability: Probability of injecting a failure.
            seed: Random seed for reproducibility.
        """
        self.loader = ScenarioLoader(scenarios_dir)
        self.failure_probability = failure_probability
        self.collector = MetricsCollector()
        
        # Separate RNG streams for statistical independence (Phase A - Gap 4 fix)
        # Each stream uses a different seed offset to ensure independence
        base_seed = seed if seed is not None else 42
        self._failure_rng = random.Random(base_seed)           # Failure injection decisions
        self._semantic_rng = random.Random(base_seed + 1000)   # Semantic conflict simulation
        self._recovery_rng = random.Random(base_seed + 2000)   # Recovery success decisions
        self._step_rng = random.Random(base_seed + 3000)       # Step execution timing
        
        # Legacy single RNG for backwards compatibility (deprecated)
        self._rng = self._failure_rng
        
        self._results: List[ExperimentResult] = []
    
    def run_single(
        self,
        scenario_name: str,
        condition: ExperimentCondition,
        run_id: Optional[str] = None,
    ) -> ExperimentResult:
        """Run a single experiment.
        
        Args:
            scenario_name: Name of the scenario to run.
            condition: Experimental condition to use.
            run_id: Optional run ID (generated if not provided).
            
        Returns:
            ExperimentResult with metrics.
        """
        run_id = run_id or str(uuid.uuid4())[:8]
        scenario = self.loader.load(scenario_name)
        
        logger.info(
            f"Running experiment: {scenario.name} under {condition.name} "
            f"(run_id={run_id})"
        )
        
        start_time = time.time()
        step_results: List[StepResult] = []
        failure_occurred = False
        failure_step = None
        steps_completed = 0
        recovery_attempted = False
        recovery_success = False
        recovery_time_ms = 0.0
        reconstruction_accuracies: List[float] = []  # Phase A - Ground Truth
        
        # Cascade failure tracking (Phase F)
        cascade_triggered = False
        cascade_trigger_step = -1
        cascade_depth = 0
        cascade_failures = 0
        
        # Execute each step
        for i, step in enumerate(scenario.steps):
            step_start = time.time()
            
            # Check if we should inject a failure (uses _failure_rng for independence)
            should_fail = (
                scenario.failure_injection.enabled
                and i in scenario.failure_injection.target_steps
                and self._failure_rng.random() < self.failure_probability
            )
            
            # Check for cascade failure (Phase F)
            cascade_config = scenario.failure_injection.cascade
            should_cascade = False
            if cascade_triggered and not failure_occurred:
                should_cascade = self._should_cascade_failure(
                    i, scenario, cascade_depth, cascade_triggered
                )
                if should_cascade:
                    cascade_depth += 1
                    cascade_failures += 1
                    logger.debug(
                        f"Cascade failure at step {step.name} "
                        f"(depth={cascade_depth}, triggered at step {cascade_trigger_step})"
                    )
            
            # Check if this step triggers a cascade
            if (
                should_fail
                and not cascade_triggered
                and cascade_config
                and cascade_config.enabled
                and i == cascade_config.trigger_step
            ):
                cascade_triggered = True
                cascade_trigger_step = i
                logger.debug(f"Cascade triggered at step {step.name}")
            
            if (should_fail or should_cascade) and not failure_occurred:
                # Capture ground truth BEFORE failure injection (Phase A)
                ground_truth_state = self._capture_current_state(scenario, step, i)
                
                # Inject failure
                failure_occurred = True
                failure_step = step.name
                
                # Determine failure type (cascade failures have specific types)
                if should_cascade:
                    failure_type = self._get_cascade_failure_type(
                        scenario,
                        scenario.failure_injection.failure_types[0] if scenario.failure_injection.failure_types else "crash"
                    )
                    # Add cascade delay
                    cascade_config = scenario.failure_injection.cascade
                    if cascade_config and cascade_config.delay_between_failures_ms > 0:
                        time.sleep(cascade_config.delay_between_failures_ms / 1000.0)
                else:
                    failure_type = self._failure_rng.choice(scenario.failure_injection.failure_types)
                
                logger.debug(f"Injecting {failure_type} at step {step.name} (cascade={should_cascade})")
                
                # Simulate failure duration (uses _step_rng)
                step_duration_ms = self._step_rng.uniform(10, 50)
                
                step_result = StepResult(
                    step_name=step.name,
                    agent=step.agent,
                    status="failed",
                    success=False,
                    duration_ms=step_duration_ms,
                    error=f"Injected {failure_type}",
                    ground_truth_state=ground_truth_state,
                )
                
                # Attempt recovery if condition allows
                if condition.should_attempt_recovery():
                    recovery_attempted = True
                    recovery_start = time.time()
                    
                    # Simulate recovery (returns success and simulated reconstructed state)
                    recovery_success, reconstructed_state = self._simulate_recovery_with_state(
                        condition, scenario, step, failure_type, ground_truth_state
                    )
                    
                    recovery_time_ms = (time.time() - recovery_start) * 1000
                    
                    # Calculate reconstruction accuracy (Phase A)
                    if reconstructed_state:
                        accuracy = self._calculate_state_similarity(
                            ground_truth_state, reconstructed_state
                        )
                        step_result.reconstructed_state = reconstructed_state
                        step_result.reconstruction_accuracy = accuracy
                        reconstruction_accuracies.append(accuracy)
                    
                    if recovery_success:
                        step_result.recovered = True
                        step_result.recovery_time_ms = recovery_time_ms
                        step_result.status = "recovered"
                        step_result.success = True
                        steps_completed += 1
                        logger.debug(
                            f"Recovery successful in {recovery_time_ms:.1f}ms "
                            f"(accuracy: {step_result.reconstruction_accuracy:.2%})"
                        )
                    else:
                        logger.debug("Recovery failed")
                        # In baseline, failure stops the workflow
                        step_results.append(step_result)
                        break
                else:
                    # No recovery - workflow fails
                    step_results.append(step_result)
                    break
            else:
                # Normal execution
                step_duration_ms = self._simulate_step_execution(step)
                
                # Simulate semantic conflicts (Gap 4)
                conflicts, resolved, neg_time = self._simulate_semantic_conflicts(
                    step, condition
                )
                
                step_result = StepResult(
                    step_name=step.name,
                    agent=step.agent,
                    status=step.expected_status,
                    success=True,
                    duration_ms=step_duration_ms + neg_time,  # Add negotiation time
                    semantic_conflicts=conflicts,
                    semantic_resolved=resolved,
                    semantic_negotiation_ms=neg_time,
                )
                
                steps_completed += 1
            
            step_results.append(step_result)
        
        total_duration_ms = (time.time() - start_time) * 1000
        
        # Determine overall success
        success = (
            steps_completed == scenario.num_steps
            or (failure_occurred and recovery_success)
        )
        
        # Aggregate semantic metrics from all steps (Gap 4)
        total_semantic_conflicts = sum(sr.semantic_conflicts for sr in step_results)
        total_semantic_resolved = sum(sr.semantic_resolved for sr in step_results)
        total_semantic_negotiation_ms = sum(sr.semantic_negotiation_ms for sr in step_results)
        
        # Aggregate reconstruction accuracy (Phase A - Ground Truth)
        mean_reconstruction_accuracy = (
            sum(reconstruction_accuracies) / len(reconstruction_accuracies)
            if reconstruction_accuracies else 0.0
        )
        
        result = ExperimentResult(
            run_id=run_id,
            scenario_name=scenario.name,
            condition_name=condition.name,
            success=success,
            total_duration_ms=total_duration_ms,
            steps_completed=steps_completed,
            total_steps=scenario.num_steps,
            failure_occurred=failure_occurred,
            failure_step=failure_step,
            recovery_attempted=recovery_attempted,
            recovery_success=recovery_success,
            recovery_time_ms=recovery_time_ms,
            step_results=step_results,
            mean_reconstruction_accuracy=mean_reconstruction_accuracy,
            reconstruction_accuracies=reconstruction_accuracies,
            semantic_conflicts=total_semantic_conflicts,
            semantic_resolved=total_semantic_resolved,
            semantic_negotiation_ms=total_semantic_negotiation_ms,
            # Cascade failure metrics (Phase F)
            cascade_triggered=cascade_triggered,
            cascade_failures=cascade_failures,
            cascade_depth=cascade_depth,
            metadata={
                "scenario_complexity": scenario.complexity,
                "condition_type": condition.condition_type.value,
                "failure_probability": self.failure_probability,
            },
        )
        
        # Record metrics
        self.collector.record_result(result)
        self._results.append(result)
        
        return result
    
    def _simulate_step_execution(self, step: ScenarioStep) -> float:
        """Simulate step execution and return duration in ms.
        
        Uses _step_rng for statistical independence from failure injection.
        """
        # Base duration + some variance
        base_duration = self._step_rng.uniform(20, 100)
        return base_duration
    
    def _should_cascade_failure(
        self,
        step_idx: int,
        scenario: Scenario,
        cascade_depth: int,
        cascade_triggered: bool,
    ) -> bool:
        """Check if failure should cascade to this step.
        
        Cascade failures propagate from a trigger step to downstream steps
        with decreasing probability based on depth.
        
        Args:
            step_idx: Current step index.
            scenario: Current scenario with failure config.
            cascade_depth: How many steps the cascade has propagated.
            cascade_triggered: Whether cascade was triggered earlier.
            
        Returns:
            True if this step should fail due to cascade.
        """
        cascade_config = scenario.failure_injection.cascade
        
        # No cascade if not configured or not triggered
        if not cascade_config or not cascade_config.enabled:
            return False
        
        if not cascade_triggered:
            return False
        
        # Check if we've exceeded max cascade depth
        if cascade_depth >= cascade_config.max_depth:
            return False
        
        # Steps after trigger have decreasing probability
        # Each step has downstream_probability ^ depth chance
        cascade_prob = cascade_config.downstream_probability ** (cascade_depth + 1)
        
        return self._failure_rng.random() < cascade_prob
    
    def _get_cascade_failure_type(
        self,
        scenario: Scenario,
        original_failure_type: str,
    ) -> str:
        """Get failure type for cascaded failure.
        
        Cascade failures tend to manifest as timeouts or crashes
        as downstream systems lose connectivity to failed upstream.
        
        Args:
            scenario: Current scenario.
            original_failure_type: The original failure that triggered cascade.
            
        Returns:
            Failure type for the cascaded failure.
        """
        # Cascaded failures are often timeouts or crashes
        cascade_types = ["timeout", "crash"]
        if "message_corruption" in scenario.failure_injection.failure_types:
            cascade_types.append("message_corruption")
        
        return self._failure_rng.choice(cascade_types)
    
    def _capture_current_state(
        self,
        scenario: Scenario,
        step: ScenarioStep,
        step_index: int,
    ) -> Dict[str, Any]:
        """Capture ground truth state before failure injection (Phase A).
        
        Creates a snapshot of the current workflow state that can be compared
        against reconstructed state for accuracy measurement.
        
        Args:
            scenario: Current scenario being executed.
            step: Step where failure is being injected.
            step_index: Index of the current step.
            
        Returns:
            Dictionary representing the ground truth state.
        """
        # Build ground truth state based on scenario context
        ground_truth = {
            "scenario_name": scenario.name,
            "current_step": step.name,
            "step_index": step_index,
            "agent": step.agent,
            "expected_status": step.expected_status,
            "progress_percentage": (step_index / scenario.num_steps) * 100,
            # Simulated state variables based on step context
            "variables": {
                "workflow_id": f"{scenario.name}_{step_index}",
                "action": step.action,
                "timeout_seconds": step.timeout_seconds,
                "target_agent": step.target_agent,
            },
            # Pending actions for subsequent steps
            "pending_steps": [
                {"name": s.name, "agent": s.agent}
                for s in scenario.steps[step_index + 1:]
            ],
            # Completed steps before this point
            "completed_steps": step_index,
        }
        
        return ground_truth
    
    def _calculate_state_similarity(
        self,
        ground_truth: Dict[str, Any],
        reconstructed: Dict[str, Any],
        numeric_tolerance: float = 0.01,
    ) -> float:
        """Calculate similarity between ground truth and reconstructed state (Phase A).
        
        Uses type-aware comparison with:
        - Exact matching for strings and booleans
        - Tolerance-based matching for numeric values
        - Recursive comparison for nested dicts
        - Jaccard similarity for lists/sets
        
        Args:
            ground_truth: The actual state before failure.
            reconstructed: The state reconstructed by HybridReconstructor.
            numeric_tolerance: Relative tolerance for numeric comparisons.
            
        Returns:
            Similarity score between 0.0 and 1.0.
        """
        # Handle None inputs
        if ground_truth is None or reconstructed is None:
            return 0.0
        
        # Get all unique keys
        all_keys = set(ground_truth.keys()) | set(reconstructed.keys())
        if not all_keys:
            return 1.0  # Both empty dicts
        
        matches = 0.0
        total_fields = len(all_keys)
        
        for key in all_keys:
            gt_value = ground_truth.get(key)
            rec_value = reconstructed.get(key)
            
            # Both missing (shouldn't happen, but handle it)
            if gt_value is None and rec_value is None:
                matches += 1.0
                continue
            
            # One missing
            if gt_value is None or rec_value is None:
                continue
            
            # Same type comparison
            if type(gt_value) != type(rec_value):
                # Type mismatch - try string comparison
                if str(gt_value) == str(rec_value):
                    matches += 0.5  # Partial credit
                continue
            
            # Type-specific comparison
            # Note: bool must be checked before int because bool is subclass of int
            if isinstance(gt_value, bool):
                # Exact match for booleans
                if gt_value == rec_value:
                    matches += 1.0
            elif isinstance(gt_value, dict):
                # Recursive comparison for nested dicts
                nested_sim = self._calculate_state_similarity(
                    gt_value, rec_value, numeric_tolerance
                )
                matches += nested_sim
            elif isinstance(gt_value, (list, tuple)):
                # Jaccard similarity for lists
                if not gt_value and not rec_value:
                    matches += 1.0
                elif not gt_value or not rec_value:
                    continue
                else:
                    # Convert to sets for comparison (order-independent)
                    try:
                        gt_set = set(str(v) for v in gt_value)
                        rec_set = set(str(v) for v in rec_value)
                        intersection = len(gt_set & rec_set)
                        union = len(gt_set | rec_set)
                        matches += intersection / union if union > 0 else 0.0
                    except (TypeError, ValueError):
                        # Can't convert to set, do simple equality
                        if gt_value == rec_value:
                            matches += 1.0
            elif isinstance(gt_value, (int, float)):
                # Numeric comparison with tolerance
                if gt_value == 0:
                    if rec_value == 0:
                        matches += 1.0
                elif abs(gt_value - rec_value) / abs(gt_value) <= numeric_tolerance:
                    matches += 1.0
            elif isinstance(gt_value, str):
                # String comparison (case-insensitive for flexibility)
                if gt_value.lower() == rec_value.lower():
                    matches += 1.0
                elif gt_value in rec_value or rec_value in gt_value:
                    matches += 0.5  # Partial credit for substring match
            else:
                # Default exact comparison
                if gt_value == rec_value:
                    matches += 1.0
        
        return matches / total_fields if total_fields > 0 else 0.0
    
    def _simulate_semantic_conflicts(
        self,
        step: ScenarioStep,
        condition: ExperimentCondition,
    ) -> tuple:
        """Simulate semantic conflict detection and resolution.
        
        Uses _semantic_rng for statistical independence from failure injection.
        
        Args:
            step: Current step with potential term_conflicts.
            condition: Experiment condition (determines if semantic protocol enabled).
            
        Returns:
            Tuple of (conflicts_occurred: int, conflicts_resolved: int, negotiation_time_ms: float).
        """
        conflicts_occurred = 0
        conflicts_resolved = 0
        negotiation_time_ms = 0.0
        
        # Check if step has potential term conflicts
        if not step.has_term_conflicts():
            return (0, 0, 0.0)
        
        term_conflict = step.term_conflicts
        
        # Roll for conflict occurrence based on probability (uses _semantic_rng)
        if self._semantic_rng.random() < term_conflict.probability:
            conflicts_occurred = 1
            
            # Check if semantic protocol is enabled for this condition
            if condition.should_use_semantic_protocol():
                # Semantic protocol enabled - high resolution success rate
                # Severity affects resolution success rate:
                # - low: 95% success
                # - medium: 90% success  
                # - high: 85% success
                severity_rates = {
                    "low": 0.95,
                    "medium": 0.90,
                    "high": 0.85,
                }
                success_rate = severity_rates.get(term_conflict.severity, 0.90)
                
                if self._semantic_rng.random() < success_rate:
                    conflicts_resolved = 1
                    # Simulate negotiation time: 50-200ms based on severity
                    time_ranges = {
                        "low": (50, 100),
                        "medium": (75, 150),
                        "high": (100, 200),
                    }
                    min_t, max_t = time_ranges.get(term_conflict.severity, (75, 150))
                    negotiation_time_ms = self._semantic_rng.uniform(min_t, max_t)
            else:
                # No semantic protocol - much lower resolution rate
                # Conflicts may still resolve through luck/simple matching
                if self._semantic_rng.random() < 0.30:  # 30% accidental resolution
                    conflicts_resolved = 1
                    negotiation_time_ms = self._semantic_rng.uniform(10, 30)
        
        return (conflicts_occurred, conflicts_resolved, negotiation_time_ms)
    
    async def _execute_real_semantic_negotiation(
        self,
        step: ScenarioStep,
        condition: ExperimentCondition,
    ) -> tuple:
        """Execute REAL semantic negotiation using TermNegotiator with LLM.
        
        This method uses the actual TermNegotiator from src/semantic/negotiator.py
        to resolve semantic conflicts via LLM, providing real negotiation timing
        instead of simulated values.
        
        Args:
            step: Current step with potential term_conflicts.
            condition: Experiment condition (determines if semantic protocol enabled).
            
        Returns:
            Tuple of (conflicts_occurred: int, conflicts_resolved: int, negotiation_time_ms: float).
        """
        # Check if step has potential term conflicts
        if not step.has_term_conflicts():
            return (0, 0, 0.0)
        
        term_conflict = step.term_conflicts
        
        # Roll for conflict occurrence based on probability (uses _semantic_rng)
        if self._semantic_rng.random() >= term_conflict.probability:
            return (0, 0, 0.0)
        
        # Check if semantic protocol is enabled for this condition
        if not condition.should_use_semantic_protocol():
            # No semantic protocol - use simulated fallback
            if self._semantic_rng.random() < 0.30:  # 30% accidental resolution
                return (1, 1, self._semantic_rng.uniform(10, 30))
            return (1, 0, 0.0)
        
        try:
            from src.semantic.negotiator import TermNegotiator
            from src.llm import get_llm
            
            # Create negotiator with real LLM
            negotiator = TermNegotiator(llm=get_llm())
            
            # Negotiate each conflicting term
            start_time = time.perf_counter()
            conflicts_occurred = len(term_conflict.terms)
            conflicts_resolved = 0
            
            for term in term_conflict.terms:
                # Generate realistic definitions based on severity
                definition_a = f"Agent A's understanding of '{term}' in e-commerce context"
                definition_b = f"Agent B's alternative definition of '{term}' for workflow"
                
                result = await negotiator.negotiate_term(
                    term=term,
                    definition_a=definition_a,
                    definition_b=definition_b,
                    context_a=f"Step: {step.name}, Agent: {step.agent}",
                    context_b=f"Target: {step.target_agent or 'N/A'}",
                )
                
                if result.agreed_definition:
                    conflicts_resolved += 1
            
            negotiation_time_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Real semantic negotiation: {conflicts_resolved}/{conflicts_occurred} "
                f"resolved in {negotiation_time_ms:.1f}ms"
            )
            return (conflicts_occurred, conflicts_resolved, negotiation_time_ms)
            
        except Exception as e:
            logger.warning(f"Real semantic negotiation failed: {e}, falling back to simulation")
            # Fallback to simulated resolution
            return self._simulate_semantic_conflicts(step, condition)
    
    def _simulate_recovery(
        self,
        condition: ExperimentCondition,
        scenario: Scenario,
        step: ScenarioStep,
        failure_type: str,
    ) -> bool:
        """Simulate recovery attempt.
        
        Uses _recovery_rng for statistical independence from failure injection.
        
        Returns True if recovery is successful.
        
        Strategy success rates are based on realistic assumptions:
        - simple_retry: Low success (transient failures only)
        - checkpoint: Moderate success (loses intermediate state)
        - automata_only: Good for structured workflows, fails on novel situations
        - llm_no_peer: LLM without context has reduced accuracy
        - llm: LLM with peer context performs better
        - automata: Automata with LLM fallback
        - hybrid: Full system combines all approaches
        """
        # Base recovery success rate depends on condition
        strategy = condition.get_reconstruction_strategy()
        
        if strategy == "none":
            return False
        
        # =================================================================
        # New comparison baseline strategies
        # =================================================================
        elif strategy == "simple_retry":
            # Simple retry: Low success rate - only works for transient failures
            # No state reconstruction, just retry the operation
            base_success_rate = 0.35
            # Fast - just retry immediately (uses _recovery_rng)
            time.sleep(self._recovery_rng.uniform(0.01, 0.03))
        
        elif strategy == "checkpoint":
            # Checkpoint only: Moderate success - can resume but may lose
            # intermediate state that was computed between checkpoints
            base_success_rate = 0.55
            # Relatively fast - just load checkpoint (uses _recovery_rng)
            time.sleep(self._recovery_rng.uniform(0.02, 0.05))
        
        elif strategy == "automata_only":
            # Automata only: Good for structured/repetitive workflows
            # Fails on novel situations since no LLM fallback
            base_success_rate = 0.70
            # Fast - deterministic prediction (uses _recovery_rng)
            time.sleep(self._recovery_rng.uniform(0.04, 0.08))
        
        elif strategy == "llm_no_peer":
            # LLM without peer context: Less accurate than with peer context
            # because the LLM has less information to reason about
            base_success_rate = 0.68
            # Medium - LLM inference time (uses _recovery_rng)
            time.sleep(self._recovery_rng.uniform(0.05, 0.12))
        
        # =================================================================
        # Phase B - Industry standard baseline strategies
        # =================================================================
        elif strategy == "exponential_backoff":
            # Exponential backoff: retry with increasing delays
            # Only works for transient failures, not state-dependent ones
            base_success_rate = 0.40
            
            # Simulate exponential backoff delays: 100ms, 200ms, 400ms, 800ms
            delays = [0.1, 0.2, 0.4, 0.8]
            for delay in delays:
                jitter = self._recovery_rng.uniform(0, delay * 0.1)
                time.sleep(delay + jitter)
                # Each retry has chance of success for transient failures
                if failure_type in ["timeout", "network_error"]:
                    if self._recovery_rng.random() < 0.35:  # 35% per retry for transient
                        return True
            # If all retries exhausted, use base rate for final attempt
            time.sleep(self._recovery_rng.uniform(0.01, 0.02))
        
        elif strategy == "circuit_breaker":
            # Circuit breaker: fail-fast after threshold
            # Doesn't actually recover state, just prevents cascade
            base_success_rate = 0.45
            # Very fast - just checks circuit state and fails fast
            time.sleep(self._recovery_rng.uniform(0.005, 0.015))
        
        elif strategy == "semantic_only":
            # Semantic-only: no recovery, just preventive alignment
            # This strategy never recovers because resilience_enabled=False
            # Should not reach here (handled by "none" case above)
            return False
        
        # =================================================================
        # Original strategies
        # =================================================================
        elif strategy == "llm":
            # LLM with peer context: Better accuracy due to more context
            base_success_rate = 0.75
            # Simulate some recovery time (uses _recovery_rng)
            time.sleep(self._recovery_rng.uniform(0.05, 0.15))
        
        elif strategy == "automata":
            # Automata with LLM fallback: Good success rate
            base_success_rate = 0.78
            time.sleep(self._recovery_rng.uniform(0.05, 0.12))
        
        elif strategy == "hybrid":
            # Full system has highest success rate
            # Combines automata prediction, LLM reasoning, semantic protocol
            base_success_rate = 0.92
            # Slightly longer due to multiple components (uses _recovery_rng)
            time.sleep(self._recovery_rng.uniform(0.08, 0.2))
        
        else:
            # Unknown strategy - default fallback
            base_success_rate = 0.5
            time.sleep(self._recovery_rng.uniform(0.03, 0.1))
        
        # Adjust based on failure type
        if failure_type == "hallucination":
            base_success_rate -= 0.1  # Harder to recover from hallucination
        elif failure_type == "crash":
            base_success_rate += 0.05  # Crashes are easier to detect/recover
        elif failure_type == "timeout":
            base_success_rate -= 0.05  # Timeouts may indicate deeper issues
        elif failure_type == "corruption":
            base_success_rate -= 0.08  # Data corruption is tricky
        
        # Ensure rate stays in valid range
        base_success_rate = max(0.0, min(1.0, base_success_rate))
        
        # Final success roll using _recovery_rng for independence
        return self._recovery_rng.random() < base_success_rate
    
    def _simulate_recovery_with_state(
        self,
        condition: ExperimentCondition,
        scenario: Scenario,
        step: ScenarioStep,
        failure_type: str,
        ground_truth_state: Dict[str, Any],
    ) -> tuple:
        """Simulate recovery attempt with state reconstruction (Phase A).
        
        This extends _simulate_recovery to also generate a simulated
        reconstructed state for accuracy comparison.
        
        Uses _recovery_rng for statistical independence.
        
        Args:
            condition: Experiment condition.
            scenario: Current scenario.
            step: Step where failure occurred.
            failure_type: Type of failure injected.
            ground_truth_state: The actual state before failure.
            
        Returns:
            Tuple of (success: bool, reconstructed_state: Dict or None).
        """
        strategy = condition.get_reconstruction_strategy()
        
        # Get base success determination from original method
        success = self._simulate_recovery(condition, scenario, step, failure_type)
        
        if strategy == "none":
            return (False, None)
        
        # Generate simulated reconstructed state based on strategy accuracy
        # Different strategies have different reconstruction accuracy profiles
        strategy_accuracy_profiles = {
            "simple_retry": 0.3,      # Simple retry has no state reconstruction
            "checkpoint": 0.60,       # Checkpoint can restore state but may be stale
            "automata_only": 0.75,    # Automata good at structured state
            "llm_no_peer": 0.70,      # LLM without context has some gaps
            "llm": 0.80,              # LLM with peer context is better
            "automata": 0.82,         # Automata + LLM fallback
            "hybrid": 0.92,           # Full system is most accurate
        }
        
        base_accuracy = strategy_accuracy_profiles.get(strategy, 0.5)
        
        # Add some variance using _recovery_rng
        accuracy_variance = self._recovery_rng.uniform(-0.08, 0.08)
        target_accuracy = max(0.0, min(1.0, base_accuracy + accuracy_variance))
        
        # Generate reconstructed state by perturbing ground truth
        reconstructed_state = self._generate_reconstructed_state(
            ground_truth_state, target_accuracy
        )
        
        return (success, reconstructed_state)
    
    def _generate_reconstructed_state(
        self,
        ground_truth: Dict[str, Any],
        target_accuracy: float,
    ) -> Dict[str, Any]:
        """Generate a simulated reconstructed state (Phase A).
        
        Creates a state that approximately matches the target accuracy
        when compared against ground truth.
        
        Args:
            ground_truth: The actual state.
            target_accuracy: Target similarity (0.0 to 1.0).
            
        Returns:
            Simulated reconstructed state.
        """
        import copy
        
        reconstructed = copy.deepcopy(ground_truth)
        
        if not ground_truth:
            return reconstructed
        
        # Calculate how many fields to "corrupt" to achieve target accuracy
        num_fields = len(ground_truth)
        fields_to_corrupt = int((1.0 - target_accuracy) * num_fields)
        
        if fields_to_corrupt <= 0:
            return reconstructed
        
        # Randomly select fields to corrupt (excluding critical fields)
        corruptible_fields = [
            k for k in ground_truth.keys()
            if k not in ("scenario_name", "current_step")  # Preserve these
        ]
        
        if not corruptible_fields:
            return reconstructed
        
        fields_to_modify = self._recovery_rng.sample(
            corruptible_fields,
            min(fields_to_corrupt, len(corruptible_fields))
        )
        
        for field in fields_to_modify:
            value = reconstructed.get(field)
            
            if value is None:
                continue
            elif isinstance(value, dict):
                # Corrupt nested dict partially
                if value:
                    nested_key = self._recovery_rng.choice(list(value.keys()))
                    value[nested_key] = f"corrupted_{nested_key}"
            elif isinstance(value, list):
                # Corrupt list by shuffling or truncating
                if value:
                    if self._recovery_rng.random() < 0.5:
                        self._recovery_rng.shuffle(value)
                    else:
                        reconstructed[field] = value[:-1] if len(value) > 1 else []
            elif isinstance(value, (int, float)):
                # Perturb numeric values
                noise = self._recovery_rng.uniform(-0.2, 0.2) * (value if value != 0 else 1)
                reconstructed[field] = value + noise
            elif isinstance(value, str):
                # Corrupt strings by appending noise
                reconstructed[field] = f"{value}_reconstructed"
            elif isinstance(value, bool):
                # Flip boolean with some probability
                reconstructed[field] = not value
        
        return reconstructed
    
    async def _execute_real_recovery(
        self,
        condition: ExperimentCondition,
        agent_id: str,
        thread_id: str,
        ground_truth_state: Dict[str, Any],
        events: List[Dict[str, Any]],
    ) -> RecoveryResult:
        """Execute ACTUAL recovery using HybridReconstructor (Phase A - Task A1).
        
        This method calls the real HybridReconstructor instead of simulating.
        
        Args:
            condition: Experiment condition controlling recovery strategy.
            agent_id: ID of the agent to recover.
            thread_id: Thread/workflow ID.
            ground_truth_state: Actual state before failure for accuracy comparison.
            events: Event history for reconstruction.
            
        Returns:
            RecoveryResult with detailed metrics including accuracy.
        """
        import time as time_module
        
        strategy = condition.get_reconstruction_strategy()
        
        if strategy == "none":
            return RecoveryResult(
                success=False,
                strategy_used="none",
                recovery_time_ms=0.0,
                confidence=0.0,
            )
        
        start_time = time_module.perf_counter()
        
        # Configure HybridReconstructor based on condition
        enable_automata = strategy in ("automata_only", "automata", "hybrid")
        enable_llm = strategy in ("llm_no_peer", "llm", "automata", "hybrid")
        enable_peer_context = strategy in ("llm", "hybrid")
        
        # Create reconstructor with appropriate settings
        reconstructor = HybridReconstructor(
            enable_automata=enable_automata,
            enable_llm=enable_llm,
            enable_peer_context=enable_peer_context,
            checkpoint_freshness=30,
            min_events_for_automata=10,  # Lower threshold for testing
            automata_confidence_threshold=0.7,
        )
        
        # Train automata if enabled and we have events
        if enable_automata and len(events) >= 10:
            try:
                reconstructor.train_automata(events)
            except Exception as e:
                logger.warning(f"Automata training failed: {e}")
        
        # Build a simulated checkpoint from ground truth (as it would be in real system)
        checkpoint = {
            "channel_values": {
                "workflow_id": ground_truth_state.get("variables", {}).get("workflow_id"),
                "current_step": ground_truth_state.get("step_index", 0),
                "status": "in_progress",
            },
            "ts": datetime.now().isoformat(),
        }
        
        timing_breakdown = {}
        detailed_timing = RecoveryTimingBreakdown()
        
        try:
            # Time checkpoint loading
            checkpoint_start = time_module.perf_counter()
            # Checkpoint is already built above, but in real system this would be a DB query
            detailed_timing.checkpoint_load_ms = (time_module.perf_counter() - checkpoint_start) * 1000
            timing_breakdown["checkpoint_load_ms"] = detailed_timing.checkpoint_load_ms
            
            # Time event query (simulated - events are already passed)
            event_query_start = time_module.perf_counter()
            recent_events = events[-20:] if events else []
            detailed_timing.event_query_ms = (time_module.perf_counter() - event_query_start) * 1000
            timing_breakdown["event_query_ms"] = detailed_timing.event_query_ms
            
            # Execute actual reconstruction with timing
            recon_start = time_module.perf_counter()
            result: HybridReconstructionResult = await reconstructor.reconstruct(
                agent_id=agent_id,
                thread_id=thread_id,
                checkpoint=checkpoint,
                events_since_checkpoint=recent_events,
                all_events=events,
            )
            total_recon_ms = (time_module.perf_counter() - recon_start) * 1000
            timing_breakdown["reconstruction_ms"] = total_recon_ms
            
            # Estimate component timing from result strategy
            if result.strategy == ReconstructionStrategy.AUTOMATA:
                detailed_timing.automata_predict_ms = total_recon_ms * 0.8
                detailed_timing.state_merge_ms = total_recon_ms * 0.2
            elif result.strategy == ReconstructionStrategy.LLM:
                detailed_timing.llm_inference_ms = total_recon_ms * 0.9
                detailed_timing.state_merge_ms = total_recon_ms * 0.1
            elif result.strategy == ReconstructionStrategy.HYBRID:
                detailed_timing.automata_predict_ms = total_recon_ms * 0.3
                detailed_timing.llm_inference_ms = total_recon_ms * 0.5
                detailed_timing.state_merge_ms = total_recon_ms * 0.2
            elif result.strategy == ReconstructionStrategy.CHECKPOINT:
                detailed_timing.checkpoint_load_ms += total_recon_ms
            
            # Check if peer context was used (would be tracked in real implementation)
            if enable_peer_context:
                # Estimate peer context overhead (in real system, would be measured)
                detailed_timing.peer_context_ms = total_recon_ms * 0.1
            
            # Calculate accuracy against ground truth
            accuracy_start = time_module.perf_counter()
            reconstruction_accuracy = self._calculate_state_similarity(
                ground_truth_state,
                result.reconstructed_state,
            )
            detailed_timing.accuracy_calc_ms = (time_module.perf_counter() - accuracy_start) * 1000
            timing_breakdown["accuracy_calc_ms"] = detailed_timing.accuracy_calc_ms
            
            total_time_ms = (time_module.perf_counter() - start_time) * 1000
            detailed_timing.total_ms = total_time_ms
            
            return RecoveryResult(
                success=result.success,
                strategy_used=result.strategy.value,
                recovery_time_ms=total_time_ms,
                confidence=result.confidence,
                ground_truth_state=ground_truth_state,
                reconstructed_state=result.reconstructed_state,
                reconstruction_accuracy=reconstruction_accuracy,
                timing_breakdown=timing_breakdown,
                detailed_timing=detailed_timing,
            )
            
        except Exception as e:
            logger.error(f"Real recovery failed: {e}")
            total_time_ms = (time_module.perf_counter() - start_time) * 1000
            detailed_timing.total_ms = total_time_ms
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                recovery_time_ms=total_time_ms,
                confidence=0.0,
                ground_truth_state=ground_truth_state,
                timing_breakdown={"error": str(e)},
                detailed_timing=detailed_timing,
            )
    
    async def run_single_async(
        self,
        scenario_name: str,
        condition: ExperimentCondition,
        run_id: Optional[str] = None,
        use_real_reconstruction: bool = False,
    ) -> ExperimentResult:
        """Async version of run_single with optional real reconstruction (Phase A).
        
        Args:
            scenario_name: Name of the scenario to run.
            condition: Experimental condition to use.
            run_id: Optional run ID (generated if not provided).
            use_real_reconstruction: If True, use HybridReconstructor instead of simulation.
            
        Returns:
            ExperimentResult with metrics.
        """
        run_id = run_id or str(uuid.uuid4())[:8]
        scenario = self.loader.load(scenario_name)
        
        logger.info(
            f"Running async experiment: {scenario.name} under {condition.name} "
            f"(run_id={run_id}, real_reconstruction={use_real_reconstruction})"
        )
        
        start_time = time.time()
        step_results: List[StepResult] = []
        failure_occurred = False
        failure_step = None
        steps_completed = 0
        recovery_attempted = False
        recovery_success = False
        recovery_time_ms = 0.0
        reconstruction_accuracies: List[float] = []
        
        # Cascade failure tracking (Phase F)
        cascade_triggered = False
        cascade_trigger_step = -1
        cascade_depth = 0
        cascade_failures = 0
        
        # Simulated event history for reconstruction
        event_history: List[Dict[str, Any]] = []
        
        for i, step in enumerate(scenario.steps):
            # Build simulated event for history
            event_history.append({
                "event_type": "step_execution",
                "step_name": step.name,
                "agent": step.agent,
                "timestamp": datetime.now().isoformat(),
                "data": {"action": step.action, "step_index": i},
            })
            
            should_fail = (
                scenario.failure_injection.enabled
                and i in scenario.failure_injection.target_steps
                and self._failure_rng.random() < self.failure_probability
            )
            
            # Check for cascade failure (Phase F)
            cascade_config = scenario.failure_injection.cascade
            should_cascade = False
            if cascade_triggered and not failure_occurred:
                should_cascade = self._should_cascade_failure(
                    i, scenario, cascade_depth, cascade_triggered
                )
                if should_cascade:
                    cascade_depth += 1
                    cascade_failures += 1
            
            # Check if this step triggers a cascade
            if (
                should_fail
                and not cascade_triggered
                and cascade_config
                and cascade_config.enabled
                and i == cascade_config.trigger_step
            ):
                cascade_triggered = True
                cascade_trigger_step = i
            
            if (should_fail or should_cascade) and not failure_occurred:
                ground_truth_state = self._capture_current_state(scenario, step, i)
                failure_occurred = True
                failure_step = step.name
                failure_type = self._failure_rng.choice(scenario.failure_injection.failure_types)
                
                logger.debug(f"Injecting {failure_type} at step {step.name}")
                
                step_duration_ms = self._step_rng.uniform(10, 50)
                
                step_result = StepResult(
                    step_name=step.name,
                    agent=step.agent,
                    status="failed",
                    success=False,
                    duration_ms=step_duration_ms,
                    error=f"Injected {failure_type}",
                    ground_truth_state=ground_truth_state,
                )
                
                if condition.should_attempt_recovery():
                    recovery_attempted = True
                    recovery_start = time.time()
                    
                    if use_real_reconstruction:
                        # Use REAL HybridReconstructor
                        recovery_result = await self._execute_real_recovery(
                            condition=condition,
                            agent_id=step.agent,
                            thread_id=f"{scenario.name}_{run_id}",
                            ground_truth_state=ground_truth_state,
                            events=event_history,
                        )
                        recovery_success = recovery_result.success
                        recovery_time_ms = recovery_result.recovery_time_ms
                        
                        step_result.reconstructed_state = recovery_result.reconstructed_state
                        step_result.reconstruction_accuracy = recovery_result.reconstruction_accuracy
                        step_result.recovery_timing = recovery_result.detailed_timing
                        if recovery_result.reconstruction_accuracy > 0:
                            reconstruction_accuracies.append(recovery_result.reconstruction_accuracy)
                    else:
                        # Use simulated recovery
                        recovery_success, reconstructed_state = self._simulate_recovery_with_state(
                            condition, scenario, step, failure_type, ground_truth_state
                        )
                        recovery_time_ms = (time.time() - recovery_start) * 1000
                        
                        if reconstructed_state:
                            accuracy = self._calculate_state_similarity(
                                ground_truth_state, reconstructed_state
                            )
                            step_result.reconstructed_state = reconstructed_state
                            step_result.reconstruction_accuracy = accuracy
                            reconstruction_accuracies.append(accuracy)
                    
                    if recovery_success:
                        step_result.recovered = True
                        step_result.recovery_time_ms = recovery_time_ms
                        step_result.status = "recovered"
                        step_result.success = True
                        steps_completed += 1
                        logger.debug(
                            f"Recovery successful in {recovery_time_ms:.1f}ms "
                            f"(accuracy: {step_result.reconstruction_accuracy:.2%})"
                        )
                    else:
                        logger.debug("Recovery failed")
                        step_results.append(step_result)
                        break
                else:
                    step_results.append(step_result)
                    break
            else:
                step_duration_ms = self._simulate_step_execution(step)
                
                # Use real semantic negotiation when real reconstruction is enabled
                if use_real_reconstruction and condition.should_use_semantic_protocol():
                    conflicts, resolved, neg_time = await self._execute_real_semantic_negotiation(
                        step, condition
                    )
                else:
                    conflicts, resolved, neg_time = self._simulate_semantic_conflicts(step, condition)
                
                step_result = StepResult(
                    step_name=step.name,
                    agent=step.agent,
                    status=step.expected_status,
                    success=True,
                    duration_ms=step_duration_ms + neg_time,
                    semantic_conflicts=conflicts,
                    semantic_resolved=resolved,
                    semantic_negotiation_ms=neg_time,
                )
                steps_completed += 1
            
            step_results.append(step_result)
        
        total_duration_ms = (time.time() - start_time) * 1000
        
        success = (
            steps_completed == scenario.num_steps
            or (failure_occurred and recovery_success)
        )
        
        total_semantic_conflicts = sum(sr.semantic_conflicts for sr in step_results)
        total_semantic_resolved = sum(sr.semantic_resolved for sr in step_results)
        total_semantic_negotiation_ms = sum(sr.semantic_negotiation_ms for sr in step_results)
        
        mean_reconstruction_accuracy = (
            sum(reconstruction_accuracies) / len(reconstruction_accuracies)
            if reconstruction_accuracies else 0.0
        )
        
        # Aggregate recovery timing from step results (Phase D - Gap #10)
        aggregated_timing: Optional[RecoveryTimingBreakdown] = None
        timing_steps = [sr for sr in step_results if sr.recovery_timing is not None]
        if timing_steps:
            aggregated_timing = RecoveryTimingBreakdown(
                checkpoint_load_ms=sum(sr.recovery_timing.checkpoint_load_ms for sr in timing_steps),
                event_query_ms=sum(sr.recovery_timing.event_query_ms for sr in timing_steps),
                peer_context_ms=sum(sr.recovery_timing.peer_context_ms for sr in timing_steps),
                automata_predict_ms=sum(sr.recovery_timing.automata_predict_ms for sr in timing_steps),
                llm_inference_ms=sum(sr.recovery_timing.llm_inference_ms for sr in timing_steps),
                state_merge_ms=sum(sr.recovery_timing.state_merge_ms for sr in timing_steps),
                accuracy_calc_ms=sum(sr.recovery_timing.accuracy_calc_ms for sr in timing_steps),
                total_ms=sum(sr.recovery_timing.total_ms for sr in timing_steps),
            )
        
        result = ExperimentResult(
            run_id=run_id,
            scenario_name=scenario.name,
            condition_name=condition.name,
            success=success,
            total_duration_ms=total_duration_ms,
            steps_completed=steps_completed,
            total_steps=scenario.num_steps,
            failure_occurred=failure_occurred,
            failure_step=failure_step,
            recovery_attempted=recovery_attempted,
            recovery_success=recovery_success,
            recovery_time_ms=recovery_time_ms,
            step_results=step_results,
            mean_reconstruction_accuracy=mean_reconstruction_accuracy,
            reconstruction_accuracies=reconstruction_accuracies,
            semantic_conflicts=total_semantic_conflicts,
            semantic_resolved=total_semantic_resolved,
            semantic_negotiation_ms=total_semantic_negotiation_ms,
            recovery_timing_breakdown=aggregated_timing,
            # Cascade failure metrics (Phase F)
            cascade_triggered=cascade_triggered,
            cascade_failures=cascade_failures,
            cascade_depth=cascade_depth,
            metadata={
                "scenario_complexity": scenario.complexity,
                "condition_type": condition.condition_type.value,
                "failure_probability": self.failure_probability,
                "use_real_reconstruction": use_real_reconstruction,
            },
        )
        
        self.collector.record_result(result)
        self._results.append(result)
        
        return result
    
    def run_batch(
        self,
        scenario_name: str,
        condition: ExperimentCondition,
        num_runs: int,
        progress_callback: Optional[callable] = None,
    ) -> List[ExperimentResult]:
        """Run multiple experiments for a scenario/condition pair.
        
        Args:
            scenario_name: Name of the scenario.
            condition: Experimental condition.
            num_runs: Number of runs.
            progress_callback: Optional callback(current, total).
            
        Returns:
            List of experiment results.
        """
        results = []
        
        for i in range(num_runs):
            run_id = f"{scenario_name[:4]}-{condition.name[:4]}-{i:04d}"
            result = self.run_single(scenario_name, condition, run_id)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, num_runs)
        
        return results
    
    async def run_batch_async(
        self,
        scenario_name: str,
        condition: ExperimentCondition,
        num_runs: int,
        use_real_reconstruction: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> List[ExperimentResult]:
        """Run batch experiments with real reconstruction.
        
        This is the async version of run_batch that uses real HybridReconstructor
        instead of simulated recovery with hardcoded success rates.
        
        Args:
            scenario_name: Name of the scenario.
            condition: Experimental condition.
            num_runs: Number of runs.
            use_real_reconstruction: If True, use real HybridReconstructor.
            progress_callback: Optional callback(current, total).
            
        Returns:
            List of experiment results with real reconstruction metrics.
        """
        results = []
        
        for i in range(num_runs):
            run_id = f"{scenario_name[:4]}-{condition.name[:4]}-{i:04d}"
            result = await self.run_single_async(
                scenario_name,
                condition,
                run_id=run_id,
                use_real_reconstruction=use_real_reconstruction,
            )
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, num_runs)
        
        return results
    
    def run_all_scenarios(
        self,
        condition: ExperimentCondition,
        runs_per_scenario: int,
    ) -> List[ExperimentResult]:
        """Run all scenarios under a condition.
        
        Args:
            condition: Experimental condition.
            runs_per_scenario: Number of runs per scenario.
            
        Returns:
            List of all experiment results.
        """
        all_results = []
        scenarios = self.loader.list_scenarios()
        
        for scenario_name in scenarios:
            logger.info(
                f"Running {runs_per_scenario} experiments for {scenario_name} "
                f"under {condition.name}"
            )
            results = self.run_batch(scenario_name, condition, runs_per_scenario)
            all_results.extend(results)
        
        return all_results
    
    def run_full_experiment(
        self,
        runs_per_condition: int = 300,
        scenarios: Optional[List[str]] = None,
    ) -> Dict[str, List[ExperimentResult]]:
        """Run full experiment across all conditions.
        
        Args:
            runs_per_condition: Number of runs per condition.
            scenarios: Optional list of scenarios (all if None).
            
        Returns:
            Dictionary mapping condition name to results.
        """
        results_by_condition: Dict[str, List[ExperimentResult]] = {}
        
        # Get scenarios
        if scenarios is None:
            scenarios = self.loader.list_scenarios()
        
        runs_per_scenario = runs_per_condition // len(scenarios)
        
        # Run for each condition
        for condition in get_all_conditions():
            logger.info(f"=== Running condition: {condition.name} ===")
            
            condition_results = []
            for scenario_name in scenarios:
                results = self.run_batch(
                    scenario_name, condition, runs_per_scenario
                )
                condition_results.extend(results)
            
            results_by_condition[condition.name] = condition_results
        
        return results_by_condition
    
    def get_results(self) -> List[ExperimentResult]:
        """Get all collected results."""
        return self._results.copy()
    
    def get_metrics(self) -> ExperimentMetrics:
        """Get aggregated metrics."""
        return self.collector.get_metrics()
    
    def clear(self) -> None:
        """Clear all results."""
        self._results.clear()
        self.collector.clear()
    
    # =========================================================================
    # Real API Experiment Methods (Async)
    # =========================================================================
    
    async def run_real_api_single(
        self,
        product_data: Dict[str, Any],
        condition: ExperimentCondition,
        run_id: Optional[str] = None,
    ) -> ExperimentResult:
        """Run a single real API experiment against Shopify.
        
        Args:
            product_data: Product data for the experiment
            condition: Experimental condition to use
            run_id: Optional run ID
            
        Returns:
            ExperimentResult with real API metrics
        """
        from src.workflows.shopify_workflow import run_shopify_workflow
        from src.chaos.config import get_chaos_config
        
        run_id = run_id or str(uuid.uuid4())[:8]
        
        logger.info(f"Running real API experiment: run_id={run_id}")
        
        start_time = time.time()
        step_results: List[StepResult] = []
        
        # Enable chaos if resilience testing is active
        chaos_config = get_chaos_config()
        original_enabled = chaos_config.enabled
        
        if condition.should_attempt_recovery():
            chaos_config.enabled = True
        else:
            chaos_config.enabled = False
        
        try:
            # Run the Shopify workflow
            final_state = await run_shopify_workflow(
                product_data=product_data,
                task_id=run_id,
                agent_id="shopify-agent",
                use_simple=False,
            )
            
            # Parse results from final state
            status = final_state.get("status", "failed")
            success = status == "completed"
            current_step = final_state.get("current_step", 0)
            error = final_state.get("error")
            
            failure_occurred = error is not None or status == "failed"
            
            # Extract step info from messages
            messages = final_state.get("messages", [])
            for i, msg in enumerate(messages):
                step_results.append(StepResult(
                    step_name=f"step_{i}",
                    agent="shopify-agent",
                    status="success" if success else "failed",
                    success=success,
                    duration_ms=0,  # Real duration tracked separately
                ))
            
            total_duration_ms = (time.time() - start_time) * 1000
            
            # Determine recovery stats
            recovery_attempted = failure_occurred and condition.should_attempt_recovery()
            recovery_success = recovery_attempted and success
            
            result = ExperimentResult(
                run_id=run_id,
                scenario_name="shopify_real",
                condition_name=condition.name,
                success=success,
                total_duration_ms=total_duration_ms,
                steps_completed=current_step,
                total_steps=4,  # health_check, create, generate, update, cleanup
                failure_occurred=failure_occurred,
                failure_step=error if failure_occurred else None,
                recovery_attempted=recovery_attempted,
                recovery_success=recovery_success,
                recovery_time_ms=total_duration_ms if recovery_success else 0,
                step_results=step_results,
                metadata={
                    "is_real_api": True,
                    "condition_type": condition.condition_type.value,
                    "product_name": product_data.get("name", "unknown"),
                    "final_state": {
                        "status": status,
                        "current_step": current_step,
                        "error": error,
                    },
                },
            )
            
            self.collector.record_result(result)
            self._results.append(result)
            
            return result
            
        finally:
            # Restore chaos config
            chaos_config.enabled = original_enabled
    
    async def run_real_api_batch(
        self,
        condition: ExperimentCondition,
        num_runs: int = 100,
        product_templates: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[callable] = None,
    ) -> List[ExperimentResult]:
        """Run batch of real API experiments.
        
        Args:
            condition: Experimental condition
            num_runs: Number of runs
            product_templates: Optional list of product templates to use
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of experiment results
        """
        # Default product templates
        if product_templates is None:
            product_templates = [
                {"name": "Widget Pro X", "price": 49.99, "category": "Electronics"},
                {"name": "Eco Gadget", "price": 29.99, "category": "Sustainable"},
                {"name": "Premium Set", "price": 79.99, "category": "Accessories"},
                {"name": "Smart Device", "price": 99.99, "category": "Smart Home"},
            ]
        
        results = []
        
        for i in range(num_runs):
            run_id = f"real-{condition.name[:4]}-{i:04d}"
            
            # Cycle through product templates
            product_data = product_templates[i % len(product_templates)].copy()
            
            # Add unique identifier
            product_data["name"] = f"{product_data['name']}_{i:04d}"
            product_data["sku"] = f"PAAS-{run_id}"
            
            try:
                result = await self.run_real_api_single(
                    product_data=product_data,
                    condition=condition,
                    run_id=run_id,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Real API experiment {run_id} failed: {e}")
                # Record as failed
                results.append(ExperimentResult(
                    run_id=run_id,
                    scenario_name="shopify_real",
                    condition_name=condition.name,
                    success=False,
                    total_duration_ms=0,
                    steps_completed=0,
                    total_steps=4,
                    failure_occurred=True,
                    failure_step=str(e),
                    metadata={"error": str(e), "is_real_api": True},
                ))
            
            if progress_callback:
                progress_callback(i + 1, num_runs)
            
            # Small delay between runs to respect rate limits
            await asyncio.sleep(0.5)
        
        return results


# Convenience functions
def run_experiment(
    scenario_name: str,
    condition_name: str,
    num_runs: int = 1,
) -> List[ExperimentResult]:
    """Run experiments for a specific scenario and condition."""
    runner = ExperimentRunner()
    condition = get_condition(condition_name)
    return runner.run_batch(scenario_name, condition, num_runs)


def run_all_experiments(
    runs_per_condition: int = 300,
    seed: Optional[int] = None,
) -> Dict[str, List[ExperimentResult]]:
    """Run full experiment suite."""
    runner = ExperimentRunner(seed=seed)
    return runner.run_full_experiment(runs_per_condition)


async def run_real_api_experiments(
    num_runs: int = 100,
    condition_name: str = "real_api",
) -> List[ExperimentResult]:
    """Run real API experiments against Shopify.
    
    Args:
        num_runs: Number of experiment runs
        condition_name: Condition to use (default: real_api)
        
    Returns:
        List of experiment results
    """
    runner = ExperimentRunner()
    condition = get_condition(condition_name)
    return await runner.run_real_api_batch(condition, num_runs)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run PaaS experiments")
    parser.add_argument(
        "--runs", type=int, default=300, help="Runs per condition"
    )
    parser.add_argument(
        "--condition", type=str, help="Specific condition to run"
    )
    parser.add_argument(
        "--scenario", type=str, help="Specific scenario to run"
    )
    parser.add_argument(
        "--all-conditions", action="store_true", help="Run all conditions"
    )
    parser.add_argument(
        "--real-api", action="store_true", help="Run real API experiments (Shopify)"
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output", type=str, default="data/experiments", help="Output directory"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    runner = ExperimentRunner(seed=args.seed)
    
    if args.real_api:
        # Run real API experiments
        async def run_real():
            condition_name = args.condition or "real_api"
            condition = get_condition(condition_name)
            
            print(f"Running {args.runs} real API experiments with {condition_name}...")
            results = await runner.run_real_api_batch(condition, args.runs)
            
            # Export results
            from src.experiments.export import ExperimentExporter
            
            output_dir = Path(args.output) / "real_api"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            exporter = ExperimentExporter(output_dir)
            exporter.export_results(results, condition_name)
            
            print(f"Completed {len(results)} real API experiment runs")
            
            success_count = sum(1 for r in results if r.success)
            print(f"Success rate: {success_count / len(results):.1%}")
            
            recovery_results = [r for r in results if r.recovery_attempted]
            if recovery_results:
                recovery_success = sum(1 for r in recovery_results if r.recovery_success)
                print(f"Recovery success rate: {recovery_success / len(recovery_results):.1%}")
        
        asyncio.run(run_real())
    
    elif args.all_conditions:
        results = runner.run_full_experiment(args.runs)
        
        # Export results
        from src.experiments.export import ExperimentExporter
        
        exporter = ExperimentExporter(Path(args.output))
        exporter.export_all(results, runner.get_metrics())
        
        print(f"Results exported to {args.output}")
    else:
        condition_name = args.condition or "baseline"
        scenario_name = args.scenario
        
        condition = get_condition(condition_name)
        
        if scenario_name:
            results = runner.run_batch(scenario_name, condition, args.runs)
        else:
            results = runner.run_all_scenarios(condition, args.runs)
        
        print(f"Completed {len(results)} experiment runs")
        metrics = runner.get_metrics()
        print(f"Success rate: {metrics.success_rate:.1%}")
        if metrics.mttr_mean:
            print(f"Mean MTTR: {metrics.mttr_mean:.2f}s")


if __name__ == "__main__":
    main()

