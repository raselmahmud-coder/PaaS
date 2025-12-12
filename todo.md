## 🎯 Overview: Gap Priority Matrix

| Phase | Focus Area | Gaps Addressed | Effort | Timeline |
|-------|-----------|----------------|--------|----------|
| **Phase A** | Critical Validation | #1, #2, #4 | High | 3-4 days |
| **Phase B** | New Baselines | +3 baselines | Medium | 2 days |
| **Phase C** | Ablation Study | #3 | Medium | 1-2 days |
| **Phase D** | Real System Integration | #5, #6, #7, #10 | High | 3-4 days |
| **Phase E** | Statistical Rigor | #8, #9 | Medium | 2 days |
| **Phase F** | Scenario Enhancement | #11, #12 | Low | 1-2 days |
| | | **Total** | | **12-16 days** |

---

## 📌 Phase A: Critical Validation (Days 1-4)
**Priority: 🔴 CRITICAL**

### A1: Run ACTUAL Reconstructions (Gap #1)

**Current Problem**: Runner simulates recovery with hardcoded probabilities.

**Tasks**:
1. Modify `runner.py` to call actual `HybridReconstructor.reconstruct()`
2. Replace `time.sleep()` with real reconstruction timing
3. Capture actual success/failure outcomes

**Code Changes Required**:

```python
# In src/experiments/runner.py

async def _execute_real_recovery(
    self,
    condition: ExperimentCondition,
    scenario: Scenario,
    step: ScenarioStep,
    failure_type: str,
    current_state: Dict[str, Any],
) -> Tuple[bool, float, Dict[str, Any]]:
    """Execute ACTUAL recovery using the reconstruction module.
    
    Returns:
        Tuple of (success, recovery_time_ms, reconstructed_state)
    """
    from src.reconstruction.hybrid import HybridReconstructor
    
    start_time = time.time()
    
    # Create reconstructor based on condition
    reconstructor = HybridReconstructor(
        enable_automata=condition.should_use_automata(),
        enable_llm=condition.config.llm_fallback_enabled,
        enable_peer_context=condition.should_query_peers(),
    )
    
    try:
        # Run ACTUAL reconstruction
        result = await reconstructor.reconstruct(
            agent_id=step.agent,
            thread_id=f"{scenario.name}-{step.name}",
            checkpoint=current_state,
            events_since_checkpoint=[],  # Load from event store
        )
        
        recovery_time_ms = (time.time() - start_time) * 1000
        
        return (
            result.success,
            recovery_time_ms,
            result.reconstructed_state,
        )
        
    except Exception as e:
        logger.error(f"Real recovery failed: {e}")
        return (False, (time.time() - start_time) * 1000, {})
```

### A2: Ground Truth Comparison (Gap #2)

**Tasks**:
1. Save ground truth state during failure injection
2. Calculate state similarity after reconstruction
3. Add `reconstruction_accuracy` metric

**Code Changes Required**:

```python
# In src/experiments/runner.py

@dataclass
class StepResult:
    # ... existing fields ...
    
    # New ground truth fields
    ground_truth_state: Optional[Dict[str, Any]] = None
    reconstructed_state: Optional[Dict[str, Any]] = None
    reconstruction_accuracy: float = 0.0  # 0.0 to 1.0


def _calculate_state_similarity(
    self,
    ground_truth: Dict[str, Any],
    reconstructed: Dict[str, Any],
) -> float:
    """Calculate similarity between ground truth and reconstructed state.
    
    Uses field-by-field comparison with type-aware matching.
    """
    if not ground_truth or not reconstructed:
        return 0.0
    
    total_fields = len(ground_truth)
    matching_fields = 0
    
    for key, truth_value in ground_truth.items():
        if key in reconstructed:
            recon_value = reconstructed[key]
            
            # Type-aware comparison
            if isinstance(truth_value, (int, float)):
                # Numeric: allow 5% tolerance
                if abs(truth_value - recon_value) / max(abs(truth_value), 1) < 0.05:
                    matching_fields += 1
            elif isinstance(truth_value, str):
                # String: exact match or semantic similarity
                if truth_value == recon_value:
                    matching_fields += 1
            elif truth_value == recon_value:
                matching_fields += 1
    
    return matching_fields / total_fields if total_fields > 0 else 0.0
```

### A3: Fix RNG Circular Dependency (Gap #4)

**Tasks**:
1. Create separate RNG for semantic conflicts
2. Ensure independent random streams

**Code Changes Required**:

```python
# In src/experiments/runner.py

class ExperimentRunner:
    def __init__(self, ...):
        # ... existing code ...
        
        # Separate RNG streams for independence
        self._failure_rng = random.Random(seed)
        self._semantic_rng = random.Random(seed + 1000 if seed else None)
        self._recovery_rng = random.Random(seed + 2000 if seed else None)
```

---

## 📌 Phase B: New Baselines (Days 5-6)
**Priority: 🟠 HIGH**

### B1: Exponential Backoff Condition

```python
# Add to src/experiments/conditions.py

class ConditionType(Enum):
    # ... existing ...
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    SEMANTIC_ONLY = "semantic_only"


class ExponentialBackoffCondition(ExperimentCondition):
    """Exponential backoff retry: Industry-standard fault tolerance.
    
    Retries with exponential delays: 100ms, 200ms, 400ms, 800ms + jitter.
    No state reconstruction - just delayed retries.
    
    Literature: Google SRE Book, AWS Best Practices.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="exponential_backoff",
            condition_type=ConditionType.EXPONENTIAL_BACKOFF,
            resilience_enabled=True,
            semantic_protocol_enabled=False,
            automata_enabled=False,
            peer_context_enabled=False,
            description="Exponential backoff (4 retries, 100-800ms + jitter)",
            max_retries=4,
            use_checkpoint_restart=False,
            llm_fallback_enabled=False,
        )
    
    def get_retry_delays(self) -> List[float]:
        """Get retry delay sequence in seconds."""
        base = 0.1  # 100ms
        delays = []
        for i in range(4):
            delay = base * (2 ** i)  # 0.1, 0.2, 0.4, 0.8
            jitter = random.uniform(0, delay * 0.1)  # 10% jitter
            delays.append(delay + jitter)
        return delays
```

### B2: Circuit Breaker Condition

```python
# Add to src/experiments/conditions.py

class CircuitBreakerCondition(ExperimentCondition):
    """Circuit breaker pattern: Fail-fast on repeated failures.
    
    After N consecutive failures, circuit "opens" and rejects requests
    for a cooldown period. Prevents cascade failures.
    
    Literature: Nygard (2018) "Release It!", Netflix Hystrix.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="circuit_breaker",
            condition_type=ConditionType.CIRCUIT_BREAKER,
            resilience_enabled=True,
            semantic_protocol_enabled=False,
            automata_enabled=False,
            peer_context_enabled=False,
            description="Circuit breaker (5 failures → 30s open)",
            max_retries=0,
            use_checkpoint_restart=False,
            llm_fallback_enabled=False,
        )


class CircuitBreakerState:
    """Track circuit breaker state across experiments."""
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    
    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30):
        self.state = self.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_failure_time = 0
    
    def record_failure(self) -> str:
        """Record a failure and return new state."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = self.OPEN
        
        return self.state
    
    def record_success(self) -> str:
        """Record success and reset if in half-open."""
        if self.state == self.HALF_OPEN:
            self.state = self.CLOSED
            self.failure_count = 0
        return self.state
    
    def can_execute(self) -> bool:
        """Check if request can proceed."""
        if self.state == self.CLOSED:
            return True
        
        if self.state == self.OPEN:
            # Check if cooldown passed
            if time.time() - self.last_failure_time > self.cooldown_seconds:
                self.state = self.HALF_OPEN
                return True  # Allow one test request
            return False
        
        # HALF_OPEN: allow request
        return True
```

### B3: Semantic-Only Condition (Ablation)

```python
# Add to src/experiments/conditions.py

class SemanticOnlyCondition(ExperimentCondition):
    """Semantic-only condition: Ablation study for semantic protocol.
    
    Enables semantic handshake for term alignment but disables all
    recovery mechanisms. Tests whether semantic protocol alone
    prevents failures (rather than recovering from them).
    
    Purpose: Isolate semantic protocol contribution.
    """
    
    def _get_config(self) -> ConditionConfig:
        return ConditionConfig(
            name="semantic_only",
            condition_type=ConditionType.SEMANTIC_ONLY,
            resilience_enabled=False,  # No recovery!
            semantic_protocol_enabled=True,  # Only this enabled
            automata_enabled=False,
            peer_context_enabled=False,
            description="Semantic handshake only - prevents but doesn't recover",
            max_retries=0,
            use_checkpoint_restart=False,
            llm_fallback_enabled=False,
        )
```

### B4: Update Registry

```python
# Update CONDITION_REGISTRY in conditions.py

CONDITION_REGISTRY: Dict[str, Type[ExperimentCondition]] = {
    # ... existing ...
    "exponential_backoff": ExponentialBackoffCondition,
    "circuit_breaker": CircuitBreakerCondition,
    "semantic_only": SemanticOnlyCondition,
}
```

### B5: Update Runner for New Strategies

```python
# In src/experiments/runner.py - _simulate_recovery method

elif strategy == "exponential_backoff":
    # Exponential backoff: only works for transient failures
    condition_obj = condition  # Type hint
    if hasattr(condition_obj, 'get_retry_delays'):
        delays = condition_obj.get_retry_delays()
        for delay in delays:
            time.sleep(delay)
            # Each retry has 40% chance if transient
            if failure_type in ["timeout", "network_error"]:
                if self._recovery_rng.random() < 0.40:
                    return True
        return False
    base_success_rate = 0.40

elif strategy == "circuit_breaker":
    # Circuit breaker: fails fast after threshold
    # In real implementation, would check CircuitBreakerState
    base_success_rate = 0.45
    time.sleep(self._rng.uniform(0.01, 0.02))  # Fast fail
```

---

## 📌 Phase C: Ablation Study (Days 7-8)
**Priority: 🔴 CRITICAL (Gap #3)**

### C1: Design Ablation Experiment Matrix

| Experiment | Semantic | Automata | LLM | Peer | Purpose |
|------------|----------|----------|-----|------|---------|
| baseline | ❌ | ❌ | ❌ | ❌ | Control |
| semantic_only | ✅ | ❌ | ❌ | ❌ | Semantic contribution |
| automata_only | ❌ | ✅ | ❌ | ❌ | Automata contribution |
| llm_only | ❌ | ❌ | ✅ | ❌ | LLM contribution |
| llm_peer | ❌ | ❌ | ✅ | ✅ | Peer context contribution |
| full_no_semantic | ❌ | ✅ | ✅ | ✅ | Full minus semantic |
| full_system | ✅ | ✅ | ✅ | ✅ | Complete PaaS |

### C2: Add Component Contribution Metrics

```python
# In src/experiments/collector.py

@dataclass
class AblationMetrics:
    """Metrics for ablation study analysis."""
    
    # Component contributions (percentage points)
    semantic_contribution: float = 0.0
    automata_contribution: float = 0.0
    peer_context_contribution: float = 0.0
    llm_contribution: float = 0.0
    
    # Interaction effects
    semantic_automata_synergy: float = 0.0
    
    @classmethod
    def calculate(cls, metrics_by_condition: Dict[str, Dict]) -> "AblationMetrics":
        """Calculate contributions from ablation results."""
        
        full = metrics_by_condition.get("full_system", {}).get("success_rate", 0)
        
        # Component contributions
        no_semantic = metrics_by_condition.get("full_no_semantic", {}).get("success_rate", 0)
        no_automata = metrics_by_condition.get("llm_peer", {}).get("success_rate", 0)
        no_peer = metrics_by_condition.get("llm_only", {}).get("success_rate", 0)
        no_llm = metrics_by_condition.get("automata_only", {}).get("success_rate", 0)
        
        return cls(
            semantic_contribution=(full - no_semantic) * 100,
            automata_contribution=(full - no_automata) * 100,
            peer_context_contribution=(no_automata - no_peer) * 100,  # LLM+peer vs LLM
            llm_contribution=(full - no_llm) * 100,
        )
```

### C3: Update Notebook for Ablation Analysis

```python
# Add to notebooks/thesis_evaluation.ipynb - New Section 11

## 11. Ablation Study: Component Contributions

# Run ablation experiments
ablation_conditions = [
    'baseline', 'semantic_only', 'automata_only', 'llm_only', 
    'reconstruction', 'full_system'
]

ablation_results = {}
for cond_name in ablation_conditions:
    condition = get_condition(cond_name)
    results = runner.run_batch("vendor_onboarding", condition, num_runs=100)
    ablation_results[cond_name] = {
        'success_rate': sum(1 for r in results if r.success) / len(results),
        'recovery_rate': ...
    }

# Calculate component contributions
full_rate = ablation_results['full_system']['success_rate']
contributions = {
    'Semantic Protocol': full_rate - ablation_results.get('full_no_semantic', {}).get('success_rate', full_rate),
    'L* Automata': full_rate - ablation_results['reconstruction']['success_rate'],
    'Peer Context': ablation_results['reconstruction']['success_rate'] - ablation_results['llm_only']['success_rate'],
}

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
components = list(contributions.keys())
values = [v * 100 for v in contributions.values()]
ax.barh(components, values, color=['#3498db', '#2ecc71', '#e74c3c'])
ax.set_xlabel('Contribution to Success Rate (percentage points)')
ax.set_title('Component Contribution Analysis (Ablation Study)')
```

---

## 📌 Phase D: Real System Integration (Days 9-12)
**Priority: 🟠 HIGH (Gaps #5, #6, #7, #10)**

### D1: Acknowledge Real API Discrepancy (Gap #5)

**Task**: Update `docs/threats_to_validity.md` with prominent acknowledgment:

```markdown
## 2.4 Synthetic vs. Real API Performance Gap

**Critical Finding**: Real API experiments show:
- Success rate: 86% (vs. 95% synthetic)
- MTTR: 22.4s (vs. 0.14s synthetic)

**Explanation**: The 160x MTTR difference reflects:
1. Real network latency (~100-500ms per API call)
2. Shopify API response time (~200-1000ms)
3. LLM inference time (~1-5s per call)
4. No parallelization in current implementation

**Implication**: Synthetic results represent *relative* comparisons 
between conditions. Absolute values should be interpreted with caution.
Real-world deployment would require optimization.
```

### D2: Run ACTUAL L* Learning (Gap #6)

**Tasks**:
1. Generate event log from synthetic runs
2. Run actual AALpy L* learning
3. Output learned automaton visualization
4. Report prediction accuracy

```python
# scripts/run_automata_learning.py

from src.automata.learner import AutomataLearner
from src.persistence.event_store import EventStore

def run_actual_lstar():
    """Run actual L* learning and report results."""
    
    # Load events from database
    event_store = EventStore()
    events = event_store.get_events_for_agent("product-agent", limit=500)
    
    # Create learner
    learner = AutomataLearner()
    
    # Run L* learning
    result = learner.learn(events)
    
    print(f"Learning Result:")
    print(f"  States: {result.num_states}")
    print(f"  Transitions: {result.num_transitions}")
    print(f"  Queries used: {result.num_queries}")
    
    # Test prediction accuracy on held-out data
    test_events = event_store.get_events_for_agent("product-agent", offset=500, limit=100)
    correct = 0
    for i, event in enumerate(test_events[:-1]):
        predicted = learner.predict_next(event)
        actual = test_events[i + 1]
        if predicted.action == actual.action_type:
            correct += 1
    
    accuracy = correct / (len(test_events) - 1)
    print(f"  Prediction accuracy: {accuracy:.1%}")
    
    # Export automaton visualization
    learner.export_dot("data/learned_automaton.dot")
    print(f"  Automaton saved to: data/learned_automaton.dot")

if __name__ == "__main__":
    run_actual_lstar()
```

### D3: Demonstrate Peer Context (Gap #7)

**Task**: Add integration test showing Kafka peer context working:

```python
# tests/test_peer_context_integration.py

@pytest.mark.integration
async def test_peer_context_retrieval_end_to_end():
    """Test actual peer context retrieval via Kafka."""
    
    # Start Kafka (requires docker)
    # This test requires: docker-compose up kafka
    
    from src.messaging.producer import AgentEventProducer
    from src.messaging.agent_context_service import AgentContextService
    
    # Setup
    producer = AgentEventProducer()
    context_service = AgentContextService()
    
    # Agent A publishes interaction with Agent B
    await producer.publish_event({
        "agent_id": "agent-A",
        "action": "handoff",
        "target_agent": "agent-B",
        "state": {"task_id": "test-123", "status": "in_progress"},
        "timestamp": datetime.now().isoformat(),
    })
    
    # Agent B fails - reconstruction queries peers
    peer_context = await context_service.get_peer_context(
        failed_agent_id="agent-B",
        timeout_seconds=5.0,
    )
    
    # Verify peer context received
    assert len(peer_context) >= 1
    assert peer_context[0]["source_agent"] == "agent-A"
    assert peer_context[0]["state"]["task_id"] == "test-123"
    
    print(f"Peer context latency: {peer_context[0]['latency_ms']:.1f}ms")
```

### D4: Fix MTTR Definition (Gap #10)

**Task**: Measure actual operation times, not sleep():

```python
# In src/experiments/runner.py

@dataclass
class RecoveryTimingBreakdown:
    """Detailed timing for recovery operations."""
    
    checkpoint_load_ms: float = 0.0
    event_query_ms: float = 0.0
    peer_context_ms: float = 0.0
    automata_predict_ms: float = 0.0
    llm_inference_ms: float = 0.0
    total_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "checkpoint_load_ms": self.checkpoint_load_ms,
            "event_query_ms": self.event_query_ms,
            "peer_context_ms": self.peer_context_ms,
            "automata_predict_ms": self.automata_predict_ms,
            "llm_inference_ms": self.llm_inference_ms,
            "total_ms": self.total_ms,
        }
```

---

## 📌 Phase E: Statistical Rigor (Days 13-14)
**Priority: 🟠 HIGH (Gaps #8, #9)**

### E1: Increase Sample Size (Gap #8)

**Recommendation**: 
- Increase to 500 runs per condition (from 300)
- Or: Run power analysis to justify current size

```python
# scripts/power_analysis.py

from scipy import stats
import numpy as np

def calculate_required_sample_size(
    effect_size: float = 0.10,  # 10 percentage point difference
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate required sample size for chi-squared test."""
    
    from statsmodels.stats.power import GofChisquarePower
    
    analysis = GofChisquarePower()
    n = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
    )
    
    return int(np.ceil(n))

# For 10% effect size detection with 80% power
required_n = calculate_required_sample_size(0.10, 0.05, 0.80)
print(f"Required sample size per condition: {required_n}")
```

### E2: Add Modern Framework Comparison Note (Gap #9)

**Task**: Add to related work documentation:

```markdown
## Comparison with Modern Frameworks

**Why direct comparison is challenging:**

1. **AutoGen (Microsoft)**: Focuses on conversation recovery, not workflow state.
   No equivalent to checkpoint + LLM reconstruction.
   
2. **CrewAI**: Uses task delegation, not explicit fault tolerance.
   Recovery is implicit through task reassignment.
   
3. **LangGraph**: Provides checkpointing (our checkpoint_only baseline)
   but no intelligent reconstruction.

**Our contribution**: PaaS is the first to combine:
- Formal methods (L* automata)
- LLM reasoning
- Semantic alignment protocol
- Peer context retrieval

No existing framework offers this combination for comparison.
```

---

## 📌 Phase F: Scenario Enhancement (Days 15-16)
**Priority: 🟡 MEDIUM (Gaps #11, #12)**

### F1: Add Parallel Execution Scenario (Gap #11)

```yaml
# scenarios/parallel_agents.yaml
name: Parallel Product Upload
description: Multiple agents working simultaneously
complexity: high

agents:
  - id: product-agent-1
    type: product_upload
  - id: product-agent-2
    type: product_upload
  - id: coordinator
    type: coordinator

steps:
  - name: split_batch
    agent: coordinator
    action: split
    parallel_targets: [product-agent-1, product-agent-2]
    
  - name: upload_batch_1
    agent: product-agent-1
    action: upload
    parallel: true
    
  - name: upload_batch_2
    agent: product-agent-2
    action: upload
    parallel: true
    
  - name: aggregate_results
    agent: coordinator
    action: aggregate
    wait_for: [upload_batch_1, upload_batch_2]
```

### F2: Add Cascade Failure Scenario

```yaml
# scenarios/cascade_failure.yaml
name: Cascade Failure Test
description: Test recovery when multiple agents fail sequentially
complexity: high

failure_injection:
  enabled: true
  cascade:
    - step: 2
      probability: 0.5
      causes_downstream: true  # Triggers failure in dependent agents
```

---

## 📅 Timeline Summary

```
Week 1: Phase A (Critical Validation)
├── Day 1-2: Real reconstruction integration
├── Day 3: Ground truth comparison
└── Day 4: RNG independence fix

Week 2: Phase B + C (Baselines + Ablation)
├── Day 5: Exponential backoff + Circuit breaker
├── Day 6: Semantic-only + Registry updates
├── Day 7: Ablation experiment design
└── Day 8: Run ablation experiments

Week 3: Phase D (Real Integration)
├── Day 9: Document real API discrepancy
├── Day 10: Run actual L* learning
├── Day 11: Peer context demonstration
└── Day 12: MTTR timing breakdown

Week 4: Phase E + F (Statistical + Scenarios)
├── Day 13: Power analysis / sample size
├── Day 14: Framework comparison notes
├── Day 15: Parallel scenario
└── Day 16: Cascade failure scenario
```

---

## ✅ Completion Checklist

### Phase A: Critical Validation
- [ ] Real reconstruction in runner.py
- [ ] Ground truth state capture
- [ ] State similarity calculation
- [ ] Separate RNG streams

### Phase B: New Baselines
- [ ] ExponentialBackoffCondition
- [ ] CircuitBreakerCondition
- [ ] CircuitBreakerState class
- [ ] SemanticOnlyCondition
- [ ] Update CONDITION_REGISTRY
- [ ] Update runner strategies

### Phase C: Ablation Study
- [ ] Design experiment matrix
- [ ] AblationMetrics class
- [ ] Notebook Section 11
- [ ] Component contribution chart

### Phase D: Real Integration
- [ ] Update threats_to_validity.md
- [ ] run_automata_learning.py script
- [ ] Automaton visualization
- [ ] Peer context integration test
- [ ] RecoveryTimingBreakdown

### Phase E: Statistical Rigor
- [ ] Power analysis script
- [ ] Sample size justification
- [ ] Framework comparison doc

### Phase F: Scenario Enhancement
- [ ] parallel_agents.yaml
- [ ] cascade_failure.yaml

---