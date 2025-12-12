## 2. What Needs Strengthening (Gaps)

### 2.1 Gap 1: Missing Comparison with Related Work

**Problem**: Your thesis claims novelty, but you haven't implemented or compared against existing approaches.

**What reviewers will ask**:
- "How does this compare to LangGraph's built-in checkpointing?"
- "What about AutoGen's error handling?"
- "Have you compared against simple retry policies?"

**Fix**: Add a simple baseline comparison:

```python
# Add to experiments:
class SimpleRetryCondition:
    """Baseline: Simple retry without reconstruction."""
    max_retries = 3
    
class CheckpointOnlyCondition:
    """Baseline: LangGraph checkpoint without LLM/automata."""
    use_reconstruction = False
```

This strengthens your claims by showing your approach outperforms simpler alternatives.

### 2.2 Gap 2: Limited Real-World Validation

**Problem**: All your experiments use **synthetic scenarios**. While controlled, this limits external validity.

**Current scenarios**:
- Vendor Onboarding (YAML template)
- Product Launch (YAML template)
- Customer Feedback (YAML template)
- Inventory Crisis (YAML template)

**What reviewers will ask**:
- "Would this work with real e-commerce APIs?"
- "What about real LLM hallucinations (not simulated)?"

**Fix options** (choose one):
1. **Add one real integration**: Connect to a real API (e.g., Shopify sandbox) for 50 runs
2. **Document clearly**: State this is a "simulation study" with explicit threats to validity
3. **Add real LLM failure cases**: Capture actual GPT-4 failures in logs and replay them

### 2.3 Gap 3: Missing Algorithm Formalization

**Problem**: Your hybrid reconstruction is implemented but not formally described.

**Current code** (informal):
```python
def reconstruct(self, ...):
    # Step 1: Check checkpoint freshness
    if self._is_checkpoint_fresh(checkpoint):
        return checkpoint
    # Step 2: Try automata
    # Step 3: Try LLM
    # Step 4: Combine
```

**What thesis needs** (formal):

```
Algorithm 1: Hybrid Reconstruction
─────────────────────────────────────────────────────────────
Input: agent_id, checkpoint C, events E
Output: reconstructed_state S

1: if age(C) < τ_checkpoint then
2:    return C with confidence = 1.0
3: end if
4: if |E| ≥ n_min then
5:    S_automata ← L*_predict(E)
6:    if conf(S_automata) ≥ τ_automata then
7:       return S_automata
8:    end if
9: end if
10: S_llm ← LLM_reconstruct(E, peer_context)
11: if S_automata ≠ ∅ then
12:    return combine(S_automata, S_llm)
13: end if
14: return S_llm
```

This makes your contribution defensible as a novel algorithm.

### 2.4 Gap 4: Semantic Handshake Not Used in Experiments

**Problem**: You implemented the semantic handshake (`HandshakeSession`) but your experiments don't measure its impact.

**Current**: Experiments compare "baseline" vs "reconstruction" vs "full_system"
**Missing**: No experiments isolating the semantic protocol's contribution

**Fix**: Add experiment condition:
```python
class SemanticOnlyCondition:
    """Tests semantic handshake without reconstruction."""
    enable_semantic = True
    enable_reconstruction = False
```

Or: Add a metric for "semantic conflicts resolved" to show the handshake adds value.

---
