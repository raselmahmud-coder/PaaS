### 2. **🔴 "Peer Context Retrieval is Stubbed/Non-Functional"**

**todo.md claims**: `_query_peer_context()` always returns `[]`.

**Reality check**:
- In `src/reconstruction/hybrid.py` lines 315-327, yes — the method has a stub that returns `[]`.
- **However**, you have `tests/test_peer_context_integration.py` with **551 lines of working Kafka integration tests**.
- These tests use `KafkaMessageProducer`, `KafkaMessageConsumer`, `AgentContextService` — all real implementations.
- All 10 Kafka integration tests **passed** during our debug session.

**Verdict**: ⚠️ **Partially valid.** The Kafka infrastructure exists and works, but `HybridReconstructor._query_peer_context()` doesn't actually use it — it's stubbed. The integration is **incomplete wiring**, not "phantom."

**Fix needed**: Wire `HybridReconstructor._query_peer_context()` to use `AgentContextService` from `src/messaging/agent_context_service.py`.

---

### 3. **🔴 "Experimental Results Are Largely Simulated"**

**todo.md claims**: `_simulate_recovery()` uses hardcoded success rates.

**Reality check**:
- Yes, `_simulate_recovery()` (lines 742-874) has hardcoded rates like `base_success_rate = 0.92` for hybrid.
- **But** the runner also has `_execute_real_recovery()` (line 1009) and `run_single_async(..., use_real_reconstruction=True)` (line 1167).
- The async path explicitly calls `HybridReconstructor` when `use_real_reconstruction=True`.

**Verdict**: ⚠️ **Partially valid.** The simulation mode IS the default and used in most reported results. The real reconstruction path EXISTS but isn't used in the main experiment reports.

**Fix needed**: Run experiments with `use_real_reconstruction=True` and report BOTH simulated (for scale/reproducibility) AND real (for validity) results.

---

### 4. **🟡 "Mind2Web Integration Absent"**

**todo.md claims**: Roadmap promised Mind2Web; it's not implemented.

**Reality check**: This is true. No Mind2Web code exists.

**Verdict**: ✅ **Valid critique**, but not critical. Mind2Web is a "nice to have" for external validity, not a core contribution. You can simply remove it from claims.

---

### 5. **🟡 "Semantic Negotiation Never Uses Real LLM"**

**todo.md claims**: `negotiator` is typically `None`, so negotiation is just string concatenation.

**Reality check**:
- `TermNegotiator` in `src/semantic/negotiator.py` lines 38-294 is a **full LLM-based implementation**.
- It has `NEGOTIATION_PROMPT`, calls the LLM, handles fallback strategies.
- `HandshakeManager` accepts a `negotiator` parameter and uses it when provided (line 481).

**Verdict**: ⚠️ **Partially valid.** The LLM negotiator exists and works. The issue is that **tests and experiments may not inject a real negotiator**. This is a test/experiment configuration issue, not a missing implementation.

---

### 6. **🟡 "MTTR-A Values are Simulated Sleep Times"**

**todo.md claims**: MTTR comes from `time.sleep()` calls.

**Reality check**:
- In simulation mode, yes — the `time.sleep()` creates artificial delays.
- In real reconstruction mode (`_execute_real_recovery`), the timing comes from actual LLM calls, Kafka queries, etc.

**Verdict**: ⚠️ **Partially valid.** Same as #3 — simulation mode is the default.

---

