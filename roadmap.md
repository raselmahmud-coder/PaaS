# Implementation Roadmap Architecture of Protocol-Aware Agentic Swarm (PaaS) for  E-Commerce Vendor

## Core Feature Priority Ranking (MVP-First Approach)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRIORITY HIERARCHY                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  P0 (Critical Path - MVP)         │  P1 (Enhanced MVP)              │
│  ────────────────────────          │  ────────────────              │
│  1. Basic Agent Framework          │  6. Advanced Reconstruction    │
│  2. State Persistence              │  7. Chaos Engineering          │
│  3. Simple Protocol (JSON)         │  8. Mind2Web Integration       │
│  4. Event Logging                  │  9. Metrics Dashboard          │
│  5. Basic Reconstruction           │ 10. Performance Optimization   │
│                                    │                                │
│  P2 (Research Novelty)            │  P3 (Optional Enhancements)     │
│  ──────────────────                │  ────────────────────          │
│  11. Semantic Negotiation          │  16. A/B Testing Framework     │
│  12. L* Automata Learning          │  17. Cost Optimization         │
│  13. Embedding Verification        │  18. Multi-tenant Support      │
│  14. MTTR-A Measurement            │  19. GUI Dashboard             │
│  15. Synthetic Scenario Gen        │  20. Production Hardening      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases (24 Weeks Total)

### **PHASE 0: Pre-Implementation Setup** (Week 0, 1-2 days)

**Goal:** Get environment ready before coding

**Activities:**
- Set up development machine with Python 3.11+
- Create GitHub repository with proper `.gitignore`
- Install Docker Desktop and set up local Kubernetes (Minikube or K3s)
- Create project structure skeleton
- Set up virtual environment

**Tools:**
```bash
# Core Development Tools
- Python 3.11+ (pyenv for version management)
- Git + GitHub (version control)
- VS Code / PyCharm (IDE)
- Docker Desktop (containerization)
- Minikube or K3s (local Kubernetes)
- PostgreSQL 15+ (via Docker)
- Redis 7+ (via Docker)
```

**Deliverable:** 
- Repository initialized
- `docker-compose.yml` with PostgreSQL + Redis
- `pyproject.toml` with initial dependencies

---

### **PHASE 1: Minimal Viable Agent System** (Weeks 1-4)

**Goal:** Get ONE agent working end-to-end with state persistence

#### Week 1: Foundation Infrastructure

**Tasks:**
1. Set up LangGraph with basic checkpointing
2. Implement PostgreSQL state persistence
3. Create a single "Product Upload" agent that can:
   - Receive a task (JSON input)
   - Execute 3 steps (with state updates after each)
   - Save checkpoints to PostgreSQL
   - Complete and return result

**Tools & Dependencies:**
```python
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.11"
langgraph = "^0.2.0"
langchain = "^0.3.0"
langchain-openai = "^0.2.0"
psycopg2-binary = "^2.9"
redis = "^5.0"
pydantic = "^2.5"
```

**Success Criterion:** 
- Agent completes task
- State visible in PostgreSQL at each checkpoint
- Can restart agent from last checkpoint

#### Week 2: Add Event Logging

**Tasks:**
1. Implement append-only event log in PostgreSQL
2. Log every agent action with:
   - `agent_id`, `timestamp`, `action_type`, `input`, `output`, `state_snapshot`
3. Create query interface to retrieve events by agent_id or time range

**Schema:**
```sql
CREATE TABLE agent_events (
    event_id SERIAL PRIMARY KEY,
    agent_id VARCHAR(100),
    timestamp TIMESTAMP,
    action_type VARCHAR(50),
    input_data JSONB,
    output_data JSONB,
    state_snapshot JSONB,
    INDEX idx_agent_timestamp (agent_id, timestamp)
);
```

**Success Criterion:** 
- Every action logged
- Can replay agent history from logs

#### Week 3: Multi-Agent Workflow

**Tasks:**
1. Add second agent: "Marketing Outreach"
2. Create LangGraph workflow:
   - Product Upload → Marketing Outreach
3. Implement handoff between agents
4. Add simple JSON message schema

**LangGraph Structure:**
```python
# Simple state graph
StateGraph([
    ("product_upload", product_agent_node),
    ("marketing", marketing_agent_node),
    ("product_upload", "marketing"),  # edge
])
```

**Success Criterion:** 
- Two agents communicate
- Workflow completes end-to-end
- All actions logged

#### Week 4: Basic State Reconstruction (MVP)

**Tasks:**
1. Implement failure detection (simulated timeout)
2. Build simple reconstruction:
   - Load last checkpoint
   - Query event log for actions since checkpoint
   - LLM call to infer "what should happen next"
   - Resume workflow from reconstructed state

**Reconstruction Logic:**
```python
def reconstruct_agent(agent_id):
    checkpoint = load_latest_checkpoint(agent_id)
    events = query_events_since(agent_id, checkpoint.timestamp)
    
    prompt = f"""
    Agent failed. Last checkpoint: {checkpoint}
    Actions since: {events}
    What should the agent do next?
    """
    
    next_action = llm.invoke(prompt)
    return spawn_agent(checkpoint, next_action)
```

**Success Criterion:** 
- Simulate agent crash
- Reconstruction module restores state
- Workflow resumes and completes

---

### **PHASE 2: Protocol Layer & Enhanced Reconstruction** (Weeks 5-8)

**Goal:** Add structured protocol and improve reconstruction quality

#### Week 5: Protocol Schema Definition

**Tasks:**
1. Define JSON Schema for agent messages (based on Analysis 1's schema)
2. Create message validator
3. Implement 4 message types:
   - `TASK_ASSIGN`
   - `TASK_COMPLETE`
   - `REQUEST_CONTEXT`
   - `PROVIDE_CONTEXT`

**Tools:**
```python
# Additional dependencies
jsonschema = "^4.20"
pydantic = "^2.5"  # For automatic validation
```

**Example Schema:**
```python
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

class MessageHeader(BaseModel):
    message_id: str
    message_type: Literal["TASK_ASSIGN", "TASK_COMPLETE", 
                          "REQUEST_CONTEXT", "PROVIDE_CONTEXT"]
    sender: str
    receiver: str
    timestamp: datetime
    protocol_version: str = "1.0"

class AgentMessage(BaseModel):
    header: MessageHeader
    body: dict
```

**Success Criterion:** 
- All inter-agent messages validated
- Invalid messages rejected with clear errors

#### Week 6: Message Broker Integration

**Tasks:**
1. Replace direct agent-to-agent calls with Kafka pub/sub
2. Agents publish to topics: `agent.{agent_id}.inbox`
3. Implement message routing layer
4. Log all messages to event log

**Tools:**
```python
# Dependencies
kafka-python = "^2.0"
# OR (lighter alternative)
aiokafka = "^0.10"
```

**Architecture:**
```
Agent A → Kafka Topic → Message Router → Agent B
              ↓
         Event Log (PostgreSQL)
```

**Success Criterion:** 
- Agents communicate via Kafka
- Messages persisted in Kafka + logged to PostgreSQL
- Can replay message stream

#### Week 7: Peer Context Retrieval

**Tasks:**
1. Implement `REQUEST_CONTEXT` protocol message
2. When agent fails, reconstruction module queries peer agents:
   - "What was your last interaction with agent X?"
3. Collect responses and feed to LLM

**Flow:**
```
Reconstruction Module
    ├─→ Load checkpoint
    ├─→ Query event log
    ├─→ Send REQUEST_CONTEXT to all agents that interacted with failed agent
    ├─→ Collect PROVIDE_CONTEXT responses
    └─→ LLM inference with all context
```

**Success Criterion:** 
- Peer agents respond to context requests
- Reconstruction uses peer data
- Improved reconstruction accuracy (measure manually)

#### Week 8: Checkpoint Optimization

**Tasks:**
1. Implement incremental checkpointing (not full state every time)
2. Add checkpoint compression
3. Tune checkpoint frequency (every N steps vs. every N seconds)
4. Benchmark checkpoint overhead

**Optimization:**
```python
# Only checkpoint on state changes
if current_state != last_checkpoint_state:
    save_checkpoint(current_state, delta=True)
```

**Success Criterion:** 
- Checkpoint overhead < 50ms per checkpoint
- Storage reduced by 60%+ with delta encoding

---

### **PHASE 3: Failure Injection & Chaos Engineering** (Weeks 9-12)

**Goal:** Build framework to test resilience systematically

#### Week 9: Fault Injection Decorators

**Tasks:**
1. Create Python decorators for fault injection
2. Implement 5 fault types:
   - `@inject_crash` - Raises exception randomly
   - `@inject_delay` - Adds latency
   - `@inject_hallucination` - Returns wrong LLM output
   - `@inject_timeout` - Simulates unresponsive agent
   - `@inject_message_corruption` - Corrupts message content

**Implementation:**
```python
import random
from functools import wraps

def inject_crash(probability=0.1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if random.random() < probability:
                raise AgentCrashException("Simulated crash")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@inject_crash(probability=0.2)
def agent_step(state):
    # agent logic
    pass
```

**Success Criterion:** 
- Decorators inject faults deterministically
- Fault injection configurable via environment variables

#### Week 10: Chaos Scenarios

**Tasks:**
1. Define 5 chaos scenarios for e-commerce:
   - **Scenario 1:** Product agent crashes during upload (50% progress)
   - **Scenario 2:** Marketing agent times out after receiving task
   - **Scenario 3:** Message corruption in handoff
   - **Scenario 4:** Network partition (agent isolated for 30s)
   - **Scenario 5:** Cascade failure (2 agents crash sequentially)

2. Create scenario runner:
```python
class ChaosScenario:
    def setup(self): pass
    def inject_fault(self): pass
    def validate_recovery(self): pass
    def teardown(self): pass
```

**Success Criterion:** 
- Each scenario runs repeatably
- Results logged (MTTR, success/failure)

#### Week 11: Kubernetes + LitmusChaos Integration

**Tasks:**
1. Containerize agents (Dockerfile)
2. Deploy to local Kubernetes cluster
3. Install LitmusChaos operator
4. Create 2 LitmusChaos experiments:
   - Pod kill (agent crash)
   - Network partition (agent isolation)

**Files:**
```yaml
# chaos-experiments/pod-kill.yaml
apiVersion: litmuschaos.io/v1alpha1
kind: ChaosEngine
metadata:
  name: product-agent-kill
spec:
  appinfo:
    appns: ecommerce-agents
    applabel: 'app=product-agent'
  chaosServiceAccount: litmus-admin
  experiments:
    - name: pod-delete
      spec:
        components:
          env:
            - name: TOTAL_CHAOS_DURATION
              value: '30'
```

**Tools:**
```bash
# Installation
kubectl apply -f https://litmuschaos.github.io/litmus/litmus-operator-v3.yaml
kubectl create ns ecommerce-agents
helm install litmus litmuschaos/litmus
```

**Success Criterion:** 
- Agents run in Kubernetes
- LitmusChaos kills pods successfully
- Reconstruction module detects and recovers

#### Week 12: Metrics Collection (MTTR-A)

**Tasks:**
1. Implement metrics collector based on Analysis 2's `ResilienceMetrics` class
2. Measure:
   - MTTR-A (Mean Time to Recovery - Agentic)
   - Task completion rate under failure
   - Reconstruction accuracy
3. Export to Prometheus

**Implementation:**
```python
from prometheus_client import Counter, Histogram, Gauge

mttr_metric = Histogram('agent_mttr_seconds', 
                         'Time to recover from failure')
recovery_success = Counter('agent_recovery_success_total',
                           'Successful recoveries')
reconstruction_accuracy = Gauge('agent_reconstruction_accuracy',
                                'Accuracy of state reconstruction')

# Usage
with mttr_metric.time():
    reconstruct_agent(failed_agent_id)
```

**Tools:**
```python
# Dependencies
prometheus-client = "^0.19"
```

**Success Criterion:** 
- Metrics visible in Prometheus
- MTTR < 15 seconds for basic scenarios

---

### **PHASE 4: Semantic Protocol & Advanced Reconstruction** (Weeks 13-16)

**Goal:** Implement novel academic contributions (Semantic negotiation + L* learning)

#### Week 13: Semantic Embeddings

**Tasks:**
1. Add embedding generation to protocol messages
2. Use Sentence-Transformers to embed key terms
3. Implement semantic similarity check:
   - When agent receives message, compare term embeddings
   - If similarity < threshold, trigger negotiation

**Tools:**
```python
# Dependencies
sentence-transformers = "^2.3"

# Code
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def validate_semantic_agreement(msg1, msg2, threshold=0.85):
    emb1 = model.encode(msg1.body["terms"])
    emb2 = model.encode(msg2.body["terms"])
    similarity = cosine_similarity(emb1, emb2)
    return similarity > threshold
```

**Success Criterion:** 
- Agents detect semantic misalignment
- Similarity scores logged for analysis

#### Week 14: Semantic Handshake Protocol

**Tasks:**
1. Implement 5-step handshake from Analysis 1:
   - `HANDSHAKE_INIT`
   - `HANDSHAKE_VERIFY`
   - `NEGOTIATE_TERM`
   - `TERM_ACCEPTED`
   - `HANDSHAKE_COMPLETE`

2. Add to protocol validator
3. Measure handshake overhead

**Flow:**
```
Agent A                        Agent B
   │ HANDSHAKE_INIT             │
   │ {terms: [t1, t2, t3]}      │
   ├────────────────────────────►│
   │                            │ (verify embeddings)
   │         HANDSHAKE_VERIFY   │
   │ {conflicts: [t2]}          │
   │◄────────────────────────────┤
   │ NEGOTIATE_TERM             │
   │ {t2: "new definition"}     │
   ├────────────────────────────►│
   │         TERM_ACCEPTED      │
   │◄────────────────────────────┤
   │ HANDSHAKE_COMPLETE         │
   ├────────────────────────────►│
```

**Success Criterion:** 
- Handshake completes successfully
- Semantic conflicts resolved
- Overhead < 500ms per handshake

#### Week 15: L* Automata Learning

**Tasks:**
1. Integrate AALpy library for L* algorithm
2. Train automaton from event log:
   - Input alphabet: agent message types
   - Output: agent actions
3. Use learned automaton for reconstruction predictions

**Tools:**
```python
# Dependencies
aalpy = "^1.4"

# Code
from aalpy import run_Lstar
from aalpy.SULs import DfaSUL
from aalpy.oracles import RandomWalkEqOracle

# Learn automaton from event log
events = query_agent_events(agent_id)
sul = create_sul_from_events(events)
learned_dfa = run_Lstar(sul, RandomWalkEqOracle(sul, 5000))
```

**Success Criterion:** 
- Automaton learned from 100+ events
- Can predict next action with >70% accuracy
- Used in hybrid reconstruction (L* + LLM)

#### Week 16: Hybrid Reconstruction

**Tasks:**
1. Combine all reconstruction techniques:
   - Load checkpoint
   - Query peer agents
   - Analyze event log
   - Run L* for structured prediction
   - Use LLM for unstructured reasoning
2. Implement decision tree from Analysis 2
3. Benchmark accuracy improvement

**Decision Logic:**
```python
def reconstruct(agent_id):
    checkpoint = load_checkpoint(agent_id)
    
    if checkpoint.age < 30:  # seconds
        return checkpoint.state
    
    events = query_events_since(checkpoint.timestamp)
    
    if len(events) > 100 and is_structured(events):
        # Use L* automata learning
        return reconstruct_with_automata(events, checkpoint)
    else:
        # Use LLM inference
        peer_context = query_peers(agent_id)
        return reconstruct_with_llm(checkpoint, events, peer_context)
```

**Success Criterion:** 
- Reconstruction accuracy > 90% (vs. ground truth)
- MTTR-A < 10 seconds

---

### **PHASE 5: Evaluation & Benchmarking** (Weeks 17-20)

**Goal:** Generate thesis experimental results

#### Week 17: Mind2Web Integration

**Tasks:**
1. Download Mind2Web dataset
2. Filter to e-commerce domains:
   - Shopping (Amazon, eBay)
   - E-commerce management
3. Create adapter to run Mind2Web tasks with your agents
4. Run baseline tests (no failures)

**Tools:**
```python
# Dependencies
playwright = "^1.40"  # For web automation
selenium = "^4.16"    # Alternative

# Dataset
# Download from: https://github.com/OSU-NLP-Group/Mind2Web
```

**Success Criterion:** 
- 10 Mind2Web e-commerce tasks run successfully
- Baseline success rate measured

#### Week 18: Synthetic Scenario Generator

**Tasks:**
1. Implement 4 scenario templates from Analysis 1:
   - `VENDOR_ONBOARDING`
   - `PRODUCT_LAUNCH_CAMPAIGN`
   - `CUSTOMER_FEEDBACK_LOOP`
   - `INVENTORY_CRISIS`

2. Parameterize scenarios:
   - Product count (1, 10, 100)
   - Complexity (simple, medium, complex)
   - Failure injection points (none, random, strategic)

**Example Template:**
```yaml
# scenarios/vendor_onboarding.yaml
name: Vendor Onboarding
steps:
  - name: register_vendor
    agent: registration_agent
    input:
      vendor_name: "{{vendor_name}}"
      email: "{{email}}"
  - name: upload_products
    agent: product_agent
    input:
      products: "{{products}}"
    failure_injection:
      type: crash
      probability: 0.2
      at_progress: 0.5
```

**Success Criterion:** 
- 20 synthetic scenarios generated
- Scenarios run with/without failures

#### Week 19: Comparison Experiments

**Tasks:**
1. Run 100 experiments across 3 conditions:
   - **Baseline:** No resilience (agent fails → workflow fails)
   - **With Reconstruction:** Your system
   - **With Reconstruction + Semantic Protocol:** Full system

2. Measure for each condition:
   - Task success rate
   - MTTR-A
   - Reconstruction accuracy
   - Protocol compliance rate
   - Cascade failure probability

**Experimental Design:**
```python
# experiments/config.yaml
experiments:
  - name: baseline
    resilience: false
    protocol: json
    runs: 100
    
  - name: with_reconstruction
    resilience: true
    protocol: json
    runs: 100
    
  - name: full_system
    resilience: true
    protocol: semantic
    runs: 100
```

**Success Criterion:** 
- Statistical significance (p < 0.05)
- Clear improvement in resilience metrics

#### Week 20: Results Analysis & Visualization

**Tasks:**
1. Aggregate experimental results
2. Create visualizations:
   - MTTR-A distribution (histogram)
   - Success rate comparison (bar chart)
   - Reconstruction accuracy over time (line chart)
   - Failure cascades (network graph)

3. Statistical analysis:
   - t-tests for comparing conditions
   - Confidence intervals

**Tools:**
```python
# Dependencies
pandas = "^2.1"
matplotlib = "^3.8"
seaborn = "^0.13"
scipy = "^1.11"
jupyter = "^1.0"
```

**Deliverable:**
- `results_analysis.ipynb` with all plots
- CSV files with raw data
- Statistical test results

---

### **PHASE 6: Thesis Writing & Finalization** (Weeks 21-24)

**Goal:** Complete thesis document and prepare defense

#### Week 21: System Documentation

**Tasks:**
1. Generate API documentation (Sphinx)
2. Write architecture document
3. Create deployment guide
4. Record demo video

**Tools:**
```python
# Dependencies
sphinx = "^7.2"
sphinx-rtd-theme = "^2.0"
```

**Structure:**
```
docs/
├── architecture.md
├── api_reference.rst
├── deployment_guide.md
├── experiment_reproduction.md
└── demo_walkthrough.md
```

#### Week 22: Thesis Chapters (Draft)

**Tasks:**
1. Write methodology chapters:
   - Chapter 4: Protocol Design (10-15 pages)
   - Chapter 5: Reconstruction Module (15-20 pages)
   - Chapter 6: Failure Injection Framework (10 pages)
   - Chapter 7: Evaluation (15-20 pages)

**Academic Contribution Highlights:**
- 14 fault types taxonomy (Table)
- 5-step semantic handshake protocol (Sequence diagram)
- Hybrid L*+LLM reconstruction algorithm (Algorithm pseudocode)
- MTTR-A measurement methodology (Equation + Python implementation)

#### Week 23: Results Chapter & Discussion

**Tasks:**
1. Write Chapter 8: Experimental Results
   - Present all comparison experiments
   - Include statistical analysis
   - Discuss limitations
   
2. Write Chapter 9: Discussion
   - Relate findings to research questions
   - Compare with related work
   - Discuss threats to validity

#### Week 24: Revision & Defense Prep

**Tasks:**
1. Revise entire thesis based on advisor feedback
2. Prepare defense slides (30-40 slides)
3. Practice presentation (20-30 minutes)
4. Prepare for Q&A

---

## Tool Selection Summary

### **Primary Stack (Recommended)**

| Layer | Tool | Alternative | Justification |
|-------|------|-------------|---------------|
| **Agent Framework** | **LangGraph** | AutoGen, CrewAI | Best state persistence, graph-based workflows |
| **LLM Provider** | **OpenAI GPT-4o** | Claude 3.5, Llama 3 | Best reasoning, widespread support |
| **State Store** | **PostgreSQL 15+** | Redis (hot), SQLite (dev) | ACID guarantees, queryable |
| **Message Broker** | **Apache Kafka** | RabbitMQ | Durable log, replay capability |
| **Chaos Tool** | **LitmusChaos** | Chaos Mesh, Toxiproxy | K8s-native, extensive fault library |
| **Observability** | **Prometheus + Grafana** | DataDog, New Relic | Open-source, widely used |
| **Tracing** | **OpenTelemetry + Jaeger** | Zipkin | Standard protocol |
| **Embeddings** | **Sentence-Transformers** | OpenAI Ada | Local inference, no API cost |
| **Automata Learning** | **AALpy** | LearnLib (Java) | Python-native, L* implementation |
| **Container Runtime** | **Docker** | Podman | Standard, good tooling |
| **Orchestration** | **Kubernetes (K3s)** | Docker Swarm | Industry standard, LitmusChaos compatible |
| **Experiment Tracking** | **MLflow** | Weights & Biases | Open-source, good for academic work |
| **Visualization** | **Matplotlib + Seaborn** | Plotly | Publication-quality plots |

### **Development Environment**

```yaml
# docker-compose.yml (for local development)
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: agent_system
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: agent123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7
    ports:
      - "6379:6379"
  
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
    ports:
      - "9092:9092"
    depends_on:
      - zookeeper
  
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
  
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"

volumes:
  postgres_data:
```

---

## Risk Mitigation Strategies

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **LLM API rate limits** | High | High | Use local Llama 3 as fallback; implement request queuing |
| **Reconstruction accuracy too low** | Medium | High | Start with simpler scenarios; use hybrid approach with multiple techniques |
| **Kubernetes complexity** | Medium | Medium | Start with docker-compose; migrate to K3s only in Phase 3 |
| **Mind2Web integration issues** | High | Medium | Focus on synthetic scenarios; Mind2Web as bonus evaluation |
| **Thesis timeline overrun** | Medium | High | Cut P2 features if needed; focus on P0 + P1 only |
| **AALpy/L* doesn't work** | Low | Medium | Use only LLM-based reconstruction; still novel contribution |
| **Semantic handshake overhead** | Low | Low | Make it optional; measure performance impact early |

---

## Academic Milestones & Checkpoints

| Week | Milestone | Thesis Progress | Advisor Meeting |
|------|-----------|-----------------|-----------------|
| **4** | MVP Demo | Intro + Lit Review complete | ✓ Show working agent |
| **8** | Protocol Implemented | Chapter 4 (Protocol Design) draft | ✓ Review protocol schema |
| **12** | Chaos Framework Done | Chapter 6 (Failure Framework) draft | ✓ Demo failure scenarios |
| **16** | Full System Working | Chapter 5 (Reconstruction) draft | ✓ Show reconstruction accuracy |
| **20** | Experiments Complete | Chapter 8 (Results) draft | ✓ Review statistical analysis |
| **24** | Thesis Submitted | Full draft complete | ✓ Defense prep |

---

## Deliverables Checklist

### Code Deliverables
- [ ] GitHub repository with clean README
- [ ] Docker containers for all services
- [ ] Kubernetes manifests for deployment
- [ ] 100+ unit tests (pytest)
- [ ] Integration test suite
- [ ] Chaos experiment definitions (5+ scenarios)
- [ ] API documentation (Sphinx)

### Data Deliverables
- [ ] Event log schema + sample data
- [ ] Protocol schema (JSON Schema files)
- [ ] Experimental results (CSV/JSON)
- [ ] Mind2Web e-commerce subset
- [ ] Synthetic scenario templates (YAML)

### Academic Deliverables
- [ ] Master's thesis (80-100 pages)
- [ ] Defense presentation (30-40 slides)
- [ ] Demo video (5-10 minutes)
- [ ] Conference paper draft (optional, for ICSE/FSE)
- [ ] Reproducibility package (Docker + scripts)

---

## Final Recommendation

**Start with Phase 1 (Weeks 1-4) immediately.** This gets you to a working MVP that can be demo'd to your advisor early. Everything else builds incrementally on this foundation.

**Focus on P0 + P1 features** (priorities 1-10). These are sufficient for a strong thesis. P2 features (11-15) add novelty but are optional if time runs short.

**Use LangGraph + PostgreSQL + OpenAI GPT-4o** as your core stack—this combination has the best documentation and community support for a 6-month thesis project.

**The key insight:** Build the simplest possible version first (Phases 1-2), then layer on complexity (Phases 3-4). This ensures you have *something working* even if you run out of time, which is critical for thesis success.