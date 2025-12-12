# Threats to Validity

This document outlines the potential threats to the validity of the experimental results presented in this thesis and the measures taken to mitigate them.

## 1. Internal Validity

Internal validity refers to the degree to which the observed results can be attributed to the independent variables rather than confounding factors.

### 1.1 Controlled Variables

**Controlled factors:**
- Random seed for reproducibility (seed=42 for all reported experiments)
- Consistent failure injection probability (30% across all scenarios)
- Identical hardware and software environment for all runs
- Same LLM model (GPT-4o-mini or Kimi-k2-turbo-preview) across conditions

**Potential threats:**
- **LLM non-determinism**: Even with temperature=0, LLM responses may vary slightly between runs.
  - *Mitigation*: Aggregated results over 300+ runs per condition to reduce variance impact.

- **Chaos injection randomness**: Failure injection uses probabilistic triggers.
  - *Mitigation*: Used deterministic random seeding and sufficient sample sizes.

### 1.2 Measurement Bias

**Potential threats:**
- **Timer precision**: System clock resolution affects MTTR measurements.
  - *Mitigation*: Used high-resolution timers; MTTR values are relative comparisons.

- **Simulation vs. Reality**: Synthetic scenarios may not capture all real-world complexity.
  - *Mitigation*: Added Shopify real API validation to cross-validate synthetic results.

## 2. External Validity

External validity refers to the generalizability of results beyond the experimental context.

### 2.1 Synthetic Scenarios

**Threat**: The four YAML-based scenarios (Vendor Onboarding, Product Launch, Customer Feedback, Inventory Crisis) are synthetic templates, not captured from real production systems.

**Limitations:**
- Scenarios may not fully represent real e-commerce workflow complexity
- Failure modes are idealized (crash, timeout, hallucination) rather than observed from production
- Workflow step durations are simulated, not measured from real operations

**Mitigations:**
1. **Shopify Real API Validation**: Added 100 real API experiments to validate synthetic results
2. **Expert Review**: Scenarios designed based on e-commerce domain knowledge
3. **Conservative Claims**: Paper explicitly states this is a "simulation study" when appropriate

### 2.2 Platform Specificity

**Threat**: Results are primarily validated against Shopify's API, which may not generalize to other e-commerce platforms (WooCommerce, Amazon, Shopee).

**Mitigations:**
1. **Abstraction Layer**: The integration uses abstract interfaces that could be implemented for other platforms
2. **Explicit Acknowledgment**: Thesis acknowledges platform-specific limitations

### 2.3 LLM Model Dependence

**Threat**: Results depend on specific LLM capabilities (GPT-4o-mini or Kimi-k2-turbo-preview). Different models may yield different reconstruction success rates.

**Mitigations:**
1. **Model Documentation**: Specific model versions and configurations are documented
2. **Reproducible Configuration**: All settings stored in `.env.example` for replication

## 3. Construct Validity

Construct validity refers to whether the measured metrics actually capture the intended concepts.

### 3.1 Metric Definitions

**MTTR-A (Mean Time to Recovery - Agentic)**
- *Definition*: Time from failure detection to successful state reconstruction
- *Potential Issue*: Only measures recovery time, not quality of recovered state
- *Mitigation*: Paired with Recovery Success Rate to assess recovery quality

**Task Success Rate**
- *Definition*: Proportion of workflows that complete all steps successfully
- *Potential Issue*: Binary metric doesn't capture partial success
- *Mitigation*: Also report steps completed / total steps for granularity

**Recovery Success Rate**
- *Definition*: Proportion of recovery attempts that result in successful workflow completion
- *Potential Issue*: Depends on definition of "successful completion"
- *Mitigation*: Explicitly defined as all remaining steps completing without failure

### 3.2 Failure Injection Fidelity

**Threat**: Simulated failures (crash, timeout, hallucination) may not accurately represent real failure modes.

**Mitigations:**
1. **Real API Validation**: Shopify experiments capture real network errors and API failures
2. **Chaos Decorator Design**: Decorators simulate realistic failure patterns based on distributed systems literature

## 4. Statistical Conclusion Validity

Statistical conclusion validity refers to the appropriateness of statistical analyses and significance of findings.

### 4.1 Sample Size Justification

**Design**: 300 runs per condition × 3 conditions = 900 total experiments (synthetic)
            100 runs × 1 condition = 100 total experiments (real API)

**Power Analysis**: With α=0.05, this sample size provides:
- >95% power to detect 10 percentage point differences in success rates
- >90% power to detect 0.1s differences in mean MTTR

### 4.2 Statistical Tests

**Chi-squared Test** for success rate comparisons:
- Appropriate for comparing proportions across conditions
- All expected cell counts > 5 (requirement met)

**Independent t-test** for MTTR comparisons:
- Assumes approximately normal distributions (verified via histograms)
- Welch's t-test used when variance equality uncertain

**Bonferroni Correction**: Applied for multiple comparisons (3 pairwise comparisons → α_adjusted = 0.05/3 = 0.017)

### 4.3 Effect Size Reporting

In addition to p-values, effect sizes are reported:
- **Cohen's h** for success rate differences
- **Cohen's d** for MTTR differences

This allows readers to assess practical significance beyond statistical significance.

## 5. Reproducibility

### 5.1 Reproducibility Measures

1. **Open Source**: All code available at [repository URL]
2. **Fixed Seeds**: Random seed (42) used for all reported experiments
3. **Configuration Files**: `.env.example` documents all settings
4. **Docker Support**: Containerized environment for consistent execution
5. **Version Pinning**: `pyproject.toml` specifies exact dependency versions

### 5.2 Known Reproducibility Limitations

1. **LLM API Costs**: Real experiments require API credits
2. **Shopify Account**: Real API validation requires developer account
3. **Kafka (Optional)**: Peer context features require Kafka broker
4. **Time-Dependent**: LLM model updates may affect results over time

## 6. Summary of Validity Threats

| Category | Primary Threat | Severity | Mitigation |
|----------|---------------|----------|------------|
| Internal | LLM non-determinism | Low | Large sample sizes |
| External | Synthetic scenarios | Medium | Real API validation |
| External | Platform specificity | Medium | Abstraction layer |
| Construct | Metric definitions | Low | Multiple complementary metrics |
| Statistical | Multiple comparisons | Low | Bonferroni correction |
| Reproducibility | LLM API changes | Medium | Version documentation |

## 7. Future Work to Address Limitations

1. **Multi-platform validation**: Extend real API validation to WooCommerce, Amazon
2. **Production log replay**: Capture and replay real production failure traces
3. **Multi-LLM comparison**: Compare reconstruction across GPT-4, Claude, Gemini
4. **Longitudinal study**: Track performance stability over multiple months
5. **User study**: Evaluate human-in-the-loop intervention quality

