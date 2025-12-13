#!/usr/bin/env python3
"""Statistical power analysis for thesis experiments.

This script justifies the sample size (300 runs per condition) used in the thesis
experiments by calculating:
1. Required sample size for detecting specified effect sizes
2. Statistical power achieved with current sample size
3. Confidence intervals for observed metrics

Usage:
    poetry run python scripts/power_analysis.py [--runs N] [--alpha A] [--power P]

References:
    - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    - Lakens, D. (2013). Calculating and reporting effect sizes.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Try to import statsmodels for advanced power analysis
try:
    from statsmodels.stats.power import TTestIndPower, NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Using scipy-only calculations.")


@dataclass
class PowerAnalysisResult:
    """Results from power analysis."""
    
    # For success rate comparison
    success_rate_required_n: int
    success_rate_power_at_300: float
    success_rate_detectable_effect: float
    
    # For MTTR comparison
    mttr_required_n: int
    mttr_power_at_300: float
    mttr_detectable_effect: float
    
    # Configuration
    alpha: float
    target_power: float
    current_n: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "success_rate": {
                "required_n": self.success_rate_required_n,
                "power_at_current_n": self.success_rate_power_at_300,
                "min_detectable_effect": self.success_rate_detectable_effect,
            },
            "mttr": {
                "required_n": self.mttr_required_n,
                "power_at_current_n": self.mttr_power_at_300,
                "min_detectable_effect": self.mttr_detectable_effect,
            },
            "config": {
                "alpha": self.alpha,
                "target_power": self.target_power,
                "current_n": self.current_n,
            }
        }


def calculate_cohens_h(p1: float, p2: float) -> float:
    """Calculate Cohen's h effect size for proportion comparison.
    
    Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    
    Interpretation:
        0.2 = small effect
        0.5 = medium effect
        0.8 = large effect
    """
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return abs(phi1 - phi2)


def calculate_cohens_d(mean1: float, mean2: float, sd_pooled: float) -> float:
    """Calculate Cohen's d effect size for mean comparison.
    
    Cohen's d = (mean1 - mean2) / sd_pooled
    
    Interpretation:
        0.2 = small effect
        0.5 = medium effect
        0.8 = large effect
    """
    if sd_pooled == 0:
        return 0.0
    return abs(mean1 - mean2) / sd_pooled


def sample_size_for_proportions(
    p1: float,
    p2: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate required sample size for two-proportion comparison.
    
    Uses the formula from Cohen (1988) for chi-squared test.
    
    Args:
        p1: Expected proportion in group 1.
        p2: Expected proportion in group 2.
        alpha: Significance level (default 0.05).
        power: Desired statistical power (default 0.80).
        
    Returns:
        Required sample size per group.
    """
    # Effect size (Cohen's h)
    h = calculate_cohens_h(p1, p2)
    
    if h == 0:
        return float('inf')
    
    # Z-values for alpha and power
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = stats.norm.ppf(power)
    
    # Sample size formula
    n = 2 * ((z_alpha + z_beta) / h) ** 2
    
    return int(math.ceil(n))


def sample_size_for_means(
    effect_size: float,
    sd: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Calculate required sample size for two-sample t-test.
    
    Args:
        effect_size: Expected difference in means.
        sd: Pooled standard deviation.
        alpha: Significance level.
        power: Desired statistical power.
        
    Returns:
        Required sample size per group.
    """
    # Convert to Cohen's d
    d = effect_size / sd if sd > 0 else 0
    
    if d == 0:
        return float('inf')
    
    # Z-values
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    # Sample size formula for two-sample t-test
    n = 2 * ((z_alpha + z_beta) / d) ** 2
    
    return int(math.ceil(n))


def calculate_power_for_proportions(
    p1: float,
    p2: float,
    n: int,
    alpha: float = 0.05,
) -> float:
    """Calculate statistical power for given sample size (proportions).
    
    Args:
        p1: Expected proportion in group 1.
        p2: Expected proportion in group 2.
        n: Sample size per group.
        alpha: Significance level.
        
    Returns:
        Statistical power (0-1).
    """
    h = calculate_cohens_h(p1, p2)
    
    if h == 0 or n == 0:
        return 0.0
    
    # Z-value for alpha
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    
    # Non-centrality parameter
    ncp = h * math.sqrt(n / 2)
    
    # Power = P(reject H0 | H1 true)
    power = 1 - stats.norm.cdf(z_alpha - ncp)
    
    return min(power, 1.0)


def calculate_power_for_means(
    effect_size: float,
    sd: float,
    n: int,
    alpha: float = 0.05,
) -> float:
    """Calculate statistical power for given sample size (means).
    
    Args:
        effect_size: Expected difference in means.
        sd: Pooled standard deviation.
        n: Sample size per group.
        alpha: Significance level.
        
    Returns:
        Statistical power (0-1).
    """
    d = effect_size / sd if sd > 0 else 0
    
    if d == 0 or n == 0:
        return 0.0
    
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    ncp = d * math.sqrt(n / 2)
    power = 1 - stats.norm.cdf(z_alpha - ncp)
    
    return min(power, 1.0)


def minimum_detectable_effect_proportion(
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
    baseline_p: float = 0.50,
) -> float:
    """Calculate minimum detectable effect size for proportions.
    
    Args:
        n: Sample size per group.
        alpha: Significance level.
        power: Desired power.
        baseline_p: Baseline proportion.
        
    Returns:
        Minimum detectable difference in proportions.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    # Minimum Cohen's h
    min_h = (z_alpha + z_beta) / math.sqrt(n / 2)
    
    # Convert Cohen's h back to proportion difference
    phi_baseline = 2 * math.asin(math.sqrt(baseline_p))
    phi_alt = phi_baseline + min_h
    
    if phi_alt > math.pi:
        phi_alt = math.pi
    if phi_alt < 0:
        phi_alt = 0
    
    p_alt = math.sin(phi_alt / 2) ** 2
    
    return abs(p_alt - baseline_p)


def minimum_detectable_effect_mean(
    n: int,
    sd: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Calculate minimum detectable effect size for means.
    
    Args:
        n: Sample size per group.
        sd: Pooled standard deviation.
        alpha: Significance level.
        power: Desired power.
        
    Returns:
        Minimum detectable difference in means.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    # Minimum Cohen's d
    min_d = (z_alpha + z_beta) / math.sqrt(n / 2)
    
    # Convert back to raw effect
    return min_d * sd


def confidence_interval_proportion(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Calculate confidence interval for a proportion (Wilson score).
    
    Args:
        successes: Number of successes.
        n: Total sample size.
        confidence: Confidence level (default 0.95).
        
    Returns:
        (lower, upper) bounds of confidence interval.
    """
    if n == 0:
        return (0.0, 1.0)
    
    p_hat = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    # Wilson score interval
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    
    return (lower, upper)


def confidence_interval_mean(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Calculate confidence interval for a mean.
    
    Args:
        values: List of observed values.
        confidence: Confidence level.
        
    Returns:
        (lower, upper) bounds of confidence interval.
    """
    if len(values) < 2:
        return (0.0, 0.0)
    
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    
    # t-distribution for small samples
    t_value = stats.t.ppf(1 - (1 - confidence) / 2, df=n-1)
    margin = t_value * se
    
    return (mean - margin, mean + margin)


def run_power_analysis(
    current_n: int = 300,
    alpha: float = 0.05,
    target_power: float = 0.80,
    success_rate_effect: float = 0.10,  # 10 percentage points
    mttr_effect: float = 0.05,  # 50ms difference
    mttr_sd: float = 0.10,  # 100ms standard deviation
) -> PowerAnalysisResult:
    """Run comprehensive power analysis.
    
    Args:
        current_n: Current sample size per condition.
        alpha: Significance level.
        target_power: Target statistical power.
        success_rate_effect: Expected difference in success rates.
        mttr_effect: Expected difference in MTTR (seconds).
        mttr_sd: Standard deviation of MTTR (seconds).
        
    Returns:
        PowerAnalysisResult with all calculations.
    """
    # Success rate analysis
    # Comparing baseline (~36%) vs full system (~95%)
    p1 = 0.36  # Baseline success rate
    p2 = 0.95  # Full system success rate
    
    success_required_n = sample_size_for_proportions(p1, p2, alpha, target_power)
    success_power = calculate_power_for_proportions(p1, p2, current_n, alpha)
    success_mde = minimum_detectable_effect_proportion(current_n, alpha, target_power)
    
    # MTTR analysis
    # Comparing baseline (no recovery) vs full system (~0.14s)
    mttr_required_n = sample_size_for_means(mttr_effect, mttr_sd, alpha, target_power)
    mttr_power = calculate_power_for_means(mttr_effect, mttr_sd, current_n, alpha)
    mttr_mde = minimum_detectable_effect_mean(current_n, mttr_sd, alpha, target_power)
    
    return PowerAnalysisResult(
        success_rate_required_n=success_required_n,
        success_rate_power_at_300=success_power,
        success_rate_detectable_effect=success_mde,
        mttr_required_n=mttr_required_n,
        mttr_power_at_300=mttr_power,
        mttr_detectable_effect=mttr_mde,
        alpha=alpha,
        target_power=target_power,
        current_n=current_n,
    )


def print_analysis_report(result: PowerAnalysisResult) -> None:
    """Print formatted power analysis report."""
    
    print("=" * 70)
    print("STATISTICAL POWER ANALYSIS - Thesis Experiment Justification")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  - Significance level (alpha): {result.alpha}")
    print(f"  - Target power: {result.target_power}")
    print(f"  - Current sample size: {result.current_n} runs per condition")
    
    print("\n" + "-" * 70)
    print("SUCCESS RATE COMPARISON")
    print("-" * 70)
    print(f"\nExpected effect: Baseline (~36%) vs Full System (~95%)")
    print(f"  Cohen's h effect size: {calculate_cohens_h(0.36, 0.95):.3f} (large)")
    print(f"\nRequired sample size for 80% power: {result.success_rate_required_n}")
    print(f"Power achieved with N={result.current_n}: {result.success_rate_power_at_300:.1%}")
    print(f"Minimum detectable effect: {result.success_rate_detectable_effect:.1%}")
    
    if result.success_rate_power_at_300 >= result.target_power:
        print(f"\n  SUFFICIENT: N={result.current_n} provides adequate power")
    else:
        print(f"\n  WARNING: N={result.current_n} may be underpowered")
    
    print("\n" + "-" * 70)
    print("MTTR COMPARISON")
    print("-" * 70)
    print(f"\nExpected effect: 50ms difference in recovery time")
    print(f"Assumed SD: 100ms")
    print(f"  Cohen's d effect size: {0.05 / 0.10:.2f} (medium)")
    print(f"\nRequired sample size for 80% power: {result.mttr_required_n}")
    print(f"Power achieved with N={result.current_n}: {result.mttr_power_at_300:.1%}")
    print(f"Minimum detectable effect: {result.mttr_detectable_effect * 1000:.1f}ms")
    
    if result.mttr_power_at_300 >= result.target_power:
        print(f"\n  SUFFICIENT: N={result.current_n} provides adequate power")
    else:
        print(f"\n  WARNING: N={result.current_n} may be underpowered")
    
    print("\n" + "-" * 70)
    print("CONFIDENCE INTERVALS (Example with N=300)")
    print("-" * 70)
    
    # Example: 95% success rate with 300 runs
    ci_95 = confidence_interval_proportion(int(0.95 * 300), 300)
    print(f"\nSuccess rate = 95% with N=300:")
    print(f"  95% CI: [{ci_95[0]:.1%}, {ci_95[1]:.1%}]")
    print(f"  Margin of error: +/- {(ci_95[1] - ci_95[0]) / 2 * 100:.1f} percentage points")
    
    # Example: 36% success rate with 300 runs
    ci_36 = confidence_interval_proportion(int(0.36 * 300), 300)
    print(f"\nSuccess rate = 36% with N=300:")
    print(f"  95% CI: [{ci_36[0]:.1%}, {ci_36[1]:.1%}]")
    print(f"  Margin of error: +/- {(ci_36[1] - ci_36[0]) / 2 * 100:.1f} percentage points")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    both_sufficient = (
        result.success_rate_power_at_300 >= result.target_power and
        result.mttr_power_at_300 >= result.target_power
    )
    
    if both_sufficient:
        print(f"""
The current sample size of {result.current_n} runs per condition is SUFFICIENT
to detect the expected effect sizes with {result.target_power:.0%} statistical power.

Key findings:
  - Success rate comparison: {result.success_rate_power_at_300:.1%} power (target: {result.target_power:.0%})
  - MTTR comparison: {result.mttr_power_at_300:.1%} power (target: {result.target_power:.0%})

This sample size can detect:
  - Success rate differences >= {result.success_rate_detectable_effect:.1%}
  - MTTR differences >= {result.mttr_detectable_effect * 1000:.1f}ms

The thesis experiments are adequately powered for statistical validity.
""")
    else:
        print(f"""
WARNING: The current sample size may be insufficient for some comparisons.

Recommendations:
  - Consider increasing sample size to at least {max(result.success_rate_required_n, result.mttr_required_n)}
  - Or acknowledge power limitations in the thesis
""")
    
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Statistical power analysis for thesis experiments"
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=300,
        help="Current sample size per condition (default: 300)"
    )
    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)"
    )
    parser.add_argument(
        "--power", "-p",
        type=float,
        default=0.80,
        help="Target statistical power (default: 0.80)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/power_analysis.json",
        help="Output file for results (default: data/power_analysis.json)"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    result = run_power_analysis(
        current_n=args.runs,
        alpha=args.alpha,
        target_power=args.power,
    )
    
    # Print report
    print_analysis_report(result)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Return exit code based on power sufficiency
    if (result.success_rate_power_at_300 >= args.power and 
        result.mttr_power_at_300 >= args.power):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

